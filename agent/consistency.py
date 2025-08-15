# ppo_cream_train.py
# PPO + CREAM (online) for RAG, mirroring your offline math.
# - Reward: avg log-likelihood over response tokens only (length-normalized)
# - Consistency: Spearman/Kendall between policy-vs-ref reward lists on K candidates (scaled to [0,1])
# - DPO-style delta optional: policy_reward - ref_reward for the chosen response
#
# Requirements:
#   pip install torch transformers trl scipy datasets accelerate
#
# Notes:
# - If VRAM is tight, set load_in_8bit=True (bitsandbytes needed) or use device_map="auto".
# - Keep K small (e.g., 2–4) to control compute.
# - Plug in your own retriever where indicated.

from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple, Dict
import math
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import PPOConfig, PPOTrainer

from scipy.stats import spearmanr, kendalltau, rankdata

# ----------------------------
# Config
# ----------------------------

@dataclass
class TrainConfig:
    policy_model_name: str = "gpt2"
    ref_model_name: Optional[str] = None         # if None, no DPO delta; consistency still requires ref to be meaningful
    max_input_len: int = 2048
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    do_sample: bool = True
    num_candidates: int = 3                      # K candidates per prompt (controls consistency computation)
    lr: float = 1e-5
    batch_size: int = 1
    grad_accum: int = 4
    seed: int = 42

    # Consistency settings
    consistency_method: Literal["spearman", "kendall", "toporder"] = "spearman"
    lambda_consistency: float = 0.5              # λ in reward_total = dpo_part + λ * consistency

    # DPO-style reward combination
    use_dpo_delta: bool = True                   # if True, reward_part = policy - ref; else reward_part = policy

    # Device/precision helpers
    device_map: Optional[str] = "auto"           # "auto" or None
    load_in_8bit: bool = False                   # set True if you rely on 8-bit quantization

# ----------------------------
# Utilities
# ----------------------------

def to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    return obj

def format_prompt(query: str, docs: List[str]) -> str:
    # Replace this with your own prompt template used in training/inference
    context = "\n".join([f"[DOC{i+1}] {d}" for i, d in enumerate(docs)])
    return f"{context}\n\nQ: {query}\nA:"

@torch.no_grad()
def generate_candidates(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    max_input_len: int,
) -> List[str]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_len)
    input_ids = inputs["input_ids"].to(model.device)
    attn = inputs.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    if top_k and top_k > 0:
        gen_kwargs["top_k"] = top_k

    candidates = []
    for _ in range(num_candidates):
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            **gen_kwargs
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        # Extract only the assistant response span (after prompt)
        # A lightweight splitter assuming prompt is suffix:
        if text.startswith(tokenizer.decode(input_ids[0], skip_special_tokens=True)):
            resp = text[len(tokenizer.decode(input_ids[0], skip_special_tokens=True)):]
        else:
            # Fallback: take tail tokens as "response" (heuristic)
            resp = text[-(max_new_tokens*4):]
        candidates.append(resp.strip())
    return candidates

def _avg_ll_response_span(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    response: str,
    max_len: int,
    normalize_by_length: bool = True,
) -> float:
    """
    Average log-likelihood over response tokens only (mirrors your reward file logic):
    1) Tokenize prompt (prefix) and prompt+response (full)
    2) Compute per-token logp on full, mask out prefix tokens
    3) Average over response positions
    """
    model.eval()
    with torch.no_grad():
        # Tokenize prompt only to get prefix length in tokens
        prefix = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
        prefix_ids = prefix["input_ids"][0]

        # Tokenize prompt+response
        full = tokenizer(
            prompt + response,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        )
        input_ids = full["input_ids"].to(model.device)        # [1, L]
        attn = full.get("attention_mask", None)
        if attn is not None:
            attn = attn.to(model.device)

        # Forward
        logits = model(input_ids=input_ids, attention_mask=attn).logits  # [1, L, V]

        # Shift for next-token prediction
        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]  # [1, L-1]

        # Build loss mask to keep only response tokens
        prefix_len = prefix_ids.shape[0]
        # position index 0 in labels corresponds to token 1 in full sequence
        # response positions start at index (prefix_len - 1) in labels
        mask = torch.zeros_like(labels, dtype=torch.float32)
        start = max(prefix_len - 1, 0)
        mask[:, start:] = 1.0

        # per-token log prob
        log_probs = F.log_softmax(logits, dim=-1)
        per_tok = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        masked = per_tok * mask

        total_logp = masked.sum(dim=-1)  # [1]
        denom = mask.sum(dim=-1).clamp_min(1.0) if normalize_by_length else 1.0
        avg = (total_logp / denom)[0].item()
        return avg

def score_candidates_ll(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    candidates: List[str],
    max_len: int,
    normalize_by_length: bool = True,
) -> List[float]:
    return [
        _avg_ll_response_span(model, tokenizer, prompt, resp, max_len, normalize_by_length)
        for resp in candidates
    ]

def calc_consistency(
    scores1: List[float],
    scores2: List[float],
    method: Literal["spearman", "kendall", "toporder"] = "spearman",
) -> float:
    """
    EXACTLY mirrors your offline script:
      - ranks via rankdata(method='ordinal')
      - Spearman or Kendall correlation
      - scale from [-1,1] to [0,1]
      - 'toporder' checks matching top & bottom
    """
    if len(scores1) != len(scores2):
        raise ValueError("Score lists must have same length")
    if len(scores1) == 0:
        return 0.0
    if any(s is None for s in scores1) or any(s is None for s in scores2):
        return 0.0

    r1 = rankdata(scores1, method='ordinal')
    r2 = rankdata(scores2, method='ordinal')

    if method == "kendall":
        val, _ = kendalltau(r1, r2)
        val = 0.0 if (val is None or math.isnan(val)) else (val + 1.0)/2.0
        return float(val)
    elif method == "spearman":
        val, _ = spearmanr(r1, r2)
        val = 0.0 if (val is None or math.isnan(val)) else (val + 1.0)/2.0
        return float(val)
    elif method == "toporder":
        top1 = int(max(range(len(scores1)), key=lambda i: scores1[i]))
        bot1 = int(min(range(len(scores1)), key=lambda i: scores1[i]))
        top2 = int(max(range(len(scores2)), key=lambda i: scores2[i]))
        bot2 = int(min(range(len(scores2)), key=lambda i: scores2[i]))
        return 1.0 if (top1 == top2 and bot1 == bot2) else 0.0
    else:
        raise ValueError(f"Unsupported consistency method: {method}")

