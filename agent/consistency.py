from scipy.stats import spearmanr, rankdata, kendalltau
import statistics
import numpy as np
import os
from CREAM.tools import tools_json_load, tools_log_on_rank, tools_json_dump, tools_get_time
from CREAM.config import ConsistencyMethodEnum, CalConsistencyConfig, parse_args
import torch
import torch.nn.functional as F


class ConsistencyEvaluator:
    """Evaluates consistency between two reward model outputs for RAG cream model.

    Notes:
      - The constructor accepts policy_model/ref_model/tokenizer for API compatibility
        with training pipelines that pass them in, but they are not required for
        file-based consistency evaluation.
    """
    def __init__(
        self,
        policy_model=None,
        ref_model=None,
        tokenizer=None,
        method: ConsistencyMethodEnum = ConsistencyMethodEnum.spearman,
        **kwargs,
    ):
        self.method = method
        # kept for compatibility; not used in this file-based evaluator
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer

    def compute(self, prompt: str, response: str, max_len: int = 1024) -> torch.Tensor:
        """
        KL(policy || reference) over generated response tokens, conditioned on prompt.
        Handles models on different devices.
        """
        if self.policy_model is None or self.ref_model is None or self.tokenizer is None:
            # no-op if not wired
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            return torch.tensor(0.0, device=dev)

        # Identify devices
        pol_dev = next(self.policy_model.parameters()).device
        ref_dev = next(self.ref_model.parameters()).device

        # Tokenize ON CPU (neutral), then copy to each device separately
        full_text = f"{prompt}{response}"
        enc_full = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_len)
        enc_resp = self.tokenizer(response,  return_tensors="pt", truncation=True, max_length=max_len)

        # after enc_full/enc_resp
        input_ids_cpu = enc_full["input_ids"].to(dtype=torch.long)
        attn_mask_cpu = enc_full.get("attention_mask", None)
        if attn_mask_cpu is not None:
            attn_mask_cpu = attn_mask_cpu.to(dtype=torch.long)

        input_ids_cpu = enc_full["input_ids"]                # [1, L] on CPU
        attn_mask_cpu = enc_full.get("attention_mask", None)
        resp_len = int(enc_resp["input_ids"].shape[1])
        if resp_len == 0 or input_ids_cpu.shape[1] < resp_len + 1:
            return torch.tensor(0.0, device=pol_dev)

        # Debug prints BEFORE moving to device
        vocab_size = self.tokenizer.vocab_size
        print("[DEBUG] input_ids_cpu shape:", input_ids_cpu.shape)
        print("[DEBUG] input_ids_cpu dtype:", input_ids_cpu.dtype)
        print("[DEBUG] input_ids_cpu values (first 50):", input_ids_cpu.flatten()[:50])
        print("[DEBUG] input_ids_cpu min:", input_ids_cpu.min().item())
        print("[DEBUG] input_ids_cpu max:", input_ids_cpu.max().item())
        print("[DEBUG] vocab_size:", vocab_size)
        invalid_mask = (input_ids_cpu < 0) | (input_ids_cpu >= vocab_size)
        if invalid_mask.any():
            print("[ERROR] Found out-of-range token ids in input_ids_cpu:", input_ids_cpu[invalid_mask])

        # Clone to each model's device
        pol_inputs = {
            "input_ids": input_ids_cpu.to(pol_dev, non_blocking=True),
            "attention_mask": attn_mask_cpu.to(pol_dev, non_blocking=True) if attn_mask_cpu is not None else None,
        }
        ref_inputs = {
            "input_ids": input_ids_cpu.to(ref_dev, non_blocking=True),
            "attention_mask": attn_mask_cpu.to(ref_dev, non_blocking=True) if attn_mask_cpu is not None else None,
        }
        if invalid_mask.any():
            print("[ERROR] Found out-of-range token ids in input_ids_cpu:", input_ids_cpu[invalid_mask])
        print("[DEBUG] input_ids_cpu values:", input_ids_cpu)
        print("[DEBUG] vocab_size:", vocab_size)

        # Forward passes (no grad through ref)
        with torch.no_grad():
            ref_logits = self.ref_model(**ref_inputs).logits  # [1, L, V] on ref_dev
        policy_logits = self.policy_model(**pol_inputs).logits # [1, L, V] on pol_dev
        print("[DEBUG] policy_logits shape:", policy_logits.shape)
        # Check for NaNs or Infs in logits
        if torch.isnan(policy_logits).any() or torch.isinf(policy_logits).any():
            print("[ERROR] NaN or Inf detected in policy_logits!")
        if torch.isnan(ref_logits).any() or torch.isinf(ref_logits).any():
            print("[ERROR] NaN or Inf detected in ref_logits!")

        # Forward passes (no grad through ref)
        with torch.no_grad():
            ref_logits = self.ref_model(**ref_inputs).logits  # [1, L, V] on ref_dev
        policy_logits = self.policy_model(**pol_inputs).logits # [1, L, V] on pol_dev

        # Align to generated span
        L = policy_logits.shape[1]
        start = max(L - resp_len - 1, 0)
        end = L - 1
        pol_slice = policy_logits[:, start:end, :]                 # [1, T, V] on pol_dev
        # move/cast ref logits to policy device/dtype for KL
        ref_slice = ref_logits[:, start:end, :].to(pol_slice.device, dtype=pol_slice.dtype)

        # Targets (not used directly for KL here, but checks lengths)
        # target_tokens = pol_inputs["input_ids"][:, start + 1 : L]

        # KL(policy || ref)
        logp = F.log_softmax(pol_slice, dim=-1)
        logq = F.log_softmax(ref_slice, dim=-1)
        p = logp.exp()
        kl_per_pos = (p * (logp - logq)).sum(dim=-1)              # [1, T]
        kl_mean = kl_per_pos.mean()

        return kl_mean

    def calculate_pairwise_consistency(self, scores1: list, scores2: list) -> float:
        """Calculate consistency between two sets of scores (lists of equal length)."""
        if len(scores1) != len(scores2):
            raise ValueError("Score lists must have the same length")

        if any(score is None for score in scores1) or any(score is None for score in scores2):
            raise ValueError("Scores cannot contain None values")

        if len(scores1) == 0:
            raise ValueError("Score lists must be non-empty")

        # guard against NaNs
        if np.isnan(scores1).any() or np.isnan(scores2).any():
            raise ValueError("Scores cannot contain NaN values")

        rank1 = rankdata(scores1, method='ordinal')
        rank2 = rankdata(scores2, method='ordinal')

        if self.method == ConsistencyMethodEnum.kendall:
            consistency, _ = kendalltau(rank1, rank2)
            # Scale from [-1, 1] to [0, 1]
            consistency = (consistency + 1) / 2
        elif self.method == ConsistencyMethodEnum.spearman:
            consistency, _ = spearmanr(rank1, rank2)
            consistency = (consistency + 1) / 2
        elif self.method == ConsistencyMethodEnum.toporder:
            # Compare top and bottom items (by score), not just first/last positions
            top1 = int(np.argmax(scores1))
            top2 = int(np.argmax(scores2))
            bot1 = int(np.argmin(scores1))
            bot2 = int(np.argmin(scores2))
            consistency = 1.0 if (top1 == top2 and bot1 == bot2) else 0.0
        else:
            raise ValueError(f"Method {self.method} not supported")

        # If the correlation returns nan (e.g., constant lists), treat as 0.0 consistency
        if consistency is None or (isinstance(consistency, float) and np.isnan(consistency)):
            return 0.0

        return float(consistency)

    def evaluate_dataset_consistency(self, file1_data: dict, file2_data: dict):
        """Evaluate consistency across entire datasets."""
        keys = set(file1_data.keys()) & set(file2_data.keys())
        consistency_rates = []
        key2consistency = {}

        for key in keys:
            try:
                score1 = file1_data[key]['reward']
                score2 = file2_data[key]['reward']

                consistency = self.calculate_pairwise_consistency(score1, score2)
                consistency_rates.append(consistency)
                key2consistency[key] = consistency

            except (ValueError, KeyError) as e:
                tools_log_on_rank(f"Key={key} skipped due to error: {e}", level='warning')
                continue

        return consistency_rates, key2consistency

    def generate_dpo_training_data(self, reward_data: dict, key2consistency: dict = None) -> dict:
        """Generate DPO training data from reward model outputs."""
        results = {}

        for k, item in reward_data.items():
            if 'reward' not in item or 'response' not in item or 'prompt' not in item:
                tools_log_on_rank(f"Key={k} missing required fields, skipped", level='warning')
                continue

            rewards = item['reward']
            if not isinstance(rewards, (list, tuple)) or len(rewards) == 0:
                tools_log_on_rank(f"Key={k} has empty/invalid rewards, skipped", level='warning')
                continue

            if any(r is None or (isinstance(r, float) and np.isnan(r)) for r in rewards):
                tools_log_on_rank(f"Key={k} has None/NaN rewards, skipped", level='warning')
                continue

            if len(item['response']) != len(rewards):
                tools_log_on_rank(f"Key={k} mismatch: {len(item['response'])} responses vs {len(rewards)} rewards, skipped", level='warning')
                continue

            selected_idx = int(np.argmax(rewards))
            rejected_idx = int(np.argmin(rewards))

            if selected_idx == rejected_idx:
                tools_log_on_rank(f"Key={k} has identical selected/rejected, skipped", level='warning')
                continue

            results[k] = {
                'prompt': item['prompt'],
                'selected': item['response'][selected_idx],
                'rejected': item['response'][rejected_idx],
                'selected_reward': float(rewards[selected_idx]),
                'rejected_reward': float(rewards[rejected_idx]),
                'consistency': key2consistency.get(k) if key2consistency else None
            }

        return results


def calculate_consistency_statistics(consistency_rates: list) -> dict:
    """Calculate statistical measures of consistency rates."""
    if not consistency_rates:
        return {}

    return {
        'mean': statistics.mean(consistency_rates),
        'min': min(consistency_rates),
        'max': max(consistency_rates),
        'std': statistics.stdev(consistency_rates) if len(consistency_rates) > 1 else 0.0,
        'count': len(consistency_rates)
    }


def main():
    """Main function for consistency evaluation and DPO data generation."""
    args: CalConsistencyConfig = parse_args(CalConsistencyConfig, pass_in=[])

    # Load data
    file1_data = tools_json_load(args.file1)
    file2_data = tools_json_load(args.file2)

    # Initialize evaluator (now compatible with calls that pass models/tokenizer)
    evaluator = ConsistencyEvaluator(method=args.method)

    # Calculate consistency
    consistency_rates, key2consistency = evaluator.evaluate_dataset_consistency(
        file1_data, file2_data
    )

    # Calculate and log statistics
    stats = calculate_consistency_statistics(consistency_rates)

    tools_log_on_rank(args)
    if args.method == ConsistencyMethodEnum.kendall:
        tools_log_on_rank(
            "Note: Kendall tau consistency rate scaled from [-1, 1] to [0, 1]",
            level='warning'
        )

    tools_log_on_rank(
        f"Files: file1={len(file1_data)}, file2={len(file2_data)}, "
        f"valid overlap={stats.get('count', 0)}\n"
        f"{args.method.name} consistency statistics:\n"
        f"mean={stats.get('mean', 0):.4f}, min={stats.get('min', 0):.4f}, "
        f"max={stats.get('max', 0):.4f}, std={stats.get('std', 0):.4f}"
    )

    # Generate DPO training data
    tools_log_on_rank(f"Using {args.file1} to generate DPO training data")
    dpo_data = evaluator.generate_dpo_training_data(file1_data, key2consistency)

    # Save results
    if args.output is None:
        args.output = f"{args.file1.removesuffix('.json')}.{args.method.name}.dpo.json"

    tools_json_dump(dpo_data, args.output)
    tools_log_on_rank(f"DPO training data saved to: {args.output}")

    return dpo_data, stats


if __name__ == '__main__':
    main()
