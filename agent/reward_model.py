"""
Reward model implementation aligned with CREAM's rewarding.py.
Provides likelihood-based reward calculation for DPO training.
"""

from tqdm import tqdm
import os
import torch
import torch.distributed as dist
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

from datasets import Dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


class PaddingCollator:
    """Collator for padding sequences for reward model inference."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features: List[Dict]) -> Dict[str, Union[List, torch.Tensor]]:
        """Apply padding and create batch."""
        self.tokenizer.padding_side = 'left'
        
        # Organize batch data
        batch = {k: [item[k] for item in features] for k in features[0].keys()}
        
        # Combine prompt and response for full conversation
        batch['response'] = [
            prompt + [{'role': 'assistant', 'content': response}]
            for prompt, response in zip(batch['prompt'], batch['response'])
        ]
        
        # Get prefix lengths (prompt only)
        prefix_tokens = self.tokenizer.apply_chat_template(
            batch['prompt'], padding=False, truncation=False
        )
        prefix_lengths = [len(p) for p in prefix_tokens]
        
        # Tokenize full sequences
        inputs = self.tokenizer.apply_chat_template(
            batch['response'], 
            padding=True, 
            return_tensors='pt', 
            max_length=self.max_length, 
            truncation=True, 
            return_dict=True
        )
        
        # Create labels (mask prompt tokens)
        labels = inputs['input_ids'].clone()
        for i in range(len(labels)):
            # Find first non-pad token
            for j in range(len(labels[i])):
                if labels[i, j] != self.tokenizer.pad_token_id:
                    # Mask prompt tokens, keep response tokens
                    labels[i, j:j+prefix_lengths[i]] = self.tokenizer.pad_token_id
                    break
        
        inputs['labels'] = labels
        batch['tensor_inputs'] = inputs
        
        return batch


class RewardModel:
    """Reward model for evaluating response quality."""
    
    def __init__(
        self, 
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer,
        device: torch.device
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def calculate_likelihood_rewards(
        self, 
        data_loader: DataLoader, 
        normalized_by_length: bool = True,
        desc: str = "Calculating rewards"
    ) -> List[Dict]:
        """Calculate likelihood-based rewards for responses."""
        self.model.eval()
        outputs = []
        
        with tqdm(data_loader, desc=desc, disable=dist.get_rank() != 0) as pbar:
            for batch in data_loader:
                with torch.no_grad():
                    # Forward pass
                    model_outputs = self.model(
                        **batch['tensor_inputs'].to(self.device), 
                        return_dict=True
                    )
                    logits = model_outputs['logits'][:, :-1, :]
                    labels = batch['tensor_inputs']['labels'][:, 1:]
                    
                    # Calculate per-token log probabilities
                    per_token_logps = torch.gather(
                        logits.log_softmax(-1), 
                        dim=2, 
                        index=labels.unsqueeze(2)
                    ).squeeze(2)
                    
                    # Create mask for non-pad tokens
                    loss_mask = (labels != self.tokenizer.pad_token_id).float()
                    
                    # Sum log probabilities
                    sum_log_probs = (per_token_logps * loss_mask).sum(-1)
                    
                    # Normalize by length if requested
                    if normalized_by_length:
                        sum_log_probs /= loss_mask.sum(-1)
                
                # Collect results
                for i, likelihood in enumerate(sum_log_probs):
                    outputs.append({
                        'uuid': batch['uuid'][i],
                        'response_id': batch['response_id'][i],
                        'prompt': batch['prompt'][i],
                        'response': batch['response'][i],
                        'dedup_key': f"{batch['uuid'][i]}_{batch['response_id'][i]}",
                        'score': likelihood.item(),
                    })
                
                pbar.update(1)
        
        return outputs
    
    def calculate_dpo_rewards(
        self,
        policy_outputs: List[Dict],
        ref_outputs: List[Dict]
    ) -> List[Dict]:
        """Calculate DPO-style rewards (policy - reference)."""
        policy_dict = {item['dedup_key']: item for item in policy_outputs}
        
        for ref_item in ref_outputs:
            key = ref_item['dedup_key']
            if key in policy_dict:
                policy_dict[key]['ref_score'] = ref_item['score']
                policy_dict[key]['reward'] = policy_dict[key]['score'] - ref_item['score']
        
        return list(policy_dict.values())











