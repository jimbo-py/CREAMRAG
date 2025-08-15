from tqdm import tqdm
import os
import torch
import torch.distributed as dist
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

from datasets import Dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from CREAM.tools import (
    tools_json_load, tools_json_dump, tools_get_checkpoint_load_path, 
    tools_log_on_rank, tools_set_device_env, tools_get_time, 
)
from CREAM.config import RewardingConfig, parse_args
from CREAM.utils import prepare_model, set_pad_token, setup, dist_sync_objects


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


class RAGCreamRewardSystem:
    """Integrated reward system for RAG cream model."""
    
    def __init__(self, args: RewardingConfig):
        self.args = args
        self.time_based = tools_get_time()
        
    def load_data(self, input_file: str, debug: bool = False) -> List[Dict]:
        """Load and prepare data for reward calculation."""
        data = tools_json_load(input_file)
        
        if debug:
            data = dict(list(data.items())[:17])
        
        # Flatten data structure
        processed_data = [
            {
                'uuid': uuid,
                'response_id': response_id,
                'prompt': item['prompt'],
                'response': response
            }
            for uuid, item in data.items()
            for response_id, response in enumerate(item['responses'])
        ]
        
        # Sort by length for efficient batching
        processed_data.sort(
            key=lambda x: len(x['prompt']) + len(x['response']), 
            reverse=True
        )
        
        return processed_data
    
    def setup_model_and_tokenizer(self, checkpoint_path: Optional[str] = None):
        """Setup model and tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.args.model.value)
        set_pad_token(tokenizer)
        
        model = prepare_model(
            rank=self.args.common.rank,
            model_config=self.args.model,
            lora_config=None,
            bf16=self.args.common.bf16,
            debug=self.args.common.debug,
            ckpt_path=checkpoint_path,
            tokenizer=tokenizer
        )
        set_pad_token(tokenizer, model)
        
        return model, tokenizer
    
    def create_data_loader(self, data: List[Dict], tokenizer: PreTrainedTokenizer):
        """Create distributed data loader."""
        dataset = Dataset.from_list(data)
        
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            collate_fn=PaddingCollator(tokenizer),
            sampler=DistributedSampler(
                dataset, 
                self.args.common.world_size, 
                self.args.common.rank, 
                shuffle=False
            )
        )
    
    def format_results(self, reward_outputs: List[Dict]) -> Dict:
        """Format results into final structure."""
        results = {}
        max_response_num = max(int(item['response_id']) for item in reward_outputs) + 1
        
        for item in reward_outputs:
            uuid = item['uuid']
            if uuid not in results:
                results[uuid] = {
                    'prompt': item['prompt'],
                    'response': [None] * max_response_num,
                    'score': [None] * max_response_num,
                    'ref_score': [None] * max_response_num,
                    'reward': [None] * max_response_num,
                }
            
            response_id = item['response_id']
            for key in ['response', 'score', 'ref_score', 'reward']:
                if key in item:
                    results[uuid][key][response_id] = item[key]
        
        return results
    
    def save_results(self, results: Dict) -> str:
        """Save results to file."""
        output_path = self.args.input_file.removesuffix('.json')
        suffix = f"{self.args.model.name}-policy_{self.args.checkpoint}"
        
        if self.args.enable_ref_model:
            suffix += f"-ref_{self.args.ref_model_ckpt}"
        
        if self.args.common.debug:
            suffix += "-debug"
        
        final_path = f"{output_path}.rewarding.{suffix}.json"
        
        if os.path.exists(final_path):
            final_path = f"{output_path}.rewarding.{suffix}.{self.time_based}.json"
        
        tools_json_dump(results, final_path)
        return final_path
    
    def run_reward_calculation(self, rank: int):
        """Main reward calculation pipeline."""
        # Setup
        self.args.common.rank = rank
        setup(self.args.common)
        group_gloo = dist.new_group(backend="gloo")
        
        # Load data
        data = self.load_data(self.args.input_file, self.args.common.debug)
        
        # Setup policy model
        policy_model, tokenizer = self.setup_model_and_tokenizer(
            tools_get_checkpoint_load_path(self.args.checkpoint)
        )
        reward_model = RewardModel(policy_model, tokenizer, policy_model.device)
        
        # Create data loader
        data_loader = self.create_data_loader(data, tokenizer)
        data_loader.sampler.set_epoch(0)
        
        # Calculate policy rewards
        policy_outputs = reward_model.calculate_likelihood_rewards(
            data_loader, 
            normalized_by_length=not self.args.enable_ref_model,
            desc='Policy rewards 1/1' if not self.args.enable_ref_model else 'Policy rewards 1/2'
        )
        
        # Sync across ranks
        policy_outputs = dist_sync_objects(
            policy_outputs, group_gloo, rank, 
            self.args.common.world_size, dedup_key='dedup_key'
        )
        
        if self.args.enable_ref_model:
            # Setup reference model
            del policy_model
            torch.cuda.empty_cache()
            
            ref_model, _ = self.setup_model_and_tokenizer(
                tools_get_checkpoint_load_path(self.args.ref_model_ckpt)
            )
            reward_model.model = ref_model
            reward_model.device = ref_model.device
            
            # Calculate reference rewards
            ref_outputs = reward_model.calculate_likelihood_rewards(
                data_loader, 
                normalized_by_length=False,
                desc='Reference rewards 2/2'
            )
            
            # Sync reference outputs
            ref_outputs = dist_sync_objects(
                ref_outputs, group_gloo, rank,
                self.args.common.world_size, dedup_key='dedup_key'
            )
            
            if rank == 0:
                # Calculate DPO rewards
                final_outputs = reward_model.calculate_dpo_rewards(policy_outputs, ref_outputs)
        else:
            if rank == 0:
                # Use likelihood as reward
                for item in policy_outputs:
                    item['reward'] = item['score']
                    item['ref_score'] = 0
                final_outputs = policy_outputs
        
        # Save results on rank 0
        if rank == 0:
            results = self.format_results(final_outputs)
            output_path = self.save_results(results)
            
            tools_log_on_rank(self.args)
            tools_log_on_rank(f"Reward calculation completed. Results saved to: {output_path}")
            
            return results
        
        return None


def main(rank: int, time_based: str, args: RewardingConfig):
    """Main function for reward model execution."""
    reward_system = RAGCreamRewardSystem(args)
    reward_system.time_based = time_based
    return reward_system.run_reward_calculation(rank)


if __name__ == '__main__':
    args: RewardingConfig = parse_args(RewardingConfig, pass_in=[])
    tools_set_device_env(args.common.device)
    time_based = tools_get_time()
    
    tools_log_on_rank(time_based, args)
    
    import torch.multiprocessing as mp
    mp.spawn(main, args=(time_based, args), nprocs=args.common.world_size, join=True)

