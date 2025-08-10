from scipy.stats import spearmanr, rankdata, kendalltau
import statistics
import numpy as np
import os
from CREAM.tools import tools_json_load, tools_log_on_rank, tools_json_dump, tools_get_time
from CREAM.config import ConsistencyMethodEnum, CalConsistencyConfig, parse_args

class ConsistencyEvaluator:
    """Evaluates consistency between two reward model outputs for RAG cream model."""
    
    def __init__(self, method: ConsistencyMethodEnum = ConsistencyMethodEnum.spearman):
        self.method = method
    
    def calculate_pairwise_consistency(self, scores1: list, scores2: list) -> float:
        """Calculate consistency between two sets of scores."""
        if len(scores1) != len(scores2):
            raise ValueError("Score lists must have the same length")
        
        if any(score is None for score in scores1) or any(score is None for score in scores2):
            raise ValueError("Scores cannot contain None values")
        
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
            # Check if top and bottom ranked items match
            if rank1[0] == rank2[0] and rank1[-1] == rank2[-1]:
                consistency = 1.0
            else:
                consistency = 0.0
        else:
            raise ValueError(f"Method {self.method} not supported")
        
        return consistency
    
    def evaluate_dataset_consistency(self, file1_data: dict, file2_data: dict) -> tuple[list, dict]:
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
            if 'reward' not in item or 'response' not in item:
                tools_log_on_rank(f"Key={k} missing required fields, skipped", level='warning')
                continue
                
            rewards = item['reward']
            if any(r is None for r in rewards):
                tools_log_on_rank(f"Key={k} has None rewards, skipped", level='warning')
                continue
            
            selected_idx = np.argmax(rewards)
            rejected_idx = np.argmin(rewards)
            
            if selected_idx == rejected_idx:
                tools_log_on_rank(f"Key={k} has identical selected/rejected, skipped", level='warning')
                continue
            
            results[k] = {
                'prompt': item['prompt'],
                'selected': item['response'][selected_idx],
                'rejected': item['response'][rejected_idx],
                'selected_reward': rewards[selected_idx],
                'rejected_reward': rewards[rejected_idx],
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
    
    # Initialize evaluator
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
