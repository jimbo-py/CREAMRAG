
"""
Consistency Reward Module for PPO Training
Integrates CREAM consistency evaluation with PPO reward computation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.stats import spearmanr, kendalltau, rankdata
import logging

logger = logging.getLogger(__name__)

class ConsistencyRewardModel:
    """Reward model that combines likelihood and consistency rewards"""
    
    def __init__(
        self,
        base_model,
        tokenizer,
        device,
        lambda_consistency: float = 0.5,
        consistency_method: str = "spearman",
        num_candidates: int = 3,
        normalize_by_length: bool = True
    ):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.lambda_consistency = lambda_consistency
        self.consistency_method = consistency_method
        self.num_candidates = num_candidates
        self.normalize_by_length = normalize_by_length
        
        # Set model to eval mode
        self.base_model.eval()
    
    def compute_likelihood_reward(self, prompt: str, response: str) -> float:
        """Compute likelihood-based reward for a single response"""
        try:
            # Tokenize prompt and response
            prompt_inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            full_text = prompt + response
            full_inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            prompt_length = prompt_inputs['input_ids'].shape[1]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.base_model(**full_inputs)
                logits = outputs.logits[0, prompt_length-1:-1, :]  # Response tokens only
                response_ids = full_inputs['input_ids'][0, prompt_length:]
                
                # Compute log probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = torch.gather(
                    log_probs, 1, response_ids.unsqueeze(1)
                ).squeeze(1)
                
                # Sum log probabilities
                total_log_prob = token_log_probs.sum()
                
                # Normalize by length if requested
                if self.normalize_by_length:
                    total_log_prob /= token_log_probs.shape[0]
                
                return total_log_prob.item()
                
        except Exception as e:
            logger.warning(f"Likelihood reward computation failed: {e}")
            return 0.0
    
    def generate_candidates(self, prompt: str, max_new_tokens: int = 128) -> List[str]:
        """Generate multiple candidate responses for consistency evaluation"""
        candidates = []
        
        for _ in range(self.num_candidates):
            try:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.base_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode response
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                candidates.append(response.strip())
                
            except Exception as e:
                logger.warning(f"Candidate generation failed: {e}")
                candidates.append("")
        
        return candidates
    
    def compute_consistency_reward(self, prompt: str, response: str) -> float:
        """Compute consistency reward by comparing rankings"""
        try:
            # Generate candidates
            candidates = self.generate_candidates(prompt)
            if len(candidates) < 2:
                return 0.0
            
            # Add the actual response to candidates
            all_candidates = candidates + [response]
            
            # Score all candidates with base model
            scores = []
            for candidate in all_candidates:
                score = self.compute_likelihood_reward(prompt, candidate)
                scores.append(score)
            
            # Ensure we have enough scores
            if len(scores) < 2:
                return 0.0
            
            # Create two ranking scenarios:
            # 1. Base model ranking (reference)
            # 2. Current policy ranking (assuming response is best)
            
            # Reference ranking (base model scores)
            ref_scores = scores[:-1]  # Exclude the actual response
            if len(ref_scores) < 2:
                return 0.0
                
            ref_ranks = rankdata(ref_scores, method='ordinal')
            
            # Policy ranking (assume response is ranked first)
            policy_scores = scores[:-1] + [scores[-1]]  # Put response first
            policy_ranks = rankdata(policy_scores, method='ordinal')
            
            # Ensure arrays have the same length for correlation
            if len(ref_ranks) != len(policy_ranks):
                # Pad the shorter array
                min_len = min(len(ref_ranks), len(policy_ranks))
                ref_ranks = ref_ranks[:min_len]
                policy_ranks = policy_ranks[:min_len]
            
            # Compute consistency between rankings
            if self.consistency_method == "spearman":
                if len(ref_ranks) >= 2:
                    consistency, _ = spearmanr(ref_ranks, policy_ranks)
                else:
                    consistency = 0.0
            elif self.consistency_method == "kendall":
                if len(ref_ranks) >= 2:
                    consistency, _ = kendalltau(ref_ranks, policy_ranks)
                else:
                    consistency = 0.0
            elif self.consistency_method == "toporder":
                # Check if top and bottom rankings match
                if len(ref_scores) >= 2:
                    ref_top = np.argmax(ref_scores)
                    ref_bottom = np.argmin(ref_scores)
                    policy_top = np.argmax(policy_scores)
                    policy_bottom = np.argmin(policy_scores)
                    consistency = 1.0 if (ref_top == policy_top and ref_bottom == policy_bottom) else 0.0
                else:
                    consistency = 0.0
            else:
                raise ValueError(f"Unsupported consistency method: {self.consistency_method}")
            
            # Handle NaN values
            if np.isnan(consistency):
                consistency = 0.0
            
            # Scale from [-1, 1] to [0, 1] for spearman and kendall
            if self.consistency_method in ["spearman", "kendall"]:
                consistency = (consistency + 1.0) / 2.0
            
            return float(consistency)
            
        except Exception as e:
            logger.warning(f"Consistency reward computation failed: {e}")
            return 0.0
    
    def compute_reward(self, prompt: str, response: str) -> float:
        """Compute combined reward: likelihood + consistency"""
        # Compute likelihood reward
        likelihood_reward = self.compute_likelihood_reward(prompt, response)
        
        # Compute consistency reward
        consistency_reward = self.compute_consistency_reward(prompt, response)
        
        # Combine rewards
        total_reward = likelihood_reward + self.lambda_consistency * consistency_reward
        
        return total_reward
    
    def compute_batch_rewards(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Compute rewards for a batch of prompt-response pairs"""
        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = self.compute_reward(prompt, response)
            rewards.append(reward)
        return rewards

class RAGConsistencyRewardModel:
    """Enhanced reward model for RAG systems with retrieval consistency"""
    
    def __init__(
        self,
        base_model,
        tokenizer,
        retriever,
        device,
        lambda_consistency: float = 0.5,
        lambda_retrieval: float = 0.1,
        consistency_method: str = "spearman",
        num_candidates: int = 3
    ):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.device = device
        self.lambda_consistency = lambda_consistency
        self.lambda_retrieval = lambda_retrieval
        self.consistency_method = consistency_method
        self.num_candidates = num_candidates
        
        # Initialize consistency reward model
        self.consistency_model = ConsistencyRewardModel(
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            lambda_consistency=lambda_consistency,
            consistency_method=consistency_method,
            num_candidates=num_candidates
        )
    
    def compute_retrieval_consistency(self, query: str, response: str) -> float:
        """Compute consistency of retrieval across different query formulations"""
        try:
            # Create different query formulations
            query_variations = [
                query,
                f"What is {query.lower()}?",
                f"Tell me about {query.lower()}",
                f"Explain {query.lower()}"
            ]
            
            # Retrieve documents for each variation
            retrieval_results = []
            for var_query in query_variations:
                try:
                    docs = self.retriever.retrieve(var_query, k=3)
                    retrieval_results.append(docs)
                except Exception as e:
                    logger.warning(f"Retrieval failed for query variation: {e}")
                    retrieval_results.append([])
            
            # Compute overlap between retrieval results
            overlaps = []
            for i in range(len(retrieval_results)):
                for j in range(i + 1, len(retrieval_results)):
                    set1 = set(retrieval_results[i])
                    set2 = set(retrieval_results[j])
                    if len(set1) > 0 and len(set2) > 0:
                        overlap = len(set1.intersection(set2)) / len(set1.union(set2))
                        overlaps.append(overlap)
            
            # Return average overlap as consistency measure
            if overlaps:
                return np.mean(overlaps)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Retrieval consistency computation failed: {e}")
            return 0.0
    
    def compute_reward(self, prompt: str, response: str) -> float:
        """Compute comprehensive reward including retrieval consistency"""
        # Extract query from prompt (assuming format: "Context: ...\n\nQuestion: ...\nAnswer:")
        try:
            if "Question:" in prompt:
                query = prompt.split("Question:")[1].split("Answer:")[0].strip()
            else:
                query = prompt  # Fallback to full prompt
            
            # Compute base consistency reward
            consistency_reward = self.consistency_model.compute_reward(prompt, response)
            
            # Compute retrieval consistency
            retrieval_consistency = self.compute_retrieval_consistency(query, response)
            
            # Combine rewards
            total_reward = consistency_reward + self.lambda_retrieval * retrieval_consistency
            
            return total_reward
            
        except Exception as e:
            logger.warning(f"RAG reward computation failed: {e}")
            return 0.0
    
    def compute_batch_rewards(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Compute rewards for a batch of prompt-response pairs"""
        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = self.compute_reward(prompt, response)
            rewards.append(reward)
        return rewards
