"""
Enhanced PPO Trainer for CREAM-RAG with Consistency Rewards
Integrates PPO training with CREAM consistency evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from ppo_utils import (
    RolloutBuffer, AdaptiveKLController, PPOLoss, 
    PPOScheduler, PPOStats, compute_explained_variance,
    get_gradient_norm, whiten
)
from agent.consistency_reward import ConsistencyRewardModel, RAGConsistencyRewardModel

logger = logging.getLogger(__name__)

class EnhancedValueHead(nn.Module):
    """Enhanced value head with better initialization and architecture"""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_size, hidden_size // 2)
        self.linear2 = nn.Linear(hidden_size // 2, 1)
        self.activation = nn.ReLU()
        
        # Initialize with orthogonal weights
        nn.init.orthogonal_(self.linear1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.linear2.weight, gain=0.01)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            values: [batch_size, seq_len]
        """
        hidden_states = self.dropout(hidden_states)
        x = self.activation(self.linear1(hidden_states))
        x = self.dropout(x)
        values = self.linear2(x).squeeze(-1)
        return values

class EnhancedPPOConfig:
    """Enhanced configuration for PPO training with consistency rewards"""
    
    def __init__(self, **kwargs):
        # PPO hyperparameters
        self.clip_range = kwargs.get('clip_range', 0.2)
        self.vf_coef = kwargs.get('vf_coef', 0.5)
        self.ent_coef = kwargs.get('ent_coef', 0.01)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.ppo_epochs = kwargs.get('ppo_epochs', 4)
        self.minibatch_size = kwargs.get('minibatch_size', 64)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        
        # KL control
        self.target_kl = kwargs.get('target_kl', 0.01)
        self.kl_coef = kwargs.get('kl_coef', 0.1)
        self.adaptive_kl = kwargs.get('adaptive_kl', True)
        
        # Value function
        self.value_clip = kwargs.get('value_clip', True)
        self.normalize_advantages = kwargs.get('normalize_advantages', True)
        
        # Learning rates
        self.learning_rate = kwargs.get('learning_rate', 1e-5)
        self.value_lr_multiplier = kwargs.get('value_lr_multiplier', 1.0)
        
        # Consistency reward settings
        self.lambda_consistency = kwargs.get('lambda_consistency', 0.5)
        self.lambda_retrieval = kwargs.get('lambda_retrieval', 0.1)
        self.consistency_method = kwargs.get('consistency_method', 'spearman')
        self.num_candidates = kwargs.get('num_candidates', 3)
        
        # Training stability
        self.use_gae = kwargs.get('use_gae', True)
        self.normalize_rewards = kwargs.get('normalize_rewards', True)
        self.reward_clipping = kwargs.get('reward_clipping', True)
        self.reward_clip_range = kwargs.get('reward_clip_range', 10.0)

class EnhancedPPOTrainer:
    """Enhanced PPO trainer with consistency rewards and better stability"""
    
    def __init__(self, model, tokenizer, retriever, config: EnhancedPPOConfig, device):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.config = config
        self.device = device
        
        # Add enhanced value head with matching dtype
        self.value_head = EnhancedValueHead(model.config.hidden_size).to(device)
        # Ensure value head matches model dtype
        if hasattr(self.model, 'dtype'):
            self.value_head = self.value_head.to(dtype=self.model.dtype)
        
        # Setup reward model
        self.reward_model = RAGConsistencyRewardModel(
            base_model=model,
            tokenizer=tokenizer,
            retriever=retriever,
            device=device,
            lambda_consistency=config.lambda_consistency,
            lambda_retrieval=config.lambda_retrieval,
            consistency_method=config.consistency_method,
            num_candidates=config.num_candidates
        )
        
        # Setup optimizers
        self.setup_optimizers()
        
        # Initialize utilities
        self.kl_controller = AdaptiveKLController(config.target_kl) if config.adaptive_kl else None
        self.stats = PPOStats()
        self.loss_fn = PPOLoss()
        
        # Training state
        self.step_count = 0
        self.reward_stats = {'mean': 0.0, 'std': 1.0}
        
        # Create a backup of the initial model state
        self.model_backup = {}
        for name, param in self.model.named_parameters():
            self.model_backup[name] = param.data.clone()
        
    def setup_optimizers(self):
        """Setup optimizers with better configuration"""
        # Actor (language model parameters)
        actor_params = list(self.model.parameters())
        
        # Critic (value head parameters)
        critic_params = list(self.value_head.parameters())
        
        # Create optimizers with weight decay
        self.actor_optimizer = torch.optim.AdamW(
            actor_params, 
            lr=self.config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.critic_optimizer = torch.optim.AdamW(
            critic_params,
            lr=self.config.learning_rate * self.config.value_lr_multiplier,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Initially freeze the actor model to train only the value head
        self.actor_frozen = True
        self.actor_freeze_steps = 0
        self.actor_unfreeze_threshold = 100  # Unfreeze after 100 steps
        for param in self.model.parameters():
            param.requires_grad = False
        logger.info("Actor model frozen initially - training only value head")
    
    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using running statistics"""
        if self.config.normalize_rewards:
            try:
                # Move to CPU for safer computation
                rewards_cpu = rewards.detach().cpu()
                
                # Handle NaN and inf values in rewards
                rewards_cpu = torch.nan_to_num(rewards_cpu, nan=0.0, posinf=10.0, neginf=-10.0)
                
                # Update running statistics
                if self.step_count == 0:
                    self.reward_stats['mean'] = rewards_cpu.mean().item()
                    self.reward_stats['std'] = rewards_cpu.std().item()
                else:
                    # Exponential moving average
                    alpha = 0.99
                    self.reward_stats['mean'] = alpha * self.reward_stats['mean'] + (1 - alpha) * rewards_cpu.mean().item()
                    self.reward_stats['std'] = alpha * self.reward_stats['std'] + (1 - alpha) * rewards_cpu.std().item()
                
                # Ensure std is not zero or NaN
                if self.reward_stats['std'] <= 0 or np.isnan(self.reward_stats['std']):
                    self.reward_stats['std'] = 1.0
                
                # Normalize
                normalized = (rewards_cpu - self.reward_stats['mean']) / (self.reward_stats['std'] + 1e-8)
                
                # Clip if requested
                if self.config.reward_clipping:
                    normalized = torch.clamp(normalized, -self.config.reward_clip_range, self.config.reward_clip_range)
                
                # Move back to original device
                return normalized.to(rewards.device)
                
            except Exception as e:
                logger.warning(f"Reward normalization failed: {e}, using original rewards")
                return rewards
        else:
            return rewards
    
    def generate_response_with_log_probs(self, prompt: str, max_new_tokens: int = 32, 
                                       temperature: float = 0.7) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Generate response and return text, log_probs, and values with better handling"""
        
        # Check if model is in a valid state before generation
        model_weights_valid = all(torch.isfinite(p).all() for p in self.model.parameters())
        if not model_weights_valid:
            logger.error("Model weights are invalid before generation, restoring from backup")
            # Restore model from backup
            for name, param in self.model.named_parameters():
                if name in self.model_backup:
                    param.data = self.model_backup[name].clone()
            logger.info("Model restored from backup before generation")
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Ensure inputs match model dtype, but keep input_ids as Long
        if hasattr(self.model, 'dtype'):
            for key in inputs:
                if key == 'input_ids':
                    # Keep input_ids as Long for embedding layer
                    inputs[key] = inputs[key].to(dtype=torch.long)
                elif inputs[key].dtype != self.model.dtype:
                    inputs[key] = inputs[key].to(dtype=self.model.dtype)
        
        prompt_length = inputs['input_ids'].shape[1]
        
        # Generate with model
        with torch.no_grad():
            try:
                generation_output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    eos_token_id=self.tokenizer.eos_token_id,  # Add EOS token
                    early_stopping=True  # Add early stopping
                )
            except RuntimeError as e:
                if "device-side assert" in str(e):
                    logger.warning(f"Generation failed due to device-side assert, using fallback: {e}")
                    # Fallback to greedy generation with minimal parameters
                    try:
                        generation_output = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=0.0,  # Greedy
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            return_dict_in_generate=True,
                            output_scores=True,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    except RuntimeError as e2:
                        logger.error(f"Even fallback generation failed: {e2}")
                        # Return a simple default response
                        return "Error in generation", torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
                else:
                    raise e
            
            generated_ids = generation_output.sequences[0]
            response_ids = generated_ids[prompt_length:]
            
            # Decode response
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Get log probabilities and values for the full sequence
        full_ids = generated_ids.unsqueeze(0)
        attention_mask = torch.ones_like(full_ids)
        
        # Ensure attention_mask matches model dtype
        if hasattr(self.model, 'dtype'):
            attention_mask = attention_mask.to(dtype=self.model.dtype)
        
        with torch.no_grad():
            # Forward pass through model
            model_outputs = self.model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Compute log probabilities for response tokens
            logits = model_outputs.logits[0, prompt_length-1:-1, :]  # [response_len, vocab_size]
            
            # Handle potential NaN/inf in logits
            logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
            
            # Additional safety check: ensure logits are finite
            if not torch.isfinite(logits).all():
                logger.warning("Non-finite logits detected, using fallback")
                return "Error in generation", torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
            
            response_log_probs = F.log_softmax(logits, dim=-1)
            
            # Get log probs for actual tokens
            token_log_probs = torch.gather(
                response_log_probs, 
                1, 
                response_ids.unsqueeze(1)
            ).squeeze(1)  # [response_len]
            
            # Handle NaN/inf in token log probs
            token_log_probs = torch.nan_to_num(token_log_probs, nan=0.0, posinf=0.0, neginf=-100.0)
            
            # Get values from value head
            hidden_states = model_outputs.hidden_states[-1][0, prompt_length-1:-1, :]  # [response_len, hidden_size]
            # Ensure hidden_states match value head dtype
            if hasattr(self.value_head, 'dtype'):
                hidden_states = hidden_states.to(dtype=self.value_head.dtype)
            values = self.value_head(hidden_states.unsqueeze(0)).squeeze(0)  # [response_len]
        
        return response, token_log_probs.sum(), values.mean()
    
    def collect_rollouts(self, prompts: List[str], rollout_size: int) -> RolloutBuffer:
        """Collect rollouts with enhanced reward computation"""
        buffer = RolloutBuffer(rollout_size, self.device)
        
        self.model.eval()
        
        successful_rollouts = 0
        
        for i, prompt in enumerate(prompts[:rollout_size]):
            try:
                # Generate response
                response, log_prob, value = self.generate_response_with_log_probs(prompt)
                
                # Compute enhanced reward with consistency
                reward = self.reward_model.compute_reward(prompt, response)
                
                # Handle NaN/inf in reward
                if np.isnan(reward) or np.isinf(reward):
                    reward = 0.0
                
                # Add to buffer
                buffer.add(
                    query=prompt,
                    response=response,
                    reward=reward,
                    value=value.item(),
                    log_prob=log_prob.item(),
                    mask=True
                )
                
                successful_rollouts += 1
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Collected {i + 1}/{rollout_size} rollouts")
                    
            except Exception as e:
                logger.warning(f"Failed to generate rollout {i}: {e}")
                # Add a default rollout with zero reward
                buffer.add(
                    query=prompt,
                    response="Error in generation",
                    reward=0.0,
                    value=0.0,
                    log_prob=0.0,
                    mask=True
                )
                continue
        
        # Check if we have any successful rollouts
        if successful_rollouts == 0:
            logger.error("No successful rollouts collected, returning empty buffer")
            return buffer
        
        # Compute advantages and returns
        if self.config.use_gae:
            buffer.compute_advantages_and_returns(
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda
            )
        else:
            # Simple advantage computation
            batch_data = buffer.get()
            rewards = batch_data['rewards']
            values = batch_data['values']
            advantages = rewards - values
            returns = rewards
            buffer.advantages = advantages.tolist()
            buffer.returns = returns.tolist()
        
        return buffer
    
    def compute_model_outputs(self, queries: List[str], responses: List[str]) -> Dict[str, torch.Tensor]:
        """Compute model outputs with better error handling"""
        batch_size = len(queries)
        all_log_probs = []
        all_values = []
        all_entropies = []
        
        for query, response in zip(queries, responses):
            try:
                # Combine query and response
                full_text = query + response
                
                # Tokenize
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=1024
                ).to(self.device)
                
                # Get query length for masking
                query_inputs = self.tokenizer(query, return_tensors="pt", padding=False)
                query_length = query_inputs['input_ids'].shape[1]
                
                # Forward pass
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Get logits for response tokens only
                logits = outputs.logits[0, query_length-1:-1, :]  # [response_len, vocab_size]
                
                # Get response token IDs
                response_ids = inputs['input_ids'][0, query_length:]  # [response_len]
                
                # Handle potential NaN/inf in logits
                logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
                
                # Compute log probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = torch.gather(log_probs, 1, response_ids.unsqueeze(1)).squeeze(1)
                
                # Handle NaN/inf in token log probs
                token_log_probs = torch.nan_to_num(token_log_probs, nan=0.0, posinf=0.0, neginf=-100.0)
                sequence_log_prob = token_log_probs.sum()
                
                # Compute entropy
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(-1).mean()
                
                # Handle NaN/inf in entropy
                if torch.isnan(entropy) or torch.isinf(entropy):
                    entropy = torch.tensor(0.0, device=self.device)
                
                # Get values
                hidden_states = outputs.hidden_states[-1][0, query_length-1:-1, :]
                values = self.value_head(hidden_states.unsqueeze(0)).squeeze(0).mean()
                
                all_log_probs.append(sequence_log_prob)
                all_values.append(values)
                all_entropies.append(entropy)
                
            except Exception as e:
                logger.warning(f"Model output computation failed: {e}")
                # Add default values
                all_log_probs.append(torch.tensor(0.0, device=self.device))
                all_values.append(torch.tensor(0.0, device=self.device))
                all_entropies.append(torch.tensor(0.0, device=self.device))
        
        return {
            'log_probs': torch.stack(all_log_probs),
            'values': torch.stack(all_values),
            'entropy': torch.stack(all_entropies).mean()
        }
    
    def ppo_update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Enhanced PPO update with better stability"""
        try:
            self.model.train()
            self.value_head.train()
            
            batch_data = buffer.get()
            batch_size = len(batch_data['queries'])
            
            # Normalize rewards
            batch_data['rewards'] = self.normalize_rewards(batch_data['rewards'])
            
            # Store old values for clipping
            old_values = batch_data['values'].clone()
            
            update_stats = {
                'policy_losses': [],
                'value_losses': [],
                'entropy_losses': [],
                'total_losses': [],
                'clipfracs': [],
                'approx_kls': [],
                'grad_norms': []
            }
        
            # PPO epochs
            for epoch in range(self.config.ppo_epochs):
                # Shuffle data
                indices = torch.randperm(batch_size)
                
                # Mini-batch updates
                for start in range(0, batch_size, self.config.minibatch_size):
                    end = start + self.config.minibatch_size
                    mb_indices = indices[start:end]
                    
                    # Get mini-batch data
                    mb_queries = [batch_data['queries'][i] for i in mb_indices]
                    mb_responses = [batch_data['responses'][i] for i in mb_indices]
                    mb_old_log_probs = batch_data['log_probs'][mb_indices]
                    mb_advantages = batch_data['advantages'][mb_indices]
                    mb_returns = batch_data['returns'][mb_indices]
                    mb_old_values = old_values[mb_indices]
                    
                    # Normalize advantages
                    if self.config.normalize_advantages:
                        mb_advantages = whiten(mb_advantages)
                    
                    # Compute current model outputs
                    model_outputs = self.compute_model_outputs(mb_queries, mb_responses)
                    current_log_probs = model_outputs['log_probs']
                    current_values = model_outputs['values']
                    entropy = model_outputs['entropy']
                    
                    # Ensure all tensors have the same dtype as the model
                    model_dtype = next(self.model.parameters()).dtype
                    current_log_probs = current_log_probs.to(dtype=model_dtype)
                    mb_old_log_probs = mb_old_log_probs.to(dtype=model_dtype)
                    mb_advantages = mb_advantages.to(dtype=model_dtype)
                    current_values = current_values.to(dtype=model_dtype)
                    mb_returns = mb_returns.to(dtype=model_dtype)
                    mb_old_values = mb_old_values.to(dtype=model_dtype)
                    entropy = entropy.to(dtype=model_dtype)
                    
                                    # Debug: Check if we have meaningful gradients
                logger.debug(f"Current log probs: {current_log_probs.mean().item():.6f} ± {current_log_probs.std().item():.6f}")
                logger.debug(f"Old log probs: {mb_old_log_probs.mean().item():.6f} ± {mb_old_log_probs.std().item():.6f}")
                logger.debug(f"Advantages: {mb_advantages.mean().item():.6f} ± {mb_advantages.std().item():.6f}")
                logger.debug(f"Current values: {current_values.mean().item():.6f} ± {current_values.std().item():.6f}")
                logger.debug(f"Returns: {mb_returns.mean().item():.6f} ± {mb_returns.std().item():.6f}")
                
                # Compute losses with better numerical stability
                if not self.actor_frozen:
                    policy_loss, policy_metrics = self.loss_fn.compute_policy_loss(
                        current_log_probs, mb_old_log_probs, mb_advantages, self.config.clip_range
                    )
                    entropy_loss = -self.config.ent_coef * entropy
                else:
                    # When actor is frozen, use dummy losses
                    policy_loss = torch.tensor(0.0, device=current_log_probs.device, requires_grad=True)
                    policy_metrics = {'policy_loss': 0.0, 'clip_fraction': 0.0, 'clipfrac': 0.0, 'approx_kl': 0.0}
                    entropy_loss = torch.tensor(0.0, device=current_log_probs.device, requires_grad=True)
                
                value_loss, value_metrics = self.loss_fn.compute_value_loss(
                    current_values, mb_returns, mb_old_values, 
                    self.config.clip_range, self.config.value_clip
                )
                
                # Debug: Check loss values
                logger.debug(f"Policy loss: {policy_loss.item():.6f}, Value loss: {value_loss.item():.6f}, Entropy loss: {entropy_loss.item():.6f}")
                
                # Handle NaN/inf in losses
                if torch.isnan(policy_loss) or torch.isinf(policy_loss):
                    policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                if torch.isnan(value_loss) or torch.isinf(value_loss):
                    value_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                if torch.isnan(entropy_loss) or torch.isinf(entropy_loss):
                    entropy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                # Total loss
                total_loss = policy_loss + self.config.vf_coef * value_loss + entropy_loss
                
                # Final safety check
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logger.warning("Total loss is NaN/inf, skipping this update")
                    continue
                
                # Backward pass
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                total_loss.backward()
                
                # Check for invalid gradients before clipping
                actor_params = list(self.model.parameters())
                critic_params = list(self.value_head.parameters())
                
                # Check if any gradients are NaN or inf
                actor_grads_valid = all(torch.isfinite(p.grad).all() for p in actor_params if p.grad is not None)
                critic_grads_valid = all(torch.isfinite(p.grad).all() for p in critic_params if p.grad is not None)
                
                if not actor_grads_valid or not critic_grads_valid:
                    logger.warning("Invalid gradients detected, skipping this update")
                    continue
                
                # Gradient clipping
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.value_head.parameters(), self.config.max_grad_norm
                )
                
                # Check if gradient norms are reasonable and scale if needed
                max_grad_norm = 1.0  # Much more conservative
                if actor_grad_norm > max_grad_norm or critic_grad_norm > max_grad_norm:
                    logger.warning(f"Gradient norms too high: actor={actor_grad_norm:.3f}, critic={critic_grad_norm:.3f}, scaling down")
                    # Scale gradients down
                    scale_factor = max_grad_norm / max(actor_grad_norm, critic_grad_norm)
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.grad *= scale_factor
                    for param in self.value_head.parameters():
                        if param.grad is not None:
                            param.grad *= scale_factor
                    logger.info(f"Gradients scaled by factor: {scale_factor:.3f}")
                
                # Step-by-step optimizer update to catch where weights become invalid
                if not self.actor_frozen:
                    self.actor_optimizer.step()
                    
                    # Check model weights after actor update
                    model_weights_valid = all(torch.isfinite(p).all() for p in self.model.parameters())
                    if not model_weights_valid:
                        logger.error("Model weights became invalid after actor update, restoring from backup")
                        # Restore model from backup
                        for name, param in self.model.named_parameters():
                            if name in self.model_backup:
                                param.data = self.model_backup[name].clone()
                        logger.info("Model restored from backup")
                        continue
                else:
                    logger.debug("Actor model frozen, skipping actor update")
                
                # Check if we should unfreeze the actor model
                if self.actor_frozen and self.actor_freeze_steps >= self.actor_unfreeze_threshold:
                    logger.info("Unfreezing actor model for full PPO training")
                    self.actor_frozen = False
                    for param in self.model.parameters():
                        param.requires_grad = True
                
                self.actor_freeze_steps += 1
                
                # Log research metrics
                if self.step_count % 5 == 0:  # More frequent logging
                    logger.info(f"Step {self.step_count}: Actor frozen={self.actor_frozen}, "
                              f"Value loss={value_loss.item():.6f}, "
                              f"Policy loss={policy_loss.item():.6f}, "
                              f"Mean reward={batch_data['rewards'].mean().item():.6f}")
                
                self.critic_optimizer.step()
                
                # Check model weights after critic update
                model_weights_valid = all(torch.isfinite(p).all() for p in self.model.parameters())
                if not model_weights_valid:
                    logger.error("Model weights became invalid after critic update, restoring from backup")
                    # Restore model from backup
                    for name, param in self.model.named_parameters():
                        if name in self.model_backup:
                            param.data = self.model_backup[name].clone()
                    logger.info("Model restored from backup")
                    continue
                
                # Debug: Check if gradients are actually flowing
                logger.debug(f"Actor grad norm: {actor_grad_norm.item():.6f}, Critic grad norm: {critic_grad_norm.item():.6f}")
                
                # Collect statistics
                update_stats['policy_losses'].append(policy_loss.item())
                update_stats['value_losses'].append(value_loss.item())
                update_stats['entropy_losses'].append(entropy_loss.item())
                update_stats['total_losses'].append(total_loss.item())
                update_stats['clipfracs'].append(policy_metrics['clipfrac'])
                update_stats['approx_kls'].append(policy_metrics['approx_kl'])
                update_stats['grad_norms'].append(actor_grad_norm.item())
                
                # Standard KL control
                if policy_metrics['approx_kl'] > 1.5 * self.config.target_kl:
                    logger.warning(f"Early stopping due to high KL: {policy_metrics['approx_kl']:.6f}")
                    break
                
                # Skip update if KL is extremely high
                if policy_metrics['approx_kl'] > 10.0:
                    logger.warning(f"Extremely high KL ({policy_metrics['approx_kl']:.6f}), skipping update to prevent instability")
                    continue
                
                # Check KL divergence for early stopping between epochs
                mean_kl = np.mean(update_stats['approx_kls'][-10:]) if update_stats['approx_kls'] else 0
                if mean_kl > 1.5 * self.config.target_kl:
                    logger.warning(f"Early stopping at epoch {epoch+1} due to high KL: {mean_kl:.6f}")
                    break
            
            # Update KL controller
            if self.kl_controller:
                mean_kl = np.mean(update_stats['approx_kls']) if update_stats['approx_kls'] else 0
                self.config.kl_coef = self.kl_controller.update(mean_kl)
            
            # Compute explained variance
            explained_var = compute_explained_variance(batch_data['values'], batch_data['returns'])
            
            # Return statistics with NaN handling
            def safe_mean(values):
                if not values:
                    return 0.0
                mean_val = np.mean(values)
                return 0.0 if np.isnan(mean_val) or np.isinf(mean_val) else mean_val
            
            stats = {
                'policy_loss': safe_mean(update_stats['policy_losses']),
                'value_loss': safe_mean(update_stats['value_losses']),
                'entropy_loss': safe_mean(update_stats['entropy_losses']),
                'total_loss': safe_mean(update_stats['total_losses']),
                'clipfrac': safe_mean(update_stats['clipfracs']),
                'approx_kl': safe_mean(update_stats['approx_kls']),
                'grad_norm': safe_mean(update_stats['grad_norms']),
                'explained_variance': 0.0 if np.isnan(explained_var) else explained_var,
                'kl_coef': self.config.kl_coef,
                'mean_reward': batch_data['rewards'].mean().item() if not torch.isnan(batch_data['rewards'].mean()) else 0.0,
                'mean_advantage': batch_data['advantages'].mean().item() if not torch.isnan(batch_data['advantages'].mean()) else 0.0,
                'mean_return': batch_data['returns'].mean().item() if not torch.isnan(batch_data['returns'].mean()) else 0.0,
                'reward_std': batch_data['rewards'].std().item() if not torch.isnan(batch_data['rewards'].std()) else 1.0
            }
            
            return stats
        
        except Exception as e:
            logger.error(f"PPO update failed: {e}")
            
            # If it's a CUDA error, try to reset the model state
            if "CUDA error" in str(e):
                logger.warning("Attempting to reset model state due to CUDA error")
                try:
                    # Clear gradients and reset optimizers
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    
                    # Reset model to eval mode
                    self.model.eval()
                    self.value_head.eval()
                    
                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as reset_error:
                    logger.error(f"Failed to reset model state: {reset_error}")
            
            # Return default statistics
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy_loss': 0.0,
                'total_loss': 0.0,
                'clipfrac': 0.0,
                'approx_kl': 0.0,
                'grad_norm': 0.0,
                'explained_variance': 0.0,
                'kl_coef': self.config.kl_coef,
                'mean_reward': 0.0,
                'mean_advantage': 0.0,
                'mean_return': 0.0,
                'reward_std': 0.0
            }
    
    def train_step(self, prompts: List[str], rollout_size: int = None) -> Dict[str, float]:
        """Complete enhanced PPO training step"""
        try:
            if rollout_size is None:
                rollout_size = len(prompts)
            
            # Collect rollouts
            logger.info(f"Collecting {rollout_size} rollouts...")
            buffer = self.collect_rollouts(prompts, rollout_size)
            
            # Check if buffer is empty
            if len(buffer.queries) == 0:
                logger.warning("Empty rollout buffer, skipping training step")
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy_loss': 0.0,
                    'total_loss': 0.0,
                    'clipfrac': 0.0,
                    'approx_kl': 0.0,
                    'grad_norm': 0.0,
                    'explained_variance': 0.0,
                    'kl_coef': self.config.kl_coef,
                    'mean_reward': 0.0,
                    'mean_advantage': 0.0,
                    'mean_return': 0.0,
                    'reward_std': 0.0
                }
            
            # PPO update
            logger.info("Performing PPO update...")
            stats = self.ppo_update(buffer)
            
            # Update step count
            self.step_count += 1
            
            # Log statistics
            self.stats.update(**stats)
            
            return stats
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            
            # If it's a CUDA error, we need to completely restart
            if "CUDA error" in str(e):
                logger.error("CUDA error detected - training cannot continue safely")
                logger.error("Please restart the training process")
                # Return a special flag to indicate CUDA corruption
                return {
                    'policy_loss': 0.0,
                    'value_loss': 0.0,
                    'entropy_loss': 0.0,
                    'total_loss': 0.0,
                    'clipfrac': 0.0,
                    'approx_kl': 0.0,
                    'grad_norm': 0.0,
                    'explained_variance': 0.0,
                    'kl_coef': self.config.kl_coef,
                    'mean_reward': 0.0,
                    'mean_advantage': 0.0,
                    'mean_return': 0.0,
                    'reward_std': 0.0,
                    'cuda_error': True  # Flag to indicate CUDA corruption
                }
            
            # Return default statistics
            return {
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy_loss': 0.0,
                'total_loss': 0.0,
                'clipfrac': 0.0,
                'approx_kl': 0.0,
                'grad_norm': 0.0,
                'explained_variance': 0.0,
                'kl_coef': self.config.kl_coef,
                'mean_reward': 0.0,
                'mean_advantage': 0.0,
                'mean_return': 0.0,
                'reward_std': 0.0
            }
    
    def save_checkpoint(self, path: str, epoch: int, additional_info: Dict = None):
        """Save training checkpoint with enhanced information"""
        checkpoint = {
            'epoch': epoch,
            'step_count': self.step_count,
            'model_state_dict': self.model.state_dict(),
            'value_head_state_dict': self.value_head.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config.__dict__,
            'kl_coef': self.config.kl_coef,
            'reward_stats': self.reward_stats
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.value_head.load_state_dict(checkpoint['value_head_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.step_count = checkpoint.get('step_count', 0)
        self.config.kl_coef = checkpoint.get('kl_coef', self.config.kl_coef)
        self.reward_stats = checkpoint.get('reward_stats', self.reward_stats)
        
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint
    
    def get_stats_summary(self) -> Dict[str, float]:
        """Get training statistics summary"""
        return self.stats.get_means()
    
    def reset_stats(self):
        """Reset training statistics"""
        self.stats.reset()

def create_enhanced_ppo_trainer(model, tokenizer, retriever, config_dict: Dict, device) -> EnhancedPPOTrainer:
    """Factory function to create enhanced PPO trainer"""
    
    # Extract enhanced PPO config
    ppo_config = EnhancedPPOConfig(
        clip_range=config_dict.get('clip_range', 0.2),
        vf_coef=config_dict.get('vf_coef', 0.5),
        ent_coef=config_dict.get('ent_coef', 0.01),
        gamma=config_dict.get('gamma', 0.99),
        gae_lambda=config_dict.get('gae_lambda', 0.95),
        ppo_epochs=config_dict.get('ppo_epochs', 4),
        minibatch_size=config_dict.get('minibatch_size', 64),
        max_grad_norm=config_dict.get('max_grad_norm', 1.0),
        target_kl=config_dict.get('target_kl', 0.01),
        learning_rate=float(config_dict.get('learning_rate', 1e-5)),
        normalize_advantages=config_dict.get('normalize_advantages', True),
        value_clip=config_dict.get('value_clip', True),
        adaptive_kl=config_dict.get('adaptive_kl', True),
        lambda_consistency=config_dict.get('lambda_consistency', 0.5),
        lambda_retrieval=config_dict.get('lambda_retrieval', 0.1),
        consistency_method=config_dict.get('consistency_method', 'spearman'),
        num_candidates=config_dict.get('num_candidates', 3),
        use_gae=config_dict.get('use_gae', True),
        normalize_rewards=config_dict.get('normalize_rewards', True),
        reward_clipping=config_dict.get('reward_clipping', True),
        reward_clip_range=config_dict.get('reward_clip_range', 10.0)
    )
    
    return EnhancedPPOTrainer(model, tokenizer, retriever, ppo_config, device)


