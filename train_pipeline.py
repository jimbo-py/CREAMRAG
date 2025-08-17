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
    
    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using running statistics"""
        if self.config.normalize_rewards:
            # Update running statistics
            if self.step_count == 0:
                self.reward_stats['mean'] = rewards.mean().item()
                self.reward_stats['std'] = rewards.std().item()
            else:
                # Exponential moving average
                alpha = 0.99
                self.reward_stats['mean'] = alpha * self.reward_stats['mean'] + (1 - alpha) * rewards.mean().item()
                self.reward_stats['std'] = alpha * self.reward_stats['std'] + (1 - alpha) * rewards.std().item()
            
            # Normalize
            normalized = (rewards - self.reward_stats['mean']) / (self.reward_stats['std'] + 1e-8)
            
            # Clip if requested
            if self.config.reward_clipping:
                normalized = torch.clamp(normalized, -self.config.reward_clip_range, self.config.reward_clip_range)
            
            return normalized
        else:
            return rewards
    
    def generate_response_with_log_probs(self, prompt: str, max_new_tokens: int = 32, 
                                       temperature: float = 0.7) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Generate response and return text, log_probs, and values with better handling"""
        
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
            response_log_probs = F.log_softmax(logits, dim=-1)
            
            # Get log probs for actual tokens
            token_log_probs = torch.gather(
                response_log_probs, 
                1, 
                response_ids.unsqueeze(1)
            ).squeeze(1)  # [response_len]
            
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
        
        for i, prompt in enumerate(prompts[:rollout_size]):
            try:
                # Generate response
                response, log_prob, value = self.generate_response_with_log_probs(prompt)
                
                # Compute enhanced reward with consistency
                reward = self.reward_model.compute_reward(prompt, response)
                
                # Add to buffer
                buffer.add(
                    query=prompt,
                    response=response,
                    reward=reward,
                    value=value.item(),
                    log_prob=log_prob.item(),
                    mask=True
                )
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Collected {i + 1}/{rollout_size} rollouts")
                    
            except Exception as e:
                logger.warning(f"Failed to generate rollout {i}: {e}")
                continue
        
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
                
                # Compute log probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = torch.gather(log_probs, 1, response_ids.unsqueeze(1)).squeeze(1)
                sequence_log_prob = token_log_probs.sum()
                
                # Compute entropy
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(-1).mean()
                
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
                
                # Compute losses
                policy_loss, policy_metrics = self.loss_fn.compute_policy_loss(
                    current_log_probs, mb_old_log_probs, mb_advantages, self.config.clip_range
                )
                
                value_loss, value_metrics = self.loss_fn.compute_value_loss(
                    current_values, mb_returns, mb_old_values, 
                    self.config.clip_range, self.config.value_clip
                )
                
                entropy_loss = -self.config.ent_coef * entropy
                
                # Total loss
                total_loss = policy_loss + self.config.vf_coef * value_loss + entropy_loss
                
                # Backward pass
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                total_loss.backward()
                
                # Gradient clipping
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.value_head.parameters(), self.config.max_grad_norm
                )
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Collect statistics
                update_stats['policy_losses'].append(policy_loss.item())
                update_stats['value_losses'].append(value_loss.item())
                update_stats['entropy_losses'].append(entropy_loss.item())
                update_stats['total_losses'].append(total_loss.item())
                update_stats['clipfracs'].append(policy_metrics['clipfrac'])
                update_stats['approx_kls'].append(policy_metrics['approx_kl'])
                update_stats['grad_norms'].append(actor_grad_norm.item())
                
                # Early stopping if KL divergence is too high
                if policy_metrics['approx_kl'] > 1.5 * self.config.target_kl:
                    logger.warning(f"Early stopping due to high KL: {policy_metrics['approx_kl']:.6f}")
                    break
            
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
        
        # Return statistics
        stats = {
            'policy_loss': np.mean(update_stats['policy_losses']),
            'value_loss': np.mean(update_stats['value_losses']),
            'entropy_loss': np.mean(update_stats['entropy_losses']),
            'total_loss': np.mean(update_stats['total_losses']),
            'clipfrac': np.mean(update_stats['clipfracs']),
            'approx_kl': np.mean(update_stats['approx_kls']),
            'grad_norm': np.mean(update_stats['grad_norms']),
            'explained_variance': explained_var,
            'kl_coef': self.config.kl_coef,
            'mean_reward': batch_data['rewards'].mean().item(),
            'mean_advantage': batch_data['advantages'].mean().item(),
            'mean_return': batch_data['returns'].mean().item(),
            'reward_std': batch_data['rewards'].std().item()
        }
        
        return stats
    
    def train_step(self, prompts: List[str], rollout_size: int = None) -> Dict[str, float]:
        """Complete enhanced PPO training step"""
        if rollout_size is None:
            rollout_size = len(prompts)
        
        # Collect rollouts
        logger.info(f"Collecting {rollout_size} rollouts...")
        buffer = self.collect_rollouts(prompts, rollout_size)
        
        # PPO update
        logger.info("Performing PPO update...")
        stats = self.ppo_update(buffer)
        
        # Update step count
        self.step_count += 1
        
        # Log statistics
        self.stats.update(**stats)
        
        return stats
    
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
