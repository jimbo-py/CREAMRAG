"""
PPO utilities for RAG training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RolloutBuffer:
    """Buffer to store rollout data for PPO training"""
    
    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.clear()
    
    def clear(self):
        """Clear the buffer"""
        self.queries = []
        self.responses = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.returns = []
        self.masks = []
        self.ptr = 0
        self.full = False
    
    def add(self, query: str, response: str, reward: float, 
            value: float, log_prob: float, mask: bool = True):
        """Add experience to buffer"""
        if self.ptr < self.buffer_size:
            if len(self.queries) <= self.ptr:
                self.queries.append(query)
                self.responses.append(response)
                self.rewards.append(reward)
                self.values.append(value)
                self.log_probs.append(log_prob)
                self.masks.append(mask)
            else:
                self.queries[self.ptr] = query
                self.responses[self.ptr] = response
                self.rewards[self.ptr] = reward
                self.values[self.ptr] = value
                self.log_probs[self.ptr] = log_prob
                self.masks[self.ptr] = mask
            
            self.ptr += 1
            if self.ptr >= self.buffer_size:
                self.full = True
                self.ptr = 0
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all buffer contents as tensors"""
        size = self.buffer_size if self.full else self.ptr
        
        return {
            'queries': self.queries[:size],
            'responses': self.responses[:size],
            'rewards': torch.tensor(self.rewards[:size], dtype=torch.float32, device=self.device),
            'values': torch.tensor(self.values[:size], dtype=torch.float32, device=self.device),
            'log_probs': torch.tensor(self.log_probs[:size], dtype=torch.float32, device=self.device),
            'advantages': torch.tensor(self.advantages[:size], dtype=torch.float32, device=self.device),
            'returns': torch.tensor(self.returns[:size], dtype=torch.float32, device=self.device),
            'masks': torch.tensor(self.masks[:size], dtype=torch.bool, device=self.device)
        }
    
    def compute_advantages_and_returns(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute GAE advantages and returns"""
        size = self.buffer_size if self.full else self.ptr
        rewards = np.array(self.rewards[:size])
        values = np.array(self.values[:size])
        masks = np.array(self.masks[:size], dtype=np.float32)
        
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        
        # Compute advantages using GAE
        gae = 0.0
        for t in reversed(range(size)):
            if t == size - 1:
                next_value = 0.0
                next_mask = 0.0
            else:
                next_value = values[t + 1]
                next_mask = masks[t + 1]
            
            delta = rewards[t] + gamma * next_value * next_mask - values[t]
            gae = delta + gamma * gae_lambda * next_mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.advantages = advantages.tolist()
        self.returns = returns.tolist()

class AdaptiveKLController:
    """Adaptive KL divergence controller for PPO"""
    
    def __init__(self, target_kl: float = 0.01, alpha: float = 1.5):
        self.target_kl = target_kl
        self.alpha = alpha
        self.kl_coef = 0.1  # Initial coefficient
    
    def update(self, current_kl: float):
        """Update KL coefficient based on current KL divergence"""
        if current_kl > self.alpha * self.target_kl:
            # KL too high, increase coefficient
            self.kl_coef *= 1.5
        elif current_kl < self.target_kl / self.alpha:
            # KL too low, decrease coefficient
            self.kl_coef /= 1.5
        
        # Clamp coefficient
        self.kl_coef = np.clip(self.kl_coef, 0.001, 10.0)
        
        return self.kl_coef

class PPOLoss:
    """PPO loss computation utilities"""
    
    @staticmethod
    def compute_policy_loss(log_probs: torch.Tensor, 
                          old_log_probs: torch.Tensor,
                          advantages: torch.Tensor,
                          clip_range: float = 0.2) -> Tuple[torch.Tensor, Dict]:
        """Compute clipped policy loss"""
        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute metrics
        with torch.no_grad():
            clipfrac = ((ratio - 1.0).abs() > clip_range).float().mean()
            approx_kl = (old_log_probs - log_probs).mean()
            
        metrics = {
            'policy_loss': policy_loss.item(),
            'clipfrac': clipfrac.item(),
            'approx_kl': approx_kl.item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_std': ratio.std().item()
        }
        
        return policy_loss, metrics
    
    @staticmethod
    def compute_value_loss(values: torch.Tensor, 
                         returns: torch.Tensor,
                         old_values: torch.Tensor = None,
                         clip_range: float = 0.2,
                         use_clipping: bool = True) -> Tuple[torch.Tensor, Dict]:
        """Compute value function loss with optional clipping"""
        if use_clipping and old_values is not None:
            # Clipped value loss (similar to OpenAI's implementation)
            values_clipped = old_values + torch.clamp(
                values - old_values, -clip_range, clip_range
            )
            value_loss1 = F.mse_loss(values, returns, reduction='none')
            value_loss2 = F.mse_loss(values_clipped, returns, reduction='none')
            value_loss = torch.max(value_loss1, value_loss2).mean()
        else:
            value_loss = F.mse_loss(values, returns)
        
        metrics = {
            'value_loss': value_loss.item(),
            'values_mean': values.mean().item(),
            'returns_mean': returns.mean().item(),
            'value_error': (values - returns).abs().mean().item()
        }
        
        return value_loss, metrics
    
    @staticmethod
    def compute_entropy_loss(logits: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute entropy loss for exploration bonus"""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(-1).mean()
        
        metrics = {
            'entropy': entropy.item(),
            'perplexity': torch.exp(entropy).item()
        }
        
        return -entropy, metrics  # Negative because we want to maximize entropy

class ExperienceBuffer:
    """Enhanced experience buffer for PPO with text data"""
    
    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.buffer = []
        self.position = 0
    
    def push(self, experience: Dict):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Dict:
        """Sample batch of experiences"""
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Collate batch
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            if key in ['query', 'response']:
                collated[key] = [item[key] for item in batch]
            else:
                values = [item[key] for item in batch]
                if isinstance(values[0], torch.Tensor):
                    collated[key] = torch.stack(values).to(self.device)
                else:
                    collated[key] = torch.tensor(values, device=self.device)
        
        return collated
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.position = 0

class PPOScheduler:
    """Learning rate and clipping scheduler for PPO"""
    
    def __init__(self, initial_lr: float, initial_clip_range: float,
                 total_steps: int, decay_type: str = 'linear'):
        self.initial_lr = initial_lr
        self.initial_clip_range = initial_clip_range
        self.total_steps = total_steps
        self.decay_type = decay_type
        self.current_step = 0
    
    def step(self) -> Tuple[float, float]:
        """Get current learning rate and clip range"""
        progress = self.current_step / self.total_steps
        progress = min(progress, 1.0)
        
        if self.decay_type == 'linear':
            decay_factor = 1.0 - progress
        elif self.decay_type == 'cosine':
            decay_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
        else:
            decay_factor = 1.0
        
        current_lr = self.initial_lr * decay_factor
        current_clip_range = self.initial_clip_range * decay_factor
        
        self.current_step += 1
        
        return current_lr, current_clip_range

def compute_explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """Compute explained variance of value function"""
    with torch.no_grad():
        var_returns = returns.var()
        if var_returns == 0:
            return 0.0
        
        residual_var = (returns - values).var()
        explained_var = 1.0 - (residual_var / var_returns)
        
        return explained_var.item()

def whiten(tensor: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Whiten (normalize) a tensor"""
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / (std + epsilon)

def get_gradient_norm(model: nn.Module) -> float:
    """Get gradient norm of model parameters"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    return total_norm ** 0.5

class PPOStats:
    """Statistics tracker for PPO training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'rewards': [],
            'advantages': [],
            'returns': [],
            'clipfrac': [],
            'approx_kl': [],
            'explained_variance': [],
            'grad_norm': []
        }
    
    def update(self, **kwargs):
        """Update statistics"""
        for key, value in kwargs.items():
            if key in self.stats:
                self.stats[key].append(value)
    
    def get_means(self) -> Dict[str, float]:
        """Get mean values of all statistics"""
        means = {}
        for key, values in self.stats.items():
            if values:
                means[f'mean_{key}'] = np.mean(values)
                means[f'std_{key}'] = np.std(values)
        
        return means
    
    def log_summary(self, logger, prefix: str = ""):
        """Log summary statistics"""
        means = self.get_means()
        
        for key, value in means.items():
            if 'mean_' in key:
                clean_key = key.replace('mean_', '')
                logger.info(f"{prefix}{clean_key}: {value:.6f}")

