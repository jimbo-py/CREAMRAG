#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Training Script with CREAM Consistency
Combines DPO training with consistency rewards for improved RAG performance.

This script implements:
1. DPO training with policy vs reference model comparison
2. Consistency rewards using Spearman/Kendall correlation
3. RAG-specific enhancements with retrieval consistency
4. Comprehensive logging and evaluation

Usage:
    python dpo_train.py --config config.yaml
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from scipy.stats import spearmanr, kendalltau, rankdata
from tqdm import tqdm
# import wandb  # Removed wandb dependency

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.consistency import calc_consistency, ConsistencyMethod
from agent.rag_retriever import LlamaRetriever
from agent.reward_model import RewardModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def apply_lora_to_model(model, config):
    """Apply LoRA to the model if enabled in config."""
    if hasattr(config, 'use_lora') and config.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=getattr(config, 'lora_r', 16),
            lora_alpha=getattr(config, 'lora_alpha', 32),
            lora_dropout=getattr(config, 'lora_dropout', 0.1),
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, lora_config)
        logger.info("Applied LoRA to model")
    return model

class ConsistencyMethod(Enum):
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    TOPORDER = "toporder"

@dataclass
class DPOConfig:
    """Configuration for DPO training"""
    # Model settings
    policy_model_name: str = "gpt2"
    reference_model_name: Optional[str] = None  # If None, uses policy model as reference
    
    # Training parameters
    learning_rate: float = 1e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Generation parameters
    max_input_length: int = 2048
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    do_sample: bool = True
    
    # DPO parameters
    beta: float = 0.1  # DPO beta parameter
    use_dpo_delta: bool = True  # Use policy - reference reward
    
    # Consistency parameters
    consistency_method: ConsistencyMethod = ConsistencyMethod.SPEARMAN
    lambda_consistency: float = 0.5  # Weight for consistency reward
    lambda_retrieval: float = 0.1    # Weight for retrieval consistency
    num_candidates: int = 3          # Number of candidates for consistency evaluation
    
    # Data parameters
    train_data_path: str = "data/train.jsonl"
    eval_data_path: Optional[str] = None
    max_samples: Optional[int] = None
    
    # Output settings
    output_dir: str = "checkpoints/dpo_training"
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    
    # Device settings
    device: str = "auto"
    load_in_8bit: bool = False
    use_8bit: bool = False
    use_4bit: bool = False
    bf16: bool = True
    
    # RAG settings
    use_rag: bool = False
    rag_index_path: Optional[str] = None
    rag_k: int = 3
    
    # LoRA settings
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    
    # Random seed
    seed: int = 42

class PreferenceDataset(Dataset):
    """Dataset for DPO training with preference pairs"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path: str) -> List[Dict]:
        """Load preference data from JSONL file"""
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Expected format: {"prompt": "...", "chosen": "...", "rejected": "..."}
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        
        # Tokenize chosen response
        chosen_text = prompt + chosen
        chosen_encoding = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize rejected response
        rejected_text = prompt + rejected
        rejected_encoding = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(0),
        }

class DPOTrainer:
    """DPO Trainer with consistency rewards"""
    
    def __init__(self, config: DPOConfig):
        self.config = config
        self.device = self._setup_device()
        
        # Set random seed
        set_seed(config.seed)
        
        # Initialize models
        self.policy_model, self.reference_model, self.tokenizer = self._setup_models()
        
        # Initialize reward models
        self.reward_model = self._setup_reward_model()
        
        # Initialize RAG retriever if needed
        self.retriever = self._setup_retriever()
        
        # Setup datasets
        self.train_dataset, self.eval_dataset = self._setup_datasets()
        
        # Initialize training arguments
        self.training_args = self._setup_training_args()
        
        # Initialize trainer
        self.trainer = self._setup_trainer()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_device(self) -> str:
        """Setup device for training"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def _setup_models(self) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
        """Setup policy and reference models with LoRA and quantization support"""
        logger.info(f"Loading policy model: {self.config.policy_model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.policy_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load policy model with optimized CUDA settings for A100
        model_kwargs = {
            "device_map": "auto" if self.config.device == "auto" else None,
            "torch_dtype": torch.bfloat16 if self.config.bf16 else torch.float32,
        }
        
        # Add flash attention if enabled
        if hasattr(self.config, 'use_flash_attention') and self.config.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Add quantization if enabled
        if hasattr(self.config, 'use_4bit') and self.config.use_4bit:
            model_kwargs["load_in_4bit"] = True
        elif hasattr(self.config, 'use_8bit') and self.config.use_8bit:
            model_kwargs["load_in_8bit"] = True
        
        policy_model = AutoModelForCausalLM.from_pretrained(
            self.config.policy_model_name, **model_kwargs
        )
        
        # Apply LoRA if enabled
        policy_model = apply_lora_to_model(policy_model, self.config)
        
        # Load reference model (or use policy model as reference)
        if self.config.reference_model_name:
            logger.info(f"Loading reference model: {self.config.reference_model_name}")
            reference_model = AutoModelForCausalLM.from_pretrained(
                self.config.reference_model_name, **model_kwargs
            )
            reference_model = apply_lora_to_model(reference_model, self.config)
        else:
            logger.info("Using policy model as reference model")
            reference_model = policy_model
        
        return policy_model, reference_model, tokenizer
    
    def _setup_reward_model(self) -> RewardModel:
        """Setup reward model for likelihood-based evaluation"""
        return RewardModel(
            model=self.policy_model,
            tokenizer=self.tokenizer,
            device=self.device
        )
    
    def _setup_retriever(self) -> Optional[LlamaRetriever]:
        """Setup RAG retriever if needed"""
        if self.config.use_rag and self.config.rag_index_path:
            logger.info(f"Loading RAG retriever from: {self.config.rag_index_path}")
            return LlamaRetriever(
                model_name=self.config.policy_model_name,
                device=self.device,
                max_length=self.config.max_input_length,
                use_4bit=self.config.use_4bit,
                use_8bit=self.config.use_8bit,
                use_flash_attention=False
            )
        return None
    
    def _setup_datasets(self) -> Tuple[PreferenceDataset, Optional[PreferenceDataset]]:
        """Setup training and evaluation datasets"""
        logger.info(f"Loading training data from: {self.config.train_data_path}")
        train_dataset = PreferenceDataset(
            self.config.train_data_path,
            self.tokenizer,
            self.config.max_input_length
        )
        
        eval_dataset = None
        if self.config.eval_data_path:
            logger.info(f"Loading evaluation data from: {self.config.eval_data_path}")
            eval_dataset = PreferenceDataset(
                self.config.eval_data_path,
                self.tokenizer,
                self.config.max_input_length
            )
        
        return train_dataset, eval_dataset
    
    def _setup_training_args(self) -> TrainingArguments:
        """Setup training arguments"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            bf16=self.config.bf16,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb for now
        )
    
    def _setup_trainer(self) -> 'DPOCustomTrainer':
        """Setup custom DPO trainer"""
        return DPOCustomTrainer(
            model=self.policy_model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            config=self.config,
            reward_model=self.reward_model,
            reference_model=self.reference_model,
        )
    
    def _setup_logging(self):
        """Setup logging"""
        logger.info("Logging setup completed")
    
    def train(self):
        """Start DPO training"""
        logger.info("Starting DPO training...")
        
        try:
            # Train the model
            train_result = self.trainer.train()
            
            # Save the final model
            self.trainer.save_model()
            
            # Log final metrics
            logger.info("Training completed successfully!")
            logger.info(f"Final loss: {train_result.training_loss}")
            
            # Log final metrics
            logger.info(f"Final loss: {train_result.training_loss}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate(self):
        """Evaluate the trained model"""
        if self.eval_dataset:
            logger.info("Evaluating model...")
            eval_results = self.trainer.evaluate()
            logger.info(f"Evaluation results: {eval_results}")
            
            # Log evaluation results
            logger.info(f"Evaluation results: {eval_results}")
            
            return eval_results
        else:
            logger.info("No evaluation dataset provided, skipping evaluation")
            return None

class DPOCustomTrainer(Trainer):
    """Custom trainer for DPO with consistency rewards"""
    
    def __init__(self, config: DPOConfig, reward_model: RewardModel, 
                 reference_model: AutoModelForCausalLM, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.reward_model = reward_model
        self.reference_model = reference_model
        
        # Set models to appropriate devices
        self.model.to(self.device)
        if self.reference_model != self.model:
            self.reference_model.to(self.device)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute DPO loss with consistency rewards"""
        batch_size = inputs["chosen_input_ids"].shape[0]
        
        # Get logits for chosen and rejected responses
        chosen_logits = model(
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"]
        ).logits
        
        rejected_logits = model(
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"]
        ).logits
        
        # Compute log probabilities
        chosen_log_probs = self._compute_log_probs(
            chosen_logits, inputs["chosen_input_ids"]
        )
        rejected_log_probs = self._compute_log_probs(
            rejected_logits, inputs["rejected_input_ids"]
        )
        
        # Get reference model log probabilities
        with torch.no_grad():
            ref_chosen_logits = self.reference_model(
                input_ids=inputs["chosen_input_ids"],
                attention_mask=inputs["chosen_attention_mask"]
            ).logits
            
            ref_rejected_logits = self.reference_model(
                input_ids=inputs["rejected_input_ids"],
                attention_mask=inputs["rejected_attention_mask"]
            ).logits
            
            ref_chosen_log_probs = self._compute_log_probs(
                ref_chosen_logits, inputs["chosen_input_ids"]
            )
            ref_rejected_log_probs = self._compute_log_probs(
                ref_rejected_logits, inputs["rejected_input_ids"]
            )
        
        # Compute DPO rewards
        chosen_rewards = self.config.beta * (chosen_log_probs - ref_chosen_log_probs)
        rejected_rewards = self.config.beta * (rejected_log_probs - ref_rejected_log_probs)
        
        # Compute consistency rewards
        consistency_rewards = []
        for i in range(batch_size):
            prompt = inputs["prompt"][i]
            chosen_response = inputs["chosen"][i]
            rejected_response = inputs["rejected"][i]
            
            # Compute consistency for chosen response
            chosen_consistency = self.reward_model.compute_consistency_reward(
                prompt, chosen_response
            )
            
            # Compute consistency for rejected response
            rejected_consistency = self.reward_model.compute_consistency_reward(
                prompt, rejected_response
            )
            
            consistency_rewards.append({
                "chosen": chosen_consistency,
                "rejected": rejected_consistency
            })
        
        # Combine DPO and consistency rewards
        chosen_total_rewards = chosen_rewards + self.config.lambda_consistency * torch.tensor(
            [cr["chosen"] for cr in consistency_rewards], device=self.device
        )
        rejected_total_rewards = rejected_rewards + self.config.lambda_consistency * torch.tensor(
            [cr["rejected"] for cr in consistency_rewards], device=self.device
        )
        
        # Compute DPO loss
        dpo_logits = chosen_total_rewards - rejected_total_rewards
        dpo_loss = -F.logsigmoid(dpo_logits).mean()
        
        # Log metrics
        metrics = {
            "dpo_loss": dpo_loss.item(),
            "chosen_rewards": chosen_total_rewards.mean().item(),
            "rejected_rewards": rejected_total_rewards.mean().item(),
            "dpo_accuracy": (dpo_logits > 0).float().mean().item(),
            "chosen_consistency": np.mean([cr["chosen"] for cr in consistency_rewards]),
            "rejected_consistency": np.mean([cr["rejected"] for cr in consistency_rewards]),
        }
        
        self.log(metrics)
        
        return (dpo_loss, None) if return_outputs else dpo_loss
    
    def _compute_log_probs(self, logits, input_ids):
        """Compute log probabilities for the response tokens"""
        # Shift logits and labels for next-token prediction
        logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding tokens
        mask = (labels != self.tokenizer.pad_token_id).float()
        masked_log_probs = token_log_probs * mask
        
        # Sum log probabilities
        return masked_log_probs.sum(dim=-1)

def create_sample_data():
    """Create sample preference data for testing"""
    sample_data = [
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris, which is a beautiful city known for its culture, art, and history.",
            "rejected": "France has many cities but I'm not sure about the capital."
        },
        {
            "prompt": "Explain photosynthesis.",
            "chosen": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This process occurs in the chloroplasts of plant cells.",
            "rejected": "Plants do something with sunlight and make food somehow."
        },
        {
            "prompt": "What are the benefits of exercise?",
            "chosen": "Exercise provides numerous benefits including improved cardiovascular health, stronger muscles and bones, better mental health, increased energy levels, and weight management.",
            "rejected": "Exercise is good for you and makes you tired."
        }
    ]
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save sample data
    with open("data/train.jsonl", "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    logger.info("Created sample training data in data/train.jsonl")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="DPO Training with Consistency Rewards")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--create-sample-data", action="store_true", 
                       help="Create sample data for testing")
    parser.add_argument("--model", type=str, default="gpt2", 
                       help="Model name or path")
    parser.add_argument("--output-dir", type=str, default="checkpoints/dpo_training",
                       help="Output directory for checkpoints")
    parser.add_argument("--train-data", type=str, default="data/train.jsonl",
                       help="Path to training data")
    parser.add_argument("--eval-data", type=str, default=None,
                       help="Path to evaluation data")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1,
                       help="DPO beta parameter")
    parser.add_argument("--lambda-consistency", type=float, default=0.5,
                       help="Consistency reward weight")
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample_data:
        create_sample_data()
        return
    
    # Load config or create from arguments
    if args.config and os.path.exists(args.config):
        # Load from config file
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = DPOConfig(**config_dict)
    else:
        # Create config from arguments
        config = DPOConfig(
            policy_model_name=args.model,
            output_dir=args.output_dir,
            train_data_path=args.train_data,
            eval_data_path=args.eval_data,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            beta=args.beta,
            lambda_consistency=args.lambda_consistency,
        )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = DPOTrainer(config)
    
    # Start training
    trainer.train()
    
    # Evaluate
    trainer.evaluate()
    
    logger.info("DPO training completed successfully!")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
