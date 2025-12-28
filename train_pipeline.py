"""
Enhanced Training Pipeline for CREAM-RAG with DPO and Consistency Rewards
"""

import yaml
import os
import torch
import torch.nn.functional as F
import numpy as np
import json
import random
import logging
from huggingface_hub import login
from typing import List, Dict, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModel, AutoProcessor
from transformers import DataCollatorWithPadding
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import gc
from tqdm import tqdm

# Import agent components for CREAM-RAG
from agent.consistency import calc_consistency, ConsistencyMethod
from agent.rag_retriever import LlamaRetriever
from agent.reward_model import RewardModel

# Import CUDA optimization utilities
try:
    from cuda_optimization import setup_cuda_optimizations, monitor_gpu_memory, get_optimal_batch_size
except ImportError:
    # Fallback if cuda_optimization.py is not available
    def setup_cuda_optimizations():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def monitor_gpu_memory():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def get_optimal_batch_size(model_size_gb, sequence_length=2048):
        return 4  # Default fallback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup CUDA optimizations
setup_cuda_optimizations()

# Disable wandb completely
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

# Login to HuggingFace
login(token="")

@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    epoch: int
    step: int
    dpo_loss: float
    consistency_loss: float
    total_loss: float
    consistency_score: float
    retrieval_consistency: float

def load_documents(path: str) -> List[str]:
    """Load documents from various formats"""
    documents = []
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text") or obj.get("document") or obj.get("content")
                if text:
                    documents.append(text)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    documents.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("document") or item.get("content")
                    if text:
                        documents.append(text)
    return documents

def create_preference_pairs_from_qa_data(data: List[Dict], max_pairs: int = 10000) -> List[Dict]:
    """Create preference pairs for DPO training from QA dataset"""
    preference_pairs = []
    
    for i, item in enumerate(data[:max_pairs]):
        question = item.get('question', '')
        context = item.get('context', '')
        answers = item.get('answers', [])
        
        # Skip if no question
        if not question:
            continue
        
        # Handle context - can be empty string for some datasets (e.g., natural_questions)
        has_context = context and context.strip()
        has_answers = answers and len(answers) > 0
        
        # Skip if neither context nor answers are available
        if not has_context and not has_answers:
            continue
            
        # Create a good answer (using context and/or answers)
        if has_context and has_answers:
            # Both context and answers available - use both
            good_answer = f"Based on the provided context, {answers[0]}. This information comes from the source material which states: {context[:150]}..."
        elif has_context:
            # Only context available - use context
            good_answer = f"Based on the context provided: {context[:200]}... The answer would be that more specific information is needed to provide a complete response."
        elif has_answers:
            # Only answers available (no context) - use answer directly
            good_answer = f"{answers[0]}"
        else:
            # Should not reach here due to check above, but handle gracefully
            continue
        
        # Create a bad answer (wrong or misleading)
        bad_answer = f"I'm not sure about the exact details, but I think the answer might be something related to that topic. Without more specific information, I can't give you a definitive answer."
        
        preference_pairs.append({
            "prompt": question,
            "chosen": good_answer,
            "rejected": bad_answer
        })
    
    return preference_pairs

def tokenize_function(examples, tokenizer, max_length=1024):
    """Tokenize the examples for DPO"""
    prompts = examples["prompt"]
    chosen = examples["chosen"]
    rejected = examples["rejected"]
    
    # Tokenize chosen responses
    chosen_tokens = tokenizer(
        [p + c for p, c in zip(prompts, chosen)],
        truncation=True,
        padding=False,  # Don't pad here, let the collator handle it
        max_length=max_length,
        return_tensors=None  # Return lists, not tensors
    )
    
    # Tokenize rejected responses
    rejected_tokens = tokenizer(
        [p + r for p, r in zip(prompts, rejected)],
        truncation=True,
        padding=False,  # Don't pad here, let the collator handle it
        max_length=max_length,
        return_tensors=None  # Return lists, not tensors
    )
    
    return {
        "chosen_input_ids": chosen_tokens["input_ids"],
        "chosen_attention_mask": chosen_tokens["attention_mask"],
        "rejected_input_ids": rejected_tokens["input_ids"],
        "rejected_attention_mask": rejected_tokens["attention_mask"],
    }

def dpo_loss(chosen_logps, rejected_logps, beta=0.1):
    """Compute DPO loss"""
    # Ensure we're working with tensors that require gradients
    chosen_rewards = chosen_logps.sum(dim=-1)  # Sum over sequence length
    rejected_rewards = rejected_logps.sum(dim=-1)  # Sum over sequence length
    
    # Compute DPO loss
    losses = -F.logsigmoid(beta * (chosen_rewards - rejected_rewards))
    return losses.mean()

def create_dpo_data_collator(tokenizer):
    """Create a data collator for DPO training"""
    def collate_fn(batch):
        # Separate chosen and rejected inputs
        chosen_input_ids = [item["chosen_input_ids"] for item in batch]
        chosen_attention_mask = [item["chosen_attention_mask"] for item in batch]
        rejected_input_ids = [item["rejected_input_ids"] for item in batch]
        rejected_attention_mask = [item["rejected_attention_mask"] for item in batch]
        
        # Pad to the maximum length in the batch
        max_chosen_len = max(len(ids) for ids in chosen_input_ids)
        max_rejected_len = max(len(ids) for ids in rejected_input_ids)
        max_len = max(max_chosen_len, max_rejected_len)
        
        # Pad chosen sequences
        padded_chosen_input_ids = []
        padded_chosen_attention_mask = []
        for ids, mask in zip(chosen_input_ids, chosen_attention_mask):
            padding_len = max_len - len(ids)
            padded_chosen_input_ids.append(ids + [tokenizer.pad_token_id] * padding_len)
            padded_chosen_attention_mask.append(mask + [0] * padding_len)
        
        # Pad rejected sequences
        padded_rejected_input_ids = []
        padded_rejected_attention_mask = []
        for ids, mask in zip(rejected_input_ids, rejected_attention_mask):
            padding_len = max_len - len(ids)
            padded_rejected_input_ids.append(ids + [tokenizer.pad_token_id] * padding_len)
            padded_rejected_attention_mask.append(mask + [0] * padding_len)
        
        # Convert to tensors and ensure they're the right type
        result = {
            "chosen_input_ids": torch.tensor(padded_chosen_input_ids, dtype=torch.long),
            "chosen_attention_mask": torch.tensor(padded_chosen_attention_mask, dtype=torch.long),
            "rejected_input_ids": torch.tensor(padded_rejected_input_ids, dtype=torch.long),
            "rejected_attention_mask": torch.tensor(padded_rejected_attention_mask, dtype=torch.long),
        }
        
        return result
    
    return collate_fn

class DPOTrainer(Trainer):
    """CREAM-RAG DPO Trainer with consistency rewards"""
    
    def __init__(self, model, tokenizer, beta=0.1, **kwargs):
        super().__init__(model=model, **kwargs)
        self.tokenizer = tokenizer
        self.beta = beta
        
        # Initialize agent components for CREAM-RAG
        self.reward_model = RewardModel(
            model=model,
            tokenizer=tokenizer,
            device=model.device
        )
        
        # Initialize consistency method
        self.consistency_method = ConsistencyMethod.SPEARMAN
        self.lambda_consistency = 1.0  # Weight for consistency reward (reduced since individual rewards are larger)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """Stable DPO loss computation with normalized and clipped rewards"""
            device = model.device
            # Required inputs from collator
            chosen_input_ids = inputs["chosen_input_ids"].to(device)
            chosen_attention_mask = inputs["chosen_attention_mask"].to(device)
            rejected_input_ids = inputs["rejected_input_ids"].to(device)
            rejected_attention_mask = inputs["rejected_attention_mask"].to(device)

            def sequence_logprobs(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                # Next-token prediction: shift logits and labels
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # [B, T, V]
                logits = logits[:, :-1, :]  # predict token t+1
                labels = input_ids[:, 1:]   # the next tokens
                mask = attention_mask[:, 1:]  # align mask with labels
                log_probs = F.log_softmax(logits, dim=-1)
                token_logprobs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B, T-1]
                # Zero out padding positions
                token_logprobs = token_logprobs * mask
                # Sum per sequence
                return token_logprobs.sum(dim=-1)  # [B]

            # --- Compute per-sample rewards as summed log-probs ---
            chosen_rewards = sequence_logprobs(chosen_input_ids, chosen_attention_mask)
            rejected_rewards = sequence_logprobs(rejected_input_ids, rejected_attention_mask)

            # --- Batch normalization for numerical stability ---
            all_rewards = torch.cat([chosen_rewards, rejected_rewards], dim=0)
            mean = all_rewards.mean()
            std = all_rewards.std().clamp(min=1e-6)
            chosen_rewards = (chosen_rewards - mean) / std
            rejected_rewards = (rejected_rewards - mean) / std

            # --- Compute (optional) consistency rewards; fall back to zeros if raw texts absent ---
            batch_size = chosen_input_ids.size(0)
            consistency_chosen_vals = [0.0] * batch_size
            consistency_rejected_vals = [0.0] * batch_size
            if "prompt" in inputs and "chosen" in inputs and "rejected" in inputs and hasattr(self.reward_model, "compute_consistency_reward"):
                for i in range(batch_size):
                    try:
                        consistency_chosen_vals[i] = float(self.reward_model.compute_consistency_reward(inputs["prompt"][i], inputs["chosen"][i]))
                    except Exception:
                        consistency_chosen_vals[i] = 0.0
                    try:
                        consistency_rejected_vals[i] = float(self.reward_model.compute_consistency_reward(inputs["prompt"][i], inputs["rejected"][i]))
                    except Exception:
                        consistency_rejected_vals[i] = 0.0

            consistency_chosen = torch.tensor(consistency_chosen_vals, device=device, dtype=torch.float32).clamp(-1.0, 1.0)
            consistency_rejected = torch.tensor(consistency_rejected_vals, device=device, dtype=torch.float32).clamp(-1.0, 1.0)

            # --- Combine rewards ---
            chosen_total = (chosen_rewards + self.lambda_consistency * consistency_chosen).clamp(-10.0, 10.0)
            rejected_total = (rejected_rewards + self.lambda_consistency * consistency_rejected).clamp(-10.0, 10.0)

            # --- DPO loss ---
            reward_diff = self.beta * (chosen_total - rejected_total)
            loss = -F.logsigmoid(reward_diff).mean()

            # Safety check
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Invalid loss detected: {loss}")
                loss = torch.tensor(0.0, requires_grad=True, device=device)

            return loss


def main():
    """Main training function"""
    logger.info("Starting enhanced CREAM-RAG DPO training")
    
    # Load configuration
    config_path = "config.yaml"
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded successfully")
    
    # Setup device and CUDA optimizations
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cuda":
        # Monitor GPU memory
        monitor_gpu_memory()
        
        # Calculate optimal batch size for Llama 7B
        optimal_batch = get_optimal_batch_size(7.0, config["training"].get("max_input_length", 2048))
        if optimal_batch != config["training"]["batch_size"]:
            logger.info(f"Recommended batch size: {optimal_batch}, current: {config['training']['batch_size']}")
            config["training"]["batch_size"] = optimal_batch
    
    # Load QA training data and convert to DPO format
    train_data_path = config["training"].get("train_data_path", "qa_data/combined_100000.jsonl")
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data not found at {train_data_path}")
    
    logger.info(f"Loading QA training data from {train_data_path}")
    qa_data = []
    max_samples = config["training"].get("max_samples", 10000)
    
    with open(train_data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            if line.strip():
                qa_data.append(json.loads(line))
    
    logger.info(f"Loaded {len(qa_data)} QA samples from {train_data_path}")
    
    # Convert to DPO preference pairs
    logger.info("Converting QA data to DPO preference pairs...")
    preference_pairs = create_preference_pairs_from_qa_data(qa_data, max_samples)
    logger.info(f"Created {len(preference_pairs)} preference pairs")
    
    # Load model and processor
    model_name = config["model"]["name"]
    logger.info(f"Loading multimodal model: {model_name}")
    
    # Load processor (contains tokenizer for multimodal models)
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model with optimized CUDA settings for A100
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if config["training"].get("use_bf16", True) else torch.float32,
    }
    
    # Add flash attention if enabled
    if config["training"].get("flash_attention", True):
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    logger.info("Loading multimodal model...")
    # Use AutoModel which will auto-detect the correct model class
    # This works for multimodal models as it reads the config and selects the appropriate class
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        **model_kwargs
    )
    
    # Add LoRA adapters
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["training"].get("lora_r", 16),
        lora_alpha=config["training"].get("lora_alpha", 32),
        lora_dropout=config["training"].get("lora_dropout", 0.1),
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # DISABLE gradient checkpointing - incompatible with DPO training
    # Use other memory optimizations instead
    logger.info("Gradient checkpointing DISABLED - using alternative memory optimizations")
    
    # Memory optimizations to compensate
    torch.cuda.empty_cache()  # Clear GPU cache
    
    # Enable memory efficient attention if available
    try:
        model.config.use_cache = False  # Disable KV cache for training
        logger.info("Disabled KV cache for memory efficiency")
    except:
        pass
    
    # Compile model for A100 optimization if enabled
    if config["training"].get("compile_model", True):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile for A100 optimization")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}. Continuing without compilation.")
    
    # Ensure model is in training mode and parameters require gradients
    model.train()
    
    # CRITICAL: Enable gradients for LoRA parameters
    lora_params_enabled = 0
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
            lora_params_enabled += 1
    logger.info(f"Enabled gradients for {lora_params_enabled} LoRA parameters")
    
    # Check if any parameters require gradients (without logging each one)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    grad_enabled_params = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {trainable_params}")
    logger.info(f"Parameters with gradients enabled: {grad_enabled_params}")
    
    # Verify gradients are enabled
    if grad_enabled_params == 0:
        raise RuntimeError("No parameters have gradients enabled! Training will not work.")
    
    # Get tokenizer from processor (processor wraps the tokenizer for multimodal models)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    dataset = Dataset.from_list(preference_pairs)
    
    # Tokenize dataset
    def tokenize_dataset(examples):
        return tokenize_function(examples, tokenizer)
    
    tokenized_dataset = dataset.map(tokenize_dataset, batched=True)
    train_dataset = tokenized_dataset
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    
    # Setup training arguments with CUDA optimizations for A100
    # NOTE: We save checkpoints per epoch and disable checkpoint rotation to avoid
    # FileNotFoundError issues when cleaning up old checkpoints.
    training_args = TrainingArguments(
        output_dir=config["training"]["save_path"],
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"].get("gradient_accumulation_steps", 8),
        learning_rate=2e-5,  # Optimal learning rate for CREAM-RAG
        save_strategy="epoch",
        logging_steps=config["training"].get("log_interval", 10),
        remove_unused_columns=False,
        report_to=[],  # Completely disable all reporting
        bf16=config["training"].get("use_bf16", True),  # Use bfloat16 for A100 efficiency
        fp16=config["training"].get("use_fp16", False),  # Disable fp16 to avoid conflicts
        gradient_checkpointing=False,  # DISABLED - breaks DPO gradient flow
        disable_tqdm=False,  # Keep progress bars
        save_total_limit=None,  # Disable automatic checkpoint rotation
        dataloader_pin_memory=False,  # Reduce memory usage
        dataloader_num_workers=2,  # Reduce memory usage
    )
    
    # Create data collator
    data_collator = create_dpo_data_collator(tokenizer)
    
    # Create trainer
    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        beta=0.05,  # Optimal beta for CREAM-RAG with consistency rewards
    )
    
    # Ensure output directory exists
    os.makedirs(config["training"]["save_path"], exist_ok=True)
    
    # Train
    logger.info("Starting DPO training...")
    
    # Monitor memory before training
    if device == "cuda":
        monitor_gpu_memory()
    
    trainer.train()
    
    # Monitor memory after training
    if device == "cuda":
        monitor_gpu_memory()
    
    # Save model and final checkpoint
    final_save_path = os.path.join(config["training"]["save_path"], "final_model")
    trainer.save_model(final_save_path)
    
    # Save training config for reproducibility
    config_save_path = os.path.join(config["training"]["save_path"], "training_config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Training completed successfully! Model saved to {final_save_path}")
    logger.info(f"Training config saved to {config_save_path}")

if __name__ == "__main__":
    main()


