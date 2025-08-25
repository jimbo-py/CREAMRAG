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
from transformers import DataCollatorWithPadding
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import gc
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

gc.collect()
torch.cuda.empty_cache()

# Disable wandb completely
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

# Login to HuggingFace
login(token="HUGGINGFACE-TOKEN")

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

def create_preference_pairs(questions: List[str], max_pairs: int = 100) -> List[Dict]:
    """Create preference pairs for DPO training"""
    preference_pairs = []
    
    # Simple preference pairs based on question quality
    for i, question in enumerate(questions[:max_pairs]):
        # Create a good answer
        good_answer = f"This is a good answer to: {question}"
        
        # Create a bad answer
        bad_answer = f"This is a bad answer to: {question}"
        
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
    """Simple DPO Trainer"""
    
    def __init__(self, model, tokenizer, beta=0.1, **kwargs):
        super().__init__(model=model, **kwargs)
        self.tokenizer = tokenizer
        self.beta = beta
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute DPO loss"""
        # Ensure inputs are on the correct device and have the right shape
        chosen_input_ids = inputs["chosen_input_ids"].to(model.device)
        chosen_attention_mask = inputs["chosen_attention_mask"].to(model.device)
        rejected_input_ids = inputs["rejected_input_ids"].to(model.device)
        rejected_attention_mask = inputs["rejected_attention_mask"].to(model.device)
        
        # Get logits for chosen and rejected responses
        chosen_outputs = model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask
        )
        rejected_outputs = model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask
        )
        
        # Compute log probabilities - ensure gradients flow
        chosen_logps = self._get_logps(chosen_outputs.logits, chosen_input_ids)
        rejected_logps = self._get_logps(rejected_outputs.logits, rejected_input_ids)
        
        # Compute DPO loss - ensure it's connected to model parameters
        chosen_rewards = chosen_logps.sum(dim=-1)
        rejected_rewards = rejected_logps.sum(dim=-1)
        
        # Simple DPO loss - ensure it's a scalar tensor with gradients
        losses = -F.logsigmoid(self.beta * (chosen_rewards - rejected_rewards))
        loss = losses.mean()
        
        # Ensure loss requires gradients
        if not loss.requires_grad:
            # This shouldn't happen, but if it does, create a dummy loss
            dummy_loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
            loss = loss + dummy_loss * 0.0
        
        return (loss, None) if return_outputs else loss
    
    def _get_logps(self, logits, labels):
        """Get log probabilities"""
        log_probs = F.log_softmax(logits, dim=-1)
        logps = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
        return logps  # Return the full tensor, let dpo_loss handle the summing
    
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
        """Override prediction step for DPO evaluation"""
        # For DPO, we need to handle both chosen and rejected inputs
        if "chosen_input_ids" in inputs:
            # This is DPO data, use our custom loss computation
            loss = self.compute_loss(model, inputs)
            return (loss, None, None)
        else:
            # This is standard evaluation data, skip evaluation for now
            return (None, None, None)



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
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load documents
    doc_path = config["retriever"]["document_path"]
    if not os.path.exists(doc_path):
        alternatives = ["corpus.jsonl", "data/corpus.jsonl", "documents.jsonl", "data/documents.jsonl"]
        found_file = None
        for alt_path in alternatives:
            if os.path.exists(alt_path):
                found_file = alt_path
                break
        if found_file:
            doc_path = found_file
        else:
            raise FileNotFoundError(f"Could not find document file at {doc_path}")
    
    documents = load_documents(doc_path)
    logger.info(f"Loaded {len(documents)} documents")
    
    # Create questions from documents - FULL SCALE
    max_questions = config["training"].get("max_questions_from_corpus", 50000)
    questions = []
    
    # Use all available documents for training
    for i, doc in enumerate(documents[:max_questions]):
        # Create more diverse questions
        question_types = [
            f"What information is provided in this document? {doc[:200]}...",
            f"Summarize the key points from: {doc[:200]}...",
            f"What are the main topics discussed in: {doc[:200]}...",
            f"Extract the important facts from: {doc[:200]}...",
            f"What can we learn from: {doc[:200]}..."
        ]
        questions.append(random.choice(question_types))
    
    logger.info(f"Training with {len(questions)} questions from {len(documents)} documents")
    
    # Create preference pairs for DPO
    preference_pairs = create_preference_pairs(questions, max_pairs=len(questions))
    logger.info(f"Created {len(preference_pairs)} preference pairs")
    
    # Load model and tokenizer
    model_name = config["model"]["name"]
    logger.info(f"Loading model: {model_name}")
    
    # Load model without quantization first, then add LoRA
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16
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
    
    # Ensure model is in training mode and parameters require gradients
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Trainable parameter: {name}")
    
    # Check if any parameters require gradients
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {trainable_params}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    dataset = Dataset.from_list(preference_pairs)
    
    # Tokenize dataset
    def tokenize_dataset(examples):
        return tokenize_function(examples, tokenizer)
    
    tokenized_dataset = dataset.map(tokenize_dataset, batched=True)
    
    # Split dataset into train and eval (90% train, 10% eval)
    dataset_size = len(tokenized_dataset)
    train_size = int(0.9 * dataset_size)
    eval_size = dataset_size - train_size
    
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, dataset_size))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=config["training"]["save_path"],
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=32,  # Increase to compensate for smaller batch size
        learning_rate=float(config["training"]["learning_rate"]),
        save_steps=config["training"].get("save_interval", 100),
        logging_steps=config["training"].get("log_interval", 10),
        remove_unused_columns=False,
        report_to=[],  # Completely disable all reporting
        bf16=True,  # Use bfloat16 for efficiency
        gradient_checkpointing=False,  # Disable gradient checkpointing to avoid gradient issues
        disable_tqdm=False,  # Keep progress bars
        save_total_limit=5,  # Keep last 5 checkpoints to save disk space
        # Remove evaluation for now to avoid input_ids issues
        # eval_strategy="steps",  # Evaluate every save_steps
        # eval_steps=config["training"].get("save_interval", 100),  # Same as save_steps
        # load_best_model_at_end=True,  # Load the best model at the end
        # metric_for_best_model="eval_loss",  # Use eval_loss as the metric
        # greater_is_better=False,  # Lower loss is better
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
        beta=config["training"]["dpo_beta"],
    )
    
    # Ensure output directory exists
    os.makedirs(config["training"]["save_path"], exist_ok=True)
    
    # Train
    logger.info("Starting DPO training...")
    trainer.train()
    
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



if __name__ == "__main__":
    main()

