"""
Enhanced Training Pipeline for CREAM-RAG with PPO and Consistency Rewards
"""

import yaml
import os
import torch
import torch.nn as nn
import numpy as np
import json
import random
import logging
from huggingface_hub import login
from typing import List, Dict, Optional
from dataclasses import dataclass
from agent.rag_retriever import LlamaRetriever
from agent.generator import LlamaGenerator
from enhanced_ppo_trainer import create_enhanced_ppo_trainer, EnhancedPPOTrainer
import gc
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from bitsandbytes.optim import AdamW8bit
    _USE_BNB = True
except Exception:
    from torch.optim import AdamW as TorchAdamW
    _USE_BNB = False

gc.collect()
torch.cuda.empty_cache()

login(token="")  # Add your HuggingFace token here for Llama access

@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    epoch: int
    step: int
    policy_loss: float
    value_loss: float
    total_loss: float
    mean_reward: float
    mean_advantage: float
    approx_kl: float
    clipfrac: float
    explained_variance: float
    consistency_reward: float
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

def load_questions(path: str) -> List[str]:
    """Load questions from various formats"""
    questions = []
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                question = obj.get("question") or obj.get("query") or obj.get("prompt") or obj.get("text")
                if question:
                    questions.append(question)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    questions.append(item)
                elif isinstance(item, dict):
                    question = item.get("question") or item.get("query") or item.get("prompt") or item.get("text")
                    if question:
                        questions.append(question)
    return questions

def create_questions_from_documents(documents: List[str], max_questions: int = 1000) -> List[str]:
    """Create questions from documents using templates"""
    questions = []
    question_templates = [
        "What information is provided in this document?",
        "Summarize the key points from this content.",
        "What are the main details mentioned?",
        "Explain what this document discusses.",
        "What specific information can you extract from this?",
        "Provide an overview of this content.",
        "What are the important facts mentioned?",
        "Describe what this document contains.",
        "What are the key insights from this text?",
        "What can you learn from this document?",
    ]
    
    for i, doc in enumerate(documents):
        if len(questions) >= max_questions:
            break
            
        # Check if document already contains a question
        if any(marker in doc.lower() for marker in ["question:", "answer:", "why", "what", "how", "when", "where"]):
            if "question:" in doc.lower():
                question_part = doc.split("Question:")[-1].split("Answer:")[0].strip()
                if question_part and len(question_part) < 200:
                    questions.append(question_part)
                    continue
        
        # Use template
        template = question_templates[i % len(question_templates)]
        questions.append(template)
        
        # Add domain-specific questions
        doc_lower = doc.lower()
        if any(word in doc_lower for word in ["restaurant", "pizza", "dining", "food"]):
            questions.append("Tell me about the restaurants mentioned.")
        elif any(word in doc_lower for word in ["crime", "statistics", "data"]):
            questions.append("What statistics or data are provided?")
        elif any(word in doc_lower for word in ["university", "application", "education"]):
            questions.append("What information about education is mentioned?")
        elif any(word in doc_lower for word in ["technology", "software", "computer"]):
            questions.append("What technology-related information is provided?")
    
    return questions[:max_questions]

def create_rag_prompts(questions: List[str], retriever: LlamaRetriever, 
                      top_k: int = 5, max_context_length: int = 800) -> List[str]:
    """Create RAG prompts with retrieved context"""
    rag_prompts = []
    
    for question in tqdm(questions, desc="Creating RAG prompts"):
        try:
            # Skip empty questions
            if not question or question.strip() == "":
                logger.warning("Skipping empty question")
                rag_prompts.append("Question: What information is provided?\nAnswer:")
                continue
                
            # Retrieve relevant documents
            retrieved_docs = retriever.retrieve(question, k=top_k)
            
            # Check if we got any documents
            if not retrieved_docs:
                logger.warning(f"No documents retrieved for question: {question[:50]}...")
                rag_prompts.append(f"Question: {question}\nAnswer:")
                continue
            
            # Build context
            context_parts = []
            total_length = 0
            
            for doc in retrieved_docs:
                # Skip empty documents
                if not doc or doc.strip() == "":
                    continue
                    
                # Truncate document if too long
                doc_truncated = doc[:400] if len(doc) > 400 else doc
                
                if total_length + len(doc_truncated) > max_context_length:
                    break
                    
                context_parts.append(doc_truncated)
                total_length += len(doc_truncated)
            
            # Create context string
            if context_parts:
                context = "\n---\n".join(context_parts)
                full_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            else:
                # No context available
                full_prompt = f"Question: {question}\nAnswer:"
                
            rag_prompts.append(full_prompt)
            
        except Exception as e:
            logger.warning(f"Failed to create RAG prompt for question '{question[:50]}...': {e}")
            # Fallback to simple prompt
            rag_prompts.append(f"Question: {question}\nAnswer:")
    
    return rag_prompts

def setup_models_and_retriever(config: Dict, device: torch.device):
    """Setup all models and retriever"""
    logger.info("Setting up models and retriever...")
    
    # Setup retriever
    retriever = LlamaRetriever(
        model_name="ignored-for-st",
        device=str(device),
        max_length=config["retriever"].get("max_length", 512),
        use_4bit=False,
        use_8bit=False,
        use_flash_attention=False,
        backend="st",
        st_model_name="intfloat/e5-base-v2",
    )
    
    # Load index
    retriever.load_index_from_components(
        index_dir="index_embeddings",
        corpus_path=config["retriever"]["document_path"],
    )
    
    # Setup generator model
    generator = LlamaGenerator(
        model_name=config["generator"]["model"],
        device=str(device),
        max_length=config["training"]["max_input_length"],
        temperature=config["generator"]["temperature"],
        use_4bit=config["generator"].get("use_4bit", False),
        use_8bit=config["generator"].get("use_8bit", False),
        use_flash_attention=config["generator"].get("use_flash_attention", False)
    )
    
    logger.info("Models and retriever setup completed")
    return retriever, generator

def create_enhanced_ppo_trainer_wrapper(generator, retriever, config: Dict, device: torch.device) -> EnhancedPPOTrainer:
    """Create enhanced PPO trainer with proper configuration"""
    logger.info("Creating enhanced PPO trainer...")
    
    # Extract training configuration
    training_config = config["training"]
    
    # Create PPO trainer
    ppo_trainer = create_enhanced_ppo_trainer(
        model=generator.model,
        tokenizer=generator.tokenizer,
        retriever=retriever,
        config_dict=training_config,
        device=device
    )
    
    logger.info("Enhanced PPO trainer created successfully")
    return ppo_trainer

def train_epoch(ppo_trainer: EnhancedPPOTrainer, prompts: List[str], 
                config: Dict, epoch: int) -> List[TrainingMetrics]:
    """Train for one epoch"""
    logger.info(f"Starting epoch {epoch + 1}")
    
    batch_size = config["training"].get("batch_size", 4)
    log_interval = config["training"].get("log_interval", 10)
    save_interval = config["training"].get("save_interval", 5)
    
    # Shuffle prompts
    shuffled_prompts = prompts.copy()
    random.shuffle(shuffled_prompts)
    
    epoch_metrics = []
    step = 0
    
    # Process in batches
    for batch_start in tqdm(range(0, len(shuffled_prompts), batch_size), 
                           desc=f"Epoch {epoch + 1}"):
        batch_prompts = shuffled_prompts[batch_start:batch_start + batch_size]
        
        try:
            # Train step
            step_stats = ppo_trainer.train_step(batch_prompts, rollout_size=len(batch_prompts))
            
            # Create metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                step=step,
                policy_loss=step_stats.get('policy_loss', 0.0),
                value_loss=step_stats.get('value_loss', 0.0),
                total_loss=step_stats.get('total_loss', 0.0),
                mean_reward=step_stats.get('mean_reward', 0.0),
                mean_advantage=step_stats.get('mean_advantage', 0.0),
                approx_kl=step_stats.get('approx_kl', 0.0),
                clipfrac=step_stats.get('clipfrac', 0.0),
                explained_variance=step_stats.get('explained_variance', 0.0),
                consistency_reward=step_stats.get('consistency_reward', 0.0),
                retrieval_consistency=step_stats.get('retrieval_consistency', 0.0)
            )
            
            epoch_metrics.append(metrics)
            
            # Log progress
            if step % log_interval == 0:
                logger.info(
                    f"Epoch {epoch + 1}, Step {step}: "
                    f"Policy Loss: {metrics.policy_loss:.4f}, "
                    f"Value Loss: {metrics.value_loss:.4f}, "
                    f"Mean Reward: {metrics.mean_reward:.4f}, "
                    f"KL: {metrics.approx_kl:.6f}"
                )
            
            step += 1
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            continue
    
    return epoch_metrics

def save_checkpoint(ppo_trainer: EnhancedPPOTrainer, epoch: int, 
                   metrics: List[TrainingMetrics], config: Dict):
    """Save training checkpoint"""
    if "save_path" not in config["training"]:
        return
    
    save_path = config["training"]["save_path"]
    checkpoint_path = f"{save_path}_enhanced_epoch_{epoch + 1}.pt"
    
    # Calculate average metrics
    avg_metrics = {
        'policy_loss': np.mean([m.policy_loss for m in metrics]),
        'value_loss': np.mean([m.value_loss for m in metrics]),
        'total_loss': np.mean([m.total_loss for m in metrics]),
        'mean_reward': np.mean([m.mean_reward for m in metrics]),
        'mean_advantage': np.mean([m.mean_advantage for m in metrics]),
        'approx_kl': np.mean([m.approx_kl for m in metrics]),
        'clipfrac': np.mean([m.clipfrac for m in metrics]),
        'explained_variance': np.mean([m.explained_variance for m in metrics])
    }
    
    ppo_trainer.save_checkpoint(
        path=checkpoint_path,
        epoch=epoch,
        additional_info={
            'avg_metrics': avg_metrics,
            'config': config
        }
    )
    
    logger.info(f"Checkpoint saved to {checkpoint_path}")

def log_epoch_summary(epoch: int, metrics: List[TrainingMetrics]):
    """Log epoch summary"""
    if not metrics:
        return
    
    avg_metrics = {
        'policy_loss': np.mean([m.policy_loss for m in metrics]),
        'value_loss': np.mean([m.value_loss for m in metrics]),
        'total_loss': np.mean([m.total_loss for m in metrics]),
        'mean_reward': np.mean([m.mean_reward for m in metrics]),
        'mean_advantage': np.mean([m.mean_advantage for m in metrics]),
        'approx_kl': np.mean([m.approx_kl for m in metrics]),
        'clipfrac': np.mean([m.clipfrac for m in metrics]),
        'explained_variance': np.mean([m.explained_variance for m in metrics])
    }
    
    logger.info(f"\n=== EPOCH {epoch + 1} SUMMARY ===")
    logger.info(f"Average Policy Loss: {avg_metrics['policy_loss']:.4f}")
    logger.info(f"Average Value Loss: {avg_metrics['value_loss']:.4f}")
    logger.info(f"Average Total Loss: {avg_metrics['total_loss']:.4f}")
    logger.info(f"Average Mean Reward: {avg_metrics['mean_reward']:.4f}")
    logger.info(f"Average Mean Advantage: {avg_metrics['mean_advantage']:.4f}")
    logger.info(f"Average KL Divergence: {avg_metrics['approx_kl']:.6f}")
    logger.info(f"Average Clip Fraction: {avg_metrics['clipfrac']:.4f}")
    logger.info(f"Average Explained Variance: {avg_metrics['explained_variance']:.4f}")
    logger.info("=" * 50)

def main():
    """Main training function"""
    logger.info("Starting enhanced CREAM-RAG PPO training")
    
    # Load configuration
    config_path = "config.yaml"
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded successfully")
    
    # Setup device
    requested_device = config["device"]
    if requested_device == "cuda" or requested_device.startswith("cuda:"):
        if torch.cuda.is_available():
            device = torch.device(requested_device)
            logger.info(f"Using device: {device}")
        else:
            logger.warning(f"CUDA requested ({requested_device}) but not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
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
    
    # Load or create questions
    questions = None
    if "training" in config and "questions_path" in config["training"]:
        questions_path = config["training"]["questions_path"]
        if os.path.exists(questions_path):
            questions = load_questions(questions_path)
    
    if questions is None:
        max_questions = config["training"].get("max_questions_from_corpus", 1000)
        questions = create_questions_from_documents(documents, max_questions)
    
    logger.info(f"Training with {len(questions)} questions")
    
    # Setup models and retriever
    retriever, generator = setup_models_and_retriever(config, device)
    
    # Create RAG prompts
    rag_prompts = create_rag_prompts(
        questions=questions,
        retriever=retriever,
        top_k=config["retriever"]["top_k"],
        max_context_length=config["training"].get("context_length_limit", 800)
    )
    
    logger.info(f"Created {len(rag_prompts)} RAG prompts")
    
    # Create enhanced PPO trainer
    ppo_trainer = create_enhanced_ppo_trainer_wrapper(generator, retriever, config, device)
    
    # Training loop
    epochs = config["training"]["epochs"]
    all_metrics = []
    
    for epoch in range(epochs):
        try:
            # Train epoch
            epoch_metrics = train_epoch(ppo_trainer, rag_prompts, config, epoch)
            all_metrics.extend(epoch_metrics)
            
            # Log summary
            log_epoch_summary(epoch, epoch_metrics)
            
            # Save checkpoint
            if (epoch + 1) % config["training"].get("save_interval", 5) == 0:
                save_checkpoint(ppo_trainer, epoch, epoch_metrics, config)
            
            # Reset stats for next epoch
            ppo_trainer.reset_stats()
            
        except Exception as e:
            logger.error(f"Epoch {epoch + 1} failed: {e}")
            continue
    
    logger.info("Training completed successfully!")
    
    # Save final checkpoint
    save_checkpoint(ppo_trainer, epochs - 1, all_metrics[-len(rag_prompts):], config)
    
    # Log final statistics
    final_stats = ppo_trainer.get_stats_summary()
    logger.info("Final training statistics:")
    for key, value in final_stats.items():
        logger.info(f"  {key}: {value:.6f}")

if __name__ == "__main__":
    main()
