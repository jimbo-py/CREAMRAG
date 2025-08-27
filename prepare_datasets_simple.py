#!/usr/bin/env python3
"""
Simplified Dataset Preparation Script for DPO Training

This script downloads and prepares smaller samples of the required datasets:
- HotpotQA (validation split)
- Natural Questions (validation split)
- SQuAD (validation split)
- TriviaQA (already processed)
- CRAG (already processed)
- RAGBench (already processed)

Usage:
    python prepare_datasets_simple.py
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# HuggingFace token for accessing restricted datasets
HF_TOKEN = None  # Set to None to use public datasets only

def create_directory(path: str):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """Save data to JSONL file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def prepare_hotpot_qa_simple(output_dir: str, max_samples: int = 1000):
    """Prepare HotpotQA dataset using validation split"""
    logger.info("Preparing HotpotQA dataset (validation split)...")
    
    try:
        # Load validation split instead of train to avoid memory issues
        dataset = load_dataset("hotpot_qa", "distractor", split="validation", token=HF_TOKEN)
        
        # Process samples
        processed_data = []
        total_samples = min(max_samples, len(dataset))
        
        for i, sample in enumerate(tqdm(dataset, desc="Processing HotpotQA", total=total_samples)):
            if i >= total_samples:
                break
                
            # Extract question and answer
            question = sample['question']
            answer = sample['answer']
            
            # Create context from supporting facts
            context_parts = []
            for fact in sample['supporting_facts']:
                try:
                    if len(fact) == 2:
                        title, sent_id = fact
                    elif len(fact) >= 2:
                        title, sent_id = fact[0], fact[1]
                    else:
                        continue
                        
                    # Find the corresponding paragraph
                    for para in sample['context']:
                        if para[0] == title:
                            if sent_id < len(para[1]):
                                context_parts.append(para[1][sent_id])
                            break
                except (ValueError, IndexError, TypeError):
                    continue
            
            context = " ".join(context_parts)
            
            processed_data.append({
                "id": f"hotpot_{i}",
                "question": question,
                "answer": answer,
                "context": context,
                "type": "hotpot_qa"
            })
        
        # Save to file
        output_file = os.path.join(output_dir, "hotpot_qa.jsonl")
        save_jsonl(processed_data, output_file)
        logger.info(f"Saved {len(processed_data)} HotpotQA samples to {output_file}")
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error preparing HotpotQA: {e}")
        return []

def prepare_natural_questions_simple(output_dir: str, max_samples: int = 1000):
    """Prepare Natural Questions dataset using validation split"""
    logger.info("Preparing Natural Questions dataset (validation split)...")
    
    try:
        # Load validation split instead of train
        dataset = load_dataset("natural_questions", split="validation", token=HF_TOKEN)
        
        # Process samples
        processed_data = []
        total_samples = min(max_samples, len(dataset))
        
        for i, sample in enumerate(tqdm(dataset, desc="Processing Natural Questions", total=total_samples)):
            if i >= total_samples:
                break
                
            # Extract question and answer
            question = sample['question']['text']
            
            # Get answer
            answer = ""
            if 'annotations' in sample and 'short_answers' in sample['annotations']:
                if sample['annotations']['short_answers']:
                    answer = sample['annotations']['short_answers'][0]['text']
                else:
                    answer = "No short answer available"
            elif 'answer' in sample:
                answer = sample['answer']
            else:
                answer = "No answer available"
            
            # Get context from document
            context = ""
            if 'document' in sample and 'tokens' in sample['document']:
                if 'token' in sample['document']['tokens']:
                    context = " ".join(sample['document']['tokens']['token'])
            elif 'context' in sample:
                context = sample['context']
            
            processed_data.append({
                "id": f"nq_{i}",
                "question": question,
                "answer": answer,
                "context": context,
                "type": "natural_questions"
            })
        
        # Save to file
        output_file = os.path.join(output_dir, "natural_questions.jsonl")
        save_jsonl(processed_data, output_file)
        logger.info(f"Saved {len(processed_data)} Natural Questions samples to {output_file}")
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error preparing Natural Questions: {e}")
        return []

def prepare_squad_simple(output_dir: str, max_samples: int = 1000):
    """Prepare SQuAD dataset using validation split"""
    logger.info("Preparing SQuAD dataset (validation split)...")
    
    try:
        # Load validation split
        dataset = load_dataset("squad", split="validation", token=HF_TOKEN)
        
        # Process samples
        processed_data = []
        total_samples = min(max_samples, len(dataset))
        
        for i, sample in enumerate(tqdm(dataset, desc="Processing SQuAD", total=total_samples)):
            if i >= total_samples:
                break
                
            # Extract question and answer
            question = sample['question']
            answer = sample['answers']['text'][0] if sample['answers']['text'] else "No answer"
            context = sample['context']
            
            processed_data.append({
                "id": f"squad_{i}",
                "question": question,
                "answer": answer,
                "context": context,
                "type": "squad"
            })
        
        # Save to file
        output_file = os.path.join(output_dir, "squad.jsonl")
        save_jsonl(processed_data, output_file)
        logger.info(f"Saved {len(processed_data)} SQuAD samples to {output_file}")
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error preparing SQuAD: {e}")
        return []

def create_combined_dataset(datasets: List[Dict[str, Any]], output_dir: str):
    """Create a combined dataset from all individual datasets"""
    logger.info("Creating combined dataset...")
    
    combined_data = []
    for dataset_name, data in datasets:
        for item in data:
            combined_data.append(item)
    
    # Save combined dataset
    output_file = os.path.join(output_dir, "combined.jsonl")
    save_jsonl(combined_data, output_file)
    logger.info(f"Saved {len(combined_data)} combined samples to {output_file}")
    
    return combined_data

def main():
    """Main function to prepare all datasets"""
    parser = argparse.ArgumentParser(description="Prepare datasets for DPO training (simplified)")
    parser.add_argument("--output-dir", type=str, default="data", 
                       help="Output directory for datasets")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="Maximum samples per dataset")
    parser.add_argument("--datasets", type=str, nargs="+", 
                       default=["hotpot_qa", "natural_questions", "squad"],
                       help="Datasets to prepare")
    
    args = parser.parse_args()
    
    # Create output directory
    create_directory(args.output_dir)
    
    # Prepare datasets
    all_datasets = []
    
    if "hotpot_qa" in args.datasets:
        hotpot_dir = os.path.join(args.output_dir, "hotpot_qa")
        create_directory(hotpot_dir)
        hotpot_data = prepare_hotpot_qa_simple(hotpot_dir, args.max_samples)
        if hotpot_data:
            all_datasets.append(("hotpot_qa", hotpot_data))
    
    if "natural_questions" in args.datasets:
        nq_dir = os.path.join(args.output_dir, "natural_questions")
        create_directory(nq_dir)
        nq_data = prepare_natural_questions_simple(nq_dir, args.max_samples)
        if nq_data:
            all_datasets.append(("natural_questions", nq_data))
    
    if "squad" in args.datasets:
        squad_dir = os.path.join(args.output_dir, "squad")
        create_directory(squad_dir)
        squad_data = prepare_squad_simple(squad_dir, args.max_samples)
        if squad_data:
            all_datasets.append(("squad", squad_data))
    
    # Create combined dataset
    if all_datasets:
        combined_dir = os.path.join(args.output_dir, "combined")
        create_directory(combined_dir)
        combined_data = create_combined_dataset(all_datasets, combined_dir)
    
    # Print summary
    logger.info("=== Dataset Preparation Summary ===")
    total_samples = sum(len(data) for _, data in all_datasets)
    logger.info(f"Total samples prepared: {total_samples}")
    
    for dataset_name, data in all_datasets:
        logger.info(f"{dataset_name}: {len(data)} samples")
    
    logger.info("Dataset preparation completed successfully!")

if __name__ == "__main__":
    main()
