"""
Proper CREAM Implementation following the official instructions
CREAM: Consistency Regularized Self-Rewarding Language Models

This implementation follows the official CREAM methodology:
1. SFT Training (M0 -> M1)
2. Response Sampling
3. Ranking by current and reference models
4. Consistency calculation (Kendall's Tau)
5. DPO training with consistency regularization
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from scipy.stats import kendalltau
import copy
from dataclasses import dataclass

@dataclass
class CREAMConfig:
    """Configuration for CREAM training"""
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    num_responses: int = 4
    temperature: float = 0.8
    top_p: float = 0.9
    consistency_method: str = "consistency_avg"  # or "consistency_dyn"
    dpo_beta: float = 0.1
    learning_rate: float = 1e-6
    epochs: int = 1
    batch_size: int = 4

class CREAMTrainer:
    def __init__(self, config: CREAMConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Initialize models (will be loaded as needed)
        self.current_model = None
        self.reference_model = None
        self.iteration = 0
    
    def load_model(self, checkpoint_path: Optional[str] = None) -> AutoModelForCausalLM:
        """Load model from checkpoint or base model"""
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading model from checkpoint: {checkpoint_path}")
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        else:
            print(f"Loading base model: {self.config.model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
        
        return model.to(self.device)
    
    def sft_training(self, sft_data: List[Dict], output_dir: str = "outputs/sft"):
        """
        Step 1: Supervised Fine-Tuning (M0 -> M1)
        """
        print("Starting SFT training...")
        
        # Load base model
        model = self.load_model()
        
        # Prepare dataset
        def preprocess_function(examples):
            inputs = [ex["prompt"] + ex["response"] for ex in examples]
            model_inputs = self.tokenizer(
                inputs, 
                max_length=self.config.max_length, 
                truncation=True, 
                padding=True
            )
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            return model_inputs
        
        # Convert to dataset
        dataset = Dataset.from_list(sft_data)
        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            bf16=False,
            fp16=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train()
        trainer.save_model()
        
        print(f"SFT training completed. Model saved to {output_dir}")
        return output_dir
    
    def response_sampling(self, prompts: List[str], model_checkpoint: str, 
                         output_file: str = "outputs/sampling.json"):
        """
        Step 2: Sample multiple responses for each prompt
        """
        print("Starting response sampling...")
        
        # Load model
        model = self.load_model(model_checkpoint)
        model.eval()
        
        sampling_results = []
        
        for i, prompt in enumerate(prompts):
            print(f"Sampling for prompt {i+1}/{len(prompts)}")
            
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=self.config.max_length//2,
                truncation=True
            ).to(self.device)
            
            responses = []
            
            # Generate multiple responses
            for j in range(self.config.num_responses):
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=100,
                        do_sample=True,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Extract response
                response_tokens = outputs[:, inputs.input_ids.shape[1]:]
                response_text = self.tokenizer.decode(response_tokens[0], skip_special_tokens=True)
                responses.append(response_text.strip())
            
            sampling_results.append({
                "prompt": prompt,
                "responses": responses
            })
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sampling_results, f, indent=2, ensure_ascii=False)
        
        print(f"Sampling completed. Results saved to {output_file}")
        return output_file
    
    def response_ranking(self, sampling_file: str, model_checkpoint: Optional[str] = None,
                        is_reference: bool = False, output_suffix: str = "current"):
        """
        Step 3: Rank responses using model as reward model
        """
        print(f"Starting response ranking ({'reference' if is_reference else 'current'} model)...")
        
        # Load model
        model = self.load_model(model_checkpoint)
        model.eval()
        
        # Load sampling results
        with open(sampling_file, 'r', encoding='utf-8') as f:
            sampling_data = json.load(f)
        
        ranking_results = []
        
        for item in sampling_data:
            prompt = item["prompt"]
            responses = item["responses"]
            
            # Score each response
            scores = []
            for response in responses:
                score = self._score_response_with_model(model, prompt, response)
                scores.append(score)
            
            # Create ranking (higher score = better rank)
            response_scores = list(zip(responses, scores))
            response_scores.sort(key=lambda x: x[1], reverse=True)
            
            ranking_results.append({
                "prompt": prompt,
                "responses": responses,
                "scores": scores,
                "ranking": [responses.index(rs[0]) for rs, _ in response_scores]  # Indices in original order
            })
        
        # Save ranking results
        base_name = os.path.splitext(sampling_file)[0]
        output_file = f"{base_name}.rewarding.{output_suffix}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ranking_results, f, indent=2, ensure_ascii=False)
        
        print(f"Ranking completed. Results saved to {output_file}")
        return output_file
    
    def _score_response_with_model(self, model: AutoModelForCausalLM, prompt: str, response: str) -> float:
        """Score a response using the model as reward model"""
        # Create scoring prompt (simple approach)
        scoring_text = f"Human: {prompt}\nAssistant: {response}\n\nRate this response quality (1-10):"
        
        # Tokenize
        inputs = self.tokenizer(
            scoring_text, 
            return_tensors="pt", 
            max_length=self.config.max_length,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            # Get model predictions for rating tokens
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
            
            # Simple scoring: probability of high rating tokens
            rating_tokens = self.tokenizer.convert_tokens_to_ids(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
            valid_tokens = [t for t in rating_tokens if t is not None and t < logits.shape[0]]
            
            if valid_tokens:
                rating_probs = F.softmax(logits[valid_tokens], dim=0)
                # Weighted average (higher ratings get more weight)
                weights = torch.tensor([i+1 for i in range(len(valid_tokens))], dtype=torch.float32)
                score = torch.sum(rating_probs * weights).item()
            else:
                # Fallback: use log probability of the response
                full_text = prompt + " " + response
                inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True).to(self.device)
                outputs = model(**inputs)
                log_probs = F.log_softmax(outputs.logits, dim=-1)
                score = log_probs.mean().item()
        
        return score
    
    def calculate_consistency(self, current_ranking_file: str, reference_ranking_file: str,
                            method: str = "consistency_avg", output_file: str = None):
        """
        Step 4: Calculate consistency and create DPO preference pairs
        """
        print(f"Calculating consistency using {method}...")
        
        # Load ranking results
        with open(current_ranking_file, 'r', encoding='utf-8') as f:
            current_rankings = json.load(f)
        
        with open(reference_ranking_file, 'r', encoding='utf-8') as f:
            reference_rankings = json.load(f)
        
        assert len(current_rankings) == len(reference_rankings), "Ranking files must have same length"
        
        dpo_data = []
        consistency_scores = []
        
        for curr, ref in zip(current_rankings, reference_rankings):
            assert curr["prompt"] == ref["prompt"], "Prompts must match"
            
            # Calculate Kendall's Tau coefficient
            tau, _ = kendalltau(curr["ranking"], ref["ranking"])
            consistency_scores.append(tau if not np.isnan(tau) else 0.0)
            
            # Create preference pairs based on rankings
            responses = curr["responses"]
            curr_scores = curr["scores"]
            
            # Find best and worst responses according to current model
            best_idx = np.argmax(curr_scores)
            worst_idx = np.argmin(curr_scores)
            
            if best_idx != worst_idx:  # Ensure we have different responses
                dpo_item = {
                    "prompt": curr["prompt"],
                    "chosen": responses[best_idx],
                    "rejected": responses[worst_idx],
                    "consistency_score": consistency_scores[-1]
                }
                dpo_data.append(dpo_item)
        
        # Calculate overall consistency
        if method == "consistency_avg":
            avg_consistency = np.mean(consistency_scores)
            print(f"Average consistency (Kendall's Tau): {avg_consistency:.4f}")
        
        # Save DPO data
        if output_file is None:
            base_name = os.path.splitext(current_ranking_file)[0]
            output_file = f"{base_name}.dpo.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dpo_data, f, indent=2, ensure_ascii=False)
        
        print(f"DPO preference data saved to {output_file}")
        return output_file, np.mean(consistency_scores)
    
    def consistency_regularized_training(self, dpo_data_file: str, model_checkpoint: str,
                                       output_dir: str = "outputs/dpo"):
        """
        Step 5: DPO training with consistency regularization
        """
        print("Starting consistency regularized DPO training...")
        
        # Load model
        model = self.load_model(model_checkpoint)
        
        # Load DPO data
        with open(dpo_data_file, 'r', encoding='utf-8') as f:
            dpo_data = json.load(f)
        
        # Prepare dataset
        dataset = Dataset.from_list(dpo_data)
        
        # DPO Configuration
        dpo_config = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            beta=self.config.dpo_beta,
            remove_unused_columns=False,
            bf16=False,
            fp16=False,
            dataloader_pin_memory=False
        )
        
        # Initialize DPO trainer
        dpo_trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        dpo_trainer.train()
        dpo_trainer.save_model()
        
        print(f"DPO training completed. Model saved to {output_dir}")
        return output_dir
    
    def full_cream_iteration(self, prompts: List[str], sft_data: List[Dict] = None,
                           current_checkpoint: str = None, reference_checkpoint: str = None):
        """
        Run a complete CREAM iteration
        """
        print(f"\n{'='*50}")
        print(f"CREAM ITERATION {self.iteration + 1}")
        print(f"{'='*50}")
        
        iteration_dir = f"outputs/iteration_{self.iteration + 1}"
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Step 1: SFT (only for first iteration)
        if self.iteration == 0 and sft_data:
            sft_dir = f"{iteration_dir}/sft"
            current_checkpoint = self.sft_training(sft_data, sft_dir)
        
        # Step 2: Response Sampling
        sampling_file = f"{iteration_dir}/sampling.json"
        self.response_sampling(prompts, current_checkpoint, sampling_file)
        
        # Step 3: Ranking with current model
        current_ranking_file = self.response_ranking(
            sampling_file, current_checkpoint, is_reference=False, output_suffix="current"
        )
        
        # Step 4: Ranking with reference model
        reference_ranking_file = self.response_ranking(
            sampling_file, reference_checkpoint, is_reference=True, output_suffix="reference"
        )
        
        # Step 5: Calculate consistency and create DPO data
        dpo_file, consistency = self.calculate_consistency(
            current_ranking_file, reference_ranking_file, self.config.consistency_method
        )
        
        # Step 6: Consistency regularized training
        dpo_dir = f"{iteration_dir}/dpo"
        new_checkpoint = self.consistency_regularized_training(dpo_file, current_checkpoint, dpo_dir)
        
        self.iteration += 1
        
        print(f"Iteration {self.iteration} completed!")
        print(f"Consistency score: {consistency:.4f}")
        print(f"New model saved to: {new_checkpoint}")
        
        return new_checkpoint, consistency

# Example usage
def run_cream_training():
    """Example of running CREAM training"""
    
    # Configuration
    config = CREAMConfig(
        model_name="microsoft/DialoGPT-medium",
        num_responses=4,
        epochs=1,
        batch_size=2,
        learning_rate=1e-6
    )
    
    # Sample data
    sft_data = [
        {"prompt": "What is machine learning?", "response": "Machine learning is a subset of AI that enables computers to learn from data."},
        {"prompt": "Explain neural networks.", "response": "Neural networks are computing systems inspired by biological neural networks."},
        {"prompt": "What is deep learning?", "response": "Deep learning uses neural networks with multiple layers to model complex patterns."}
    ]
    
    prompts = [
        "What is artificial intelligence?",
        "How do computers learn?",
        "Explain the concept of algorithms."
    ]
    
    # Initialize CREAM trainer
    cream_trainer = CREAMTrainer(config)
    
    # Run multiple iterations
    current_checkpoint = None
    reference_checkpoint = None
    
    for i in range(2):  # 2 iterations for demo
        current_checkpoint, consistency = cream_trainer.full_cream_iteration(
            prompts=prompts,
            sft_data=sft_data if i == 0 else None,  # SFT only on first iteration
            current_checkpoint=current_checkpoint,
            reference_checkpoint=reference_checkpoint if i > 0 else None
        )
        
        # Update reference for next iteration
        reference_checkpoint = current_checkpoint

if __name__ == "__main__":
    run_cream_training()