"""
CREAM Trainer with CREAM methodology + RAG support - FIXED VERSION
"""
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from scipy.stats import kendalltau
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class CREAMConfig:
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    num_responses: int = 4
    temperature: float = 0.8
    top_p: float = 0.9
    consistency_method: str = "consistency_avg"
    dpo_beta: float = 0.1
    learning_rate: float = 1e-6
    epochs: int = 1
    batch_size: int = 4

class CREAMTrainer:
    def __init__(self, config: CREAMConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.iteration = 0

    def load_model(self, checkpoint: Optional[str] = None):
        if checkpoint and os.path.exists(checkpoint):
            print(f"Loading from {checkpoint}")
            model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float32)
        else:
            model = AutoModelForCausalLM.from_pretrained(self.config.model_name, torch_dtype=torch.float32)
        return model.to(self.device)

    def sft_training(self, sft_data: List[Dict], output_dir: str):
        model = self.load_model()
        
        # Convert data to the format expected by the tokenizer
        formatted_data = []
        for item in sft_data:
            # Combine prompt and response into a single text
            combined_text = f"{item['prompt']} {item['response']}"
            formatted_data.append({"text": combined_text})
        
        dataset = Dataset.from_list(formatted_data)
        
        def tokenize_function(examples):
            # Tokenize the text with padding and truncation
            result = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",  # Pad to max_length to ensure consistent sizes
                max_length=self.config.max_length,
                return_tensors=None,  # Keep as lists for now
            )
            # For causal language modeling, labels are the same as input_ids
            # But we need to handle padding tokens properly
            labels = []
            for input_ids in result["input_ids"]:
                # Create labels, setting -100 for padding tokens (ignored in loss)
                label_ids = []
                for token_id in input_ids:
                    if token_id == self.tokenizer.pad_token_id:
                        label_ids.append(-100)  # Ignore padding tokens in loss
                    else:
                        label_ids.append(token_id)
                labels.append(label_ids)
            
            result["labels"] = labels
            return result
        
        # Apply tokenization and remove original text column
        tokenized = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=["text"]
        )
        
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            save_steps=500,
            save_total_limit=1,
            remove_unused_columns=False,
            prediction_loss_only=True,
            dataloader_pin_memory=False,  # Disable pin_memory to avoid warnings
            dataloader_num_workers=0,  # Use single process to avoid multiprocessing issues
        )
        
        from transformers import DataCollatorForLanguageModeling
        
        # Create a data collator specifically for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling, not masked LM
        )
        
        trainer = Trainer(
            model=model, 
            args=args, 
            train_dataset=tokenized, 
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        trainer.train()
        trainer.save_model()
        return output_dir

    def response_sampling(self, prompts: List[str], model_checkpoint: str,
                          output_file: str, retriever=None):
        model = self.load_model(model_checkpoint)
        model.eval()
        results = []
        
        for q in prompts:
            if retriever:
                ctxs = retriever.retrieve(q)
                full = f"Context:\n" + "\n".join(ctxs) + f"\n\nQuestion: {q}"
            else:
                full = q
                
            inp = self.tokenizer(
                full, 
                return_tensors='pt', 
                truncation=True,
                max_length=self.config.max_length // 2
            ).to(self.device)
            
            responses = []
            for _ in range(self.config.num_responses):
                with torch.no_grad():
                    out = model.generate(
                        inp.input_ids, 
                        max_new_tokens=100,
                        do_sample=True, 
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                text = self.tokenizer.decode(
                    out[0, inp.input_ids.shape[1]:], 
                    skip_special_tokens=True
                )
                responses.append(text.strip())
            results.append({'prompt': full, 'responses': responses})
            
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f: 
            json.dump(results, f, indent=2)
        return output_file

    def _score_response_with_model(self, model, prompt, response):
        txt = f"Human: {prompt}\nAssistant: {response}\n\nRate this response quality (1-10):"
        inp = self.tokenizer(
            txt, 
            return_tensors='pt', 
            truncation=True,
            max_length=self.config.max_length
        ).to(self.device)
        
        with torch.no_grad():
            logits = model(**inp).logits[0, -1]
            # Try to get scores for numbers 1-10
            ids = [self.tokenizer.convert_tokens_to_ids(str(i)) for i in range(1, 11)]
            ids = [i for i in ids if i is not None and i < logits.size(0)]
            if ids:
                probs = F.softmax(logits[ids], dim=0)
                weights = torch.arange(1, len(ids) + 1, dtype=torch.float32, device=self.device)
                return (probs * weights).sum().item()
        
        # Fallback scoring method
        full = prompt + response
        inp = self.tokenizer(full, return_tensors='pt', truncation=True).to(self.device)
        with torch.no_grad():
            out = model(**inp)
            lp = F.log_softmax(out.logits, dim=-1)
            return lp.mean().item()

    def response_ranking(self, sampling_file, model_checkpoint, is_reference, output_suffix):
        model = self.load_model(model_checkpoint)
        model.eval()
        
        with open(sampling_file, 'r') as f:
            data = json.load(f)
            
        ranked = []
        for item in data:
            scores = [
                self._score_response_with_model(model, item['prompt'], r) 
                for r in item['responses']
            ]
            idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            ranked.append({
                'prompt': item['prompt'],
                'responses': item['responses'],
                'scores': scores,
                'ranking': idxs
            })
            
        output_file = f"{os.path.splitext(sampling_file)[0]}.rewarding.{'reference' if is_reference else 'current'}.json"
        with open(output_file, 'w') as f:
            json.dump(ranked, f, indent=2)
        return output_file

    def calculate_consistency(self, curr_file, ref_file, method, output_file=None):
        with open(curr_file, 'r') as f:
            curr = json.load(f)
        with open(ref_file, 'r') as f:
            ref = json.load(f)
            
        assert len(curr) == len(ref)
        dpo_data = []
        scores = []
        
        for c, r in zip(curr, ref):
            tau, _ = kendalltau(c['ranking'], r['ranking'])
            sc = 0.0 if np.isnan(tau) else tau
            scores.append(sc)
            
            idx_best = np.argmax(c['scores'])
            idx_worst = np.argmin(c['scores'])
            
            if idx_best != idx_worst:
                dpo_data.append({
                    'prompt': c['prompt'],
                    'chosen': c['responses'][idx_best],
                    'rejected': c['responses'][idx_worst],
                    'consistency_score': sc
                })
                
        avg = np.mean(scores)
        
        if output_file is None:
            base = os.path.splitext(curr_file)[0]
            output_file = f"{base}.dpo.json"
            
        with open(output_file, 'w') as f:
            json.dump(dpo_data, f, indent=2)
            
        print(f"Avg consistency: {avg:.4f}")
        return output_file, avg

    def consistency_regularized_training(self, dpo_file, model_checkpoint, output_dir):
        model = self.load_model(model_checkpoint)
        
        with open(dpo_file, 'r') as f:
            data = json.load(f)
        
        # Check if we have data for DPO training
        if not data:
            print("No DPO data available, skipping consistency regularized training")
            return model_checkpoint
            
        # Ensure the data format is correct for DPOTrainer
        formatted_data = []
        for item in data:
            formatted_item = {
                'prompt': item['prompt'],
                'chosen': item['chosen'], 
                'rejected': item['rejected']
            }
            # Remove any extra fields that might confuse the trainer
            formatted_data.append(formatted_item)
            
        dataset = Dataset.from_list(formatted_data)
        
        cfg = DPOConfig(
            output_dir=output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            beta=self.config.dpo_beta,
            remove_unused_columns=False,
            bf16=False,  # Disable bf16 for CPU/compatibility
            fp16=False,  # Disable fp16 for CPU
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            logging_steps=10,
        )
        
        trainer = DPOTrainer(
            model=model,
            args=cfg,
            train_dataset=dataset,
            processing_class=self.tokenizer  # Use processing_class instead of tokenizer
        )
        trainer.train()
        trainer.save_model()
        return output_dir

    def full_cream_iteration(self, prompts: List[str], sft_data: Optional[List[Dict]] = None,
                             current_checkpoint: Optional[str] = None,
                             reference_checkpoint: Optional[str] = None,
                             retriever=None):
        it = self.iteration + 1
        base = f"outputs/iteration_{it}"
        os.makedirs(base, exist_ok=True)
        
        # Initial SFT training if this is the first iteration
        if it == 1 and sft_data:
            current_checkpoint = self.sft_training(sft_data, f"{base}/sft")
        
        # Response sampling
        samp_file = f"{base}/sampling.json"
        self.response_sampling(prompts, current_checkpoint, samp_file, retriever)
        
        # Ranking with current and reference models
        cur_rank = self.response_ranking(samp_file, current_checkpoint, False, "current")
        ref_rank = self.response_ranking(samp_file, reference_checkpoint, True, "reference")
        
        # Calculate consistency and prepare DPO data
        dpo_file, consist = self.calculate_consistency(cur_rank, ref_rank, self.config.consistency_method)
        
        # Consistency regularized training
        new_ckpt = self.consistency_regularized_training(dpo_file, current_checkpoint, f"{base}/dpo")
        
        self.iteration += 1
        return new_ckpt, consist
