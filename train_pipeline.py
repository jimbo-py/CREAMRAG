
import yaml
import os
import torch
import torch.nn as nn
import numpy as np
import json
import random
from huggingface_hub import login
from typing import List
from dataclasses import dataclass
from agent.rag_retriever import LlamaRetriever
from agent.generator import LlamaGenerator
from agent.reward_model import RewardModel
import gc

try:
    from bitsandbytes.optim import AdamW8bit
    _USE_BNB = True
except Exception:
    from torch.optim import AdamW as TorchAdamW
    _USE_BNB = False

gc.collect()
torch.cuda.empty_cache()

# login(token="")  # You may want to set your token here

@dataclass
class PPOBatch:
    queries: List[str]
    responses: List[str]
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

class ValueNetwork(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.value_head = nn.Linear(model.config.hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        pooled_output = last_hidden_state.mean(dim=1)
        values = self.value_head(pooled_output).squeeze(-1)
        return values

def load_documents(path):
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

def load_questions(path):
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
                    question = item.get("question") or item.get("query") or obj.get("prompt") or obj.get("text")
                    if question:
                        questions.append(question)
    return questions

def load_embeddings(path):
    np_array = np.load(path)
    tensor = torch.from_numpy(np_array)
    return tensor

def create_questions_from_all_documents(documents, max_questions=1000):
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
    ]
    for i, doc in enumerate(documents):
        if len(questions) >= max_questions:
            break
        template = question_templates[i % len(question_templates)]
        if any(marker in doc.lower() for marker in ["question:", "answer:", "why", "what", "how", "when", "where"]):
            if "question:" in doc.lower():
                question_part = doc.split("Question:")[-1].split("Answer:")[0].strip()
                if question_part and len(question_part) < 200:
                    questions.append(question_part)
                    continue
        questions.append(template)
        if "restaurant" in doc.lower() or "pizza" in doc.lower() or "dining" in doc.lower():
            questions.append("Tell me about the restaurants mentioned.")
        elif "crime" in doc.lower() or "statistics" in doc.lower():
            questions.append("What statistics or data are provided?")
        elif "university" in doc.lower() or "application" in doc.lower():
            questions.append("What information about education is mentioned?")
    return questions

def main():
    config_path = r"/root/CREAMRAG/config.yaml"
    print(f"Looking for config file at: {config_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print("Config keys found:", list(config.keys()))
    if "device" not in config:
        raise KeyError("Missing 'device' key in config.yaml")
    requested_device = config["device"]
    if requested_device == "cuda" or requested_device.startswith("cuda:"):
        if torch.cuda.is_available():
            device = torch.device(requested_device)
            print(f"Using device: {device}")
        else:
            print(f"WARNING: CUDA requested ({requested_device}) but not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device(requested_device)
        print(f"Using device: {device}")

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
    print(f"Loaded {len(documents)} documents")

    questions = None
    if "training" in config and "questions_path" in config["training"]:
        questions_path = config["training"]["questions_path"]
        if os.path.exists(questions_path):
            questions = load_questions(questions_path)
    if questions is None:
        max_questions = config["training"].get("max_questions_from_corpus", 1000)
        questions = create_questions_from_all_documents(documents, max_questions)
    print(f"Training with {len(questions)} questions")

    embedding_path = config["retriever"]["embedding_path"]
    index_embeddings = load_embeddings(embedding_path).to(device)

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
    retriever.load_index_from_components(
        index_dir="index_embeddings",
        corpus_path="data/corpus.jsonl",
    )
    if hasattr(retriever, 'set_documents'):
        retriever.set_documents(documents)
    else:
        retriever.documents = documents
    if hasattr(retriever, 'set_embeddings'):
        retriever.set_embeddings(index_embeddings)
    else:
        retriever.index_embeddings = index_embeddings

    generator = LlamaGenerator(
        model_name=config["generator"]["model"],
        device=str(device),
        max_length=config["training"]["max_input_length"],
        temperature=config["generator"]["temperature"],
        use_4bit=False,
        use_8bit=False,
        use_flash_attention=False
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer
    reward_model_name = config["reward_model"]["model"]
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_model_base = AutoModelForCausalLM.from_pretrained(
        reward_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    reward_model_wrapper = RewardModel(
        model=reward_model_base,
        tokenizer=reward_tokenizer,
        device=device
    )

    class SimpleRewardModel:
        def __init__(self, reward_model, tokenizer):
            self.reward_model = reward_model
            self.tokenizer = tokenizer
            self.reward_model.model.eval()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        def compute_reward(self, prompt: str, response: str) -> float:
            try:
                enc_p = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                enc_r = self.tokenizer(response, return_tensors="pt", truncation=True, max_length=256)
                input_ids = torch.cat([enc_p["input_ids"], enc_r["input_ids"]], dim=1)
                attention = torch.cat([enc_p["attention_mask"], enc_r["attention_mask"]], dim=1)
                labels = input_ids.clone()
                labels[:, :enc_p["input_ids"].shape[1]] = -100
                input_ids = input_ids.long()
                labels = labels.long()
                attention = attention.long()
                dev = next(self.reward_model.model.parameters()).device
                input_ids = input_ids.to(dev, non_blocking=True)
                labels = labels.to(dev, non_blocking=True)
                attention = attention.to(dev, non_blocking=True)
                with torch.no_grad():
                    out = self.reward_model.model(
                        input_ids=input_ids,
                        attention_mask=attention,
                        labels=labels,
                        return_dict=True,
                    )
                    loss = out.loss
                return float(-loss.item())
            except Exception as e:
                print(f"Warning: Reward computation failed: {e}")
                return 0.0

    reward_model = SimpleRewardModel(reward_model_wrapper, reward_tokenizer)

    # Import PPOTrainer from your implementation
    from ppo_trainer import PPOTrainer

    ppo_trainer = PPOTrainer(generator, reward_model, config, device)
    batch_size = config["training"].get("batch_size", 4)

    for epoch in range(config["training"]["epochs"]):
        print(f"\n=== EPOCH {epoch+1}/{config['training']['epochs']} ===")
        shuffled_questions = questions.copy()
        random.shuffle(shuffled_questions)
        epoch_stats = {
            "actor_losses": [],
            "critic_losses": [],
            "rewards": [],
            "advantages": []
        }
        for batch_start in range(0, len(shuffled_questions), batch_size):
            batch_questions = shuffled_questions[batch_start:batch_start + batch_size]
            rag_prompts = []
            for question in batch_questions:
                retrieved_docs = retriever.retrieve(question, k=config["retriever"]["top_k"])
                context_parts = []
                for doc in retrieved_docs[:3]:
                    doc_truncated = doc[:400] if len(doc) > 400 else doc
                    context_parts.append(doc_truncated)
                context = "\n---\n".join(context_parts)
                full_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                rag_prompts.append(full_prompt)
            ppo_batch = ppo_trainer.generate_batch(rag_prompts, batch_size=len(rag_prompts))
            step_stats = ppo_trainer.train_step(ppo_batch)
            epoch_stats["actor_losses"].append(step_stats["actor_loss"])
            epoch_stats["critic_losses"].append(step_stats["critic_loss"])
            epoch_stats["rewards"].append(step_stats["mean_reward"])
            epoch_stats["advantages"].append(step_stats["mean_advantage"])
            if (batch_start // batch_size) % config["training"].get("log_interval", 10) == 0:
                print(f"  Batch {batch_start//batch_size + 1}: "
                      f"Actor Loss: {step_stats['actor_loss']:.4f}, "
                      f"Critic Loss: {step_stats['critic_loss']:.4f}, "
                      f"Mean Reward: {step_stats['mean_reward']:.4f}")
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Actor Loss: {np.mean(epoch_stats['actor_losses']):.4f}")
        print(f"  Average Critic Loss: {np.mean(epoch_stats['critic_losses']):.4f}")
        print(f"  Average Reward: {np.mean(epoch_stats['rewards']):.4f}")
        print(f"  Average Advantage: {np.mean(epoch_stats['advantages']):.4f}")
        if "save_path" in config["training"] and (epoch + 1) % config["training"].get("save_interval", 5) == 0:
            save_path = f"{config['training']['save_path']}_ppo_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.model.state_dict(),
                'value_network_state_dict': ppo_trainer.value_network.state_dict(),
                'actor_optimizer_state_dict': ppo_trainer.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': ppo_trainer.critic_optimizer.state_dict(),
                'avg_reward': np.mean(epoch_stats['rewards']),
            }, save_path)
            print(f"Checkpoint saved to {save_path}")

if __name__ == "__main__":
    main()
