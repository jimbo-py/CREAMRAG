
import yaml
import os
import torch
from agent.rag_retriever import FlexRAGRetriever
from agent.generator import LLMGenerator
from agent.reward_model import RewardModel
from agent.consistency import Consistency
import numpy as np
from transformers import AutoModelForCausalLM
from torch.optim import Adam
import json
import random

def load_documents(path):
    """Load documents from JSON or JSONL file"""
    if path.endswith('.jsonl'):
        documents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                # Handle your corpus format with 'id' and 'text' fields
                text = obj.get("text", "")
                if text:
                    documents.append(text)
        return documents
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                documents = []
                for item in data:
                    if isinstance(item, dict):
                        text = item.get("text", "")
                        if text:
                            documents.append(text)
                return documents
            return data

def load_questions(path):
    """Load training questions from JSON or JSONL file"""
    questions = []
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                # Try different possible keys for questions
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

def load_corpus_and_questions(path):
    """Load both documents and questions from a single JSONL file"""
    documents = []
    questions = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            
            # Check if this entry specifies its type
            entry_type = obj.get("type", "").lower()
            
            if entry_type == "document":
                documents.append(obj.get("text", ""))
            elif entry_type == "question":
                questions.append(obj.get("text", ""))
            else:
                # If no type specified, try to infer or treat as document
                if "question" in obj or "query" in obj or "prompt" in obj:
                    question = obj.get("question") or obj.get("query") or obj.get("prompt")
                    if question:
                        questions.append(question)
                else:
                    # Treat as document
                    text = obj.get("text", "")
                    if text:
                        documents.append(text)
    
    return documents, questions

def load_embeddings(path):
    """Load embeddings from numpy file"""
    np_array = np.load(path)
    tensor = torch.from_numpy(np_array)
    return tensor

def create_questions_from_all_documents(documents, max_questions=1000):
    """Create questions from every document in the corpus"""
    questions = []
    
    # Generic question templates that work with any content
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
            
        # Use different question templates for variety
        template = question_templates[i % len(question_templates)]
        
        # For documents that are questions themselves, use them directly
        if any(marker in doc.lower() for marker in ["question:", "answer:", "why", "what", "how", "when", "where"]):
            # Extract the actual question if it's formatted as a Q&A
            if "question:" in doc.lower():
                question_part = doc.split("Question:")[-1].split("Answer:")[0].strip()
                if question_part and len(question_part) < 200:
                    questions.append(question_part)
                    continue
        
        # For regular documents, use templates
        questions.append(template)
        
        # Add document-specific questions for restaurants, businesses, etc.
        if "restaurant" in doc.lower() or "pizza" in doc.lower() or "dining" in doc.lower():
            questions.append("Tell me about the restaurants mentioned.")
        elif "crime" in doc.lower() or "statistics" in doc.lower():
            questions.append("What statistics or data are provided?")
        elif "university" in doc.lower() or "application" in doc.lower():
            questions.append("What information about education is mentioned?")
    
    return questions

def main():
    config_path = r"C:\Users\lavai\Downloads\creamtestingstuff\config.yaml"
    
    print(f"Looking for config file at: {config_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("Config keys found:", list(config.keys()))
    if "device" not in config:
        raise KeyError("Missing 'device' key in config.yaml")
    
    device = torch.device(config["device"])
    print(f"Using device: {device}")
    
    # Load documents and questions
    documents = None
    questions = None
    
    # Check if there's a separate questions file in config
    if "training" in config and "questions_path" in config["training"]:
        questions_path = config["training"]["questions_path"]
        if os.path.exists(questions_path):
            questions = load_questions(questions_path)
            print(f"Loaded {len(questions)} questions from separate file")
        else:
            print(f"Questions file not found at {questions_path}, creating questions from corpus")
            max_questions = config["training"].get("max_questions_from_corpus", 1000)
            questions = create_questions_from_all_documents(documents, max_questions)
    
    # Load documents
    documents = load_documents(config["retriever"]["document_path"])
    print(f"Loaded {len(documents)} documents")
    
    # If no separate questions file, create questions from all documents
    if questions is None:
        max_questions = config["training"].get("max_questions_from_corpus", 1000)
        questions = create_questions_from_all_documents(documents, max_questions)
        print(f"Created {len(questions)} questions from corpus content (covers more documents)")
    
    print(f"Training with {len(questions)} questions")
    
    # Load embeddings
    index_embeddings = load_embeddings(config["retriever"]["embedding_path"]).to(device)
    print(f"Loaded index embeddings with shape {index_embeddings.shape}")
    
    # Initialize components
    retriever = FlexRAGRetriever(documents, index_embeddings, device)
    generator = LLMGenerator(config["generator"]["model"], device)
    reward_model = RewardModel(config["reward_model"]["model"], device)
    
    generator.tokenizer.pad_token = generator.tokenizer.eos_token
    
    # Prepare frozen reference model for consistency loss
    tokenizer = generator.tokenizer
    ref_model = AutoModelForCausalLM.from_pretrained(config["generator"]["model"]).to(device)
    ref_model.eval()
    consistency = Consistency(generator.model, ref_model, tokenizer)
    
    learning_rate = float(config["training"]["learning_rate"])
    optimizer = Adam(generator.model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(config["training"]["epochs"]):
        # Shuffle questions each epoch for variety
        shuffled_questions = questions.copy()
        random.shuffle(shuffled_questions)
        
        epoch_rewards = []
        epoch_consistency_losses = []
        
        for step, prompt in enumerate(shuffled_questions):
            # Initialize variables for both training modes
            context_parts = []
            training_mode = "retrieval"
            
            # Option: Use direct document training occasionally
            use_direct_training = config["training"].get("direct_document_training", False)
            
            if use_direct_training and step % 3 == 0:  # Every 3rd example
                # Train directly on a random document (no retrieval)
                random_doc = random.choice(documents)[:800]  # Truncate for safety
                full_prompt = f"Document: {random_doc}\n\nQuestion: {prompt}\nAnswer:"
                context_parts = ["[Direct document training]"]  # For logging
                training_mode = "direct"
            else:
                # Regular retrieval-based training
                # Add some randomness to retrieval
                top_k = config["retriever"]["top_k"]
                # Occasionally vary the number of retrieved documents
                if random.random() < 0.1:  # 10% chance
                    top_k = max(1, top_k + random.randint(-1, 1))
                
                retrieved_docs = retriever.retrieve(prompt, k=top_k)
                
                # Truncate retrieved documents to prevent context length issues
                max_context_length = 800  # Leave room for question and answer
                context_parts = []
                current_length = 0
                
                for doc in retrieved_docs:
                    # Truncate each document to reasonable length
                    doc_truncated = doc[:500] + "..." if len(doc) > 500 else doc
                    doc_tokens = len(tokenizer.encode(doc_truncated))
                    
                    if current_length + doc_tokens > max_context_length:
                        break
                        
                    context_parts.append(doc_truncated)
                    current_length += doc_tokens
                
                context = "\n---\n".join(context_parts)
                full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
            
            # Final safety check - truncate the entire prompt if still too long
            prompt_tokens = tokenizer.encode(full_prompt)
            max_input_length = 700  # Conservative limit leaving room for generation
            
            if len(prompt_tokens) > max_input_length:
                # Truncate from the middle of context, keep question intact
                question_part = f"\n\nQuestion: {prompt}\nAnswer:"
                question_tokens = len(tokenizer.encode(question_part))
                available_context_tokens = max_input_length - question_tokens - 10  # safety margin
                
                context_tokens = tokenizer.encode(f"Context:\n{context}")[:available_context_tokens]
                truncated_context = tokenizer.decode(context_tokens, skip_special_tokens=True)
                full_prompt = f"{truncated_context}...{question_part}"
            
            # Generate response with reduced max tokens to prevent overflow
            max_tokens = min(config["generator"].get("max_new_tokens", 100), 150)  # Cap at 150
            
            try:
                response = generator.generate(full_prompt, max_new_tokens=max_tokens)
            except Exception as e:
                print(f"Generation failed for prompt length {len(tokenizer.encode(full_prompt))}: {e}")
                print(f"Skipping this example...")
                continue
            
            # Compute losses
            reward = reward_model.compute_reward(prompt, response)
            consistency_loss = consistency.compute(full_prompt, response)
            
            # Total loss = -reward + lambda * consistency
            lambda_consistency = config["training"].get("lambda_consistency", 0.1)
            total_loss = -torch.tensor(reward, device=device) + lambda_consistency * consistency_loss
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(generator.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            epoch_rewards.append(reward)
            epoch_consistency_losses.append(consistency_loss.item())
            
            # Logging with better prompt length info
            log_interval = config["training"].get("log_interval", 10)
            if step % log_interval == 0:
                prompt_length = len(tokenizer.encode(full_prompt))
                print(f"[Epoch {epoch+1}/{config['training']['epochs']} Step {step+1}/{len(shuffled_questions)}]")
                print(f"  Training mode: {training_mode}")
                print(f"  Question: {prompt[:50]}...")
                print(f"  Prompt length: {prompt_length} tokens")
                if training_mode == "retrieval":
                    print(f"  Retrieved docs: {len(context_parts)}")
                print(f"  Reward: {reward:.4f}")
                print(f"  Consistency Loss: {consistency_loss.item():.4f}")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"  Response: {response[:100]}...")
                print("-" * 50)
        
        # Epoch summary
        avg_reward = np.mean(epoch_rewards)
        avg_consistency = np.mean(epoch_consistency_losses)
        print(f"\nEPOCH {epoch+1} SUMMARY:")
        print(f"  Average Reward: {avg_reward:.4f}")
        print(f"  Average Consistency Loss: {avg_consistency:.4f}")
        print(f"  Questions processed: {len(shuffled_questions)}")
        print("=" * 60)
        
        # Optional: Save model checkpoint
        if "save_path" in config["training"] and (epoch + 1) % config["training"].get("save_interval", 5) == 0:
            save_path = f"{config['training']['save_path']}_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': generator.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_reward': avg_reward,
                'avg_consistency': avg_consistency,
            }, save_path)
            print(f"Model checkpoint saved to {save_path}")

if __name__ == "__main__":
    main()