"""
Main script for RAG + CREAM training pipeline
"""
import torch
import json
import os
from datasets import load_dataset
from cream_trainer import CREAMTrainer, CREAMConfig
from sentence_transformers import SentenceTransformer, util

class SimpleRetriever:
    """
    Simple in-memory retriever using Sentence Transformers and cosine similarity
    """
    def __init__(self, documents):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.docs = documents
        # Precompute embeddings
        self.embeddings = self.model.encode(self.docs, convert_to_tensor=True)

    def retrieve(self, query, k=3):
        # Encode query
        query_emb = self.model.encode(query, convert_to_tensor=True)
        # Compute cosine similarities
        scores = util.pytorch_cos_sim(query_emb, self.embeddings)[0]
        topk = torch.topk(scores, k=k).indices
        return [self.docs[i] for i in topk]


def prepare_sft_data(dataset_name="imdb", num_samples=50):
    """Prepare SFT data from a dataset"""
    dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
    sft_data = []
    for item in dataset:
        prompt = f"Analyze this text and provide a thoughtful response: {item['text'][:200]}..."
        resp = 'positive' if item['label']==1 else 'negative'
        response = f"This text appears {resp} in sentiment, discussing key themes."
        sft_data.append({"prompt": prompt, "response": response})
    return sft_data


def prepare_prompts(dataset_name="imdb", num_samples=20):
    """Prepare prompts for RAG queries"""
    dataset = load_dataset(dataset_name, split=f"test[:{num_samples}]")
    return [f"What can you tell me about this text: {item['text'][:150]}..." for item in dataset]


def main():
    print("Starting RAG + CREAM Training Pipeline")

    # Configuration
    config = CREAMConfig(
        model_name="microsoft/DialoGPT-medium",
        max_length=512,
        num_responses=4,
        temperature=0.8,
        top_p=0.9,
        consistency_method="consistency_avg",
        dpo_beta=0.1,
        learning_rate=1e-6,
        epochs=1,
        batch_size=2
    )

    print(f"Model: {config.model_name} | Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Initialize components
    cream_trainer = CREAMTrainer(config)

    # Prepare retrieval corpus (replace with your KB)
    retrieval_docs = [
        "Artificial intelligence enables machines to perform tasks requiring human intelligence.",
        "Machine learning improves performance as more data is available.",
        "Neural networks are inspired by the human brain.",
        "Deep learning models learn hierarchical representations.",
        "Sentiment analysis detects emotional tones in text."
    ]
    retriever = SimpleRetriever(retrieval_docs)

    # Prepare SFT data and prompts
    sft_data = prepare_sft_data("imdb", num_samples=20)
    prompts = prepare_prompts("imdb", num_samples=10)

    current_ckpt, reference_ckpt = None, None
    history = []
    num_iters = 2

    for it in range(num_iters):
        print(f"\n===== Iteration {it+1}/{num_iters} =====")
        current_ckpt, consistency = cream_trainer.full_cream_iteration(
            prompts=prompts,
            sft_data=sft_data if it==0 else None,
            current_checkpoint=current_ckpt,
            reference_checkpoint=reference_ckpt,
            retriever=retriever
        )
        history.append(consistency)
        reference_ckpt = current_ckpt
        print(f"Iteration {it+1} consistency: {consistency:.4f}")

    # Save summary
    summary = {"consistency_history": history, "final_checkpoint": current_ckpt}
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/cream_training_summary.json","w") as f:
        json.dump(summary, f, indent=2)
    print("Training completed.")

if __name__ == '__main__':
    main()
