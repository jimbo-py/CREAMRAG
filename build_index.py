


import os
import json
import numpy as np
import faiss
import argparse
from sentence_transformers import SentenceTransformer
from pathlib import Path

# === CONFIGURATION ===
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_corpus_from_jsonl(path, max_docs=None):
    """Load corpus from JSONL file with text field"""
    corpus = []
    ids = []
    doc_count = 0
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if max_docs and doc_count >= max_docs:
                break
                
            obj = json.loads(line)
            # Try different possible text fields
            text = ""
            if 'text' in obj and obj['text'].strip():
                text = obj['text']
            elif 'context' in obj and obj['context'].strip():
                text = obj['context']
            elif 'document' in obj and obj['document'].strip():
                text = obj['document']
            elif 'passage' in obj and obj['passage'].strip():
                text = obj['passage']
            else:
                # Combine question and answer if no context available
                question = obj.get('question', '')
                answer = obj.get('answer', '')
                if question.strip() or answer.strip():  # At least one should be non-empty
                    text = f"Question: {question} Answer: {answer}"
            
            if text.strip():  # Only add non-empty texts
                corpus.append(text)
                ids.append(obj.get('id', f"doc_{len(ids)}"))
                doc_count += 1
                
    return corpus, ids

def load_combined_datasets(data_dir, max_docs_per_dataset=None, total_max_docs=None):
    """Load and combine all datasets from the data directory"""
    corpus = []
    ids = []
    total_docs = 0
    
    # List of dataset directories to process (ordered by size for efficiency)
    dataset_dirs = [
        "crag",           # Smallest first
        "ragbench", 
        "squad",
        "hotpot_qa",
        "trivia_qa",      # Large
        "natural_questions"  # Largest
    ]
    
    for dataset_name in dataset_dirs:
        dataset_path = os.path.join(data_dir, dataset_name)
        jsonl_path = os.path.join(dataset_path, f"{dataset_name}.jsonl")
        
        if os.path.exists(jsonl_path):
            print(f"📖 Loading {dataset_name} dataset...")
            
            # Calculate how many docs we can still add
            remaining_docs = None
            if total_max_docs:
                remaining_docs = total_max_docs - total_docs
                if remaining_docs <= 0:
                    print(f"   Skipping {dataset_name} - reached total limit")
                    break
            
            # Use the smaller of max_docs_per_dataset or remaining_docs
            docs_to_load = max_docs_per_dataset
            if remaining_docs and (docs_to_load is None or remaining_docs < docs_to_load):
                docs_to_load = remaining_docs
            
            texts, doc_ids = load_corpus_from_jsonl(jsonl_path, docs_to_load)
            
            # Add dataset prefix to IDs to avoid conflicts
            prefixed_ids = [f"{dataset_name}_{doc_id}" for doc_id in doc_ids]
            
            corpus.extend(texts)
            ids.extend(prefixed_ids)
            total_docs += len(texts)
            
            print(f"   Loaded {len(texts)} documents from {dataset_name} (Total: {total_docs})")
            
            if total_max_docs and total_docs >= total_max_docs:
                print(f"   Reached total document limit of {total_max_docs}")
                break
        else:
            print(f"⚠️  Skipping {dataset_name} - file not found: {jsonl_path}")
    
    return corpus, ids

def save_index(index, embeddings, ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    np.save(os.path.join(output_dir, "doc_embeddings.npy"), embeddings)
    with open(os.path.join(output_dir, "doc_ids.json"), "w") as f:
        json.dump(ids, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from combined datasets")
    parser.add_argument("--data_dir", type=str, default="data", 
                       help="Directory containing dataset folders")
    parser.add_argument("--corpus", type=str, default=None,
                       help="Path to single corpus file (optional)")
    parser.add_argument("--output_dir", type=str, default="index_embeddings",
                       help="Output directory for index")
    parser.add_argument("--max_docs_per_dataset", type=int, default=10000,
                       help="Maximum documents per dataset (default: 10000)")
    parser.add_argument("--total_max_docs", type=int, default=50000,
                       help="Total maximum documents across all datasets (default: 50000)")
    
    args = parser.parse_args()
    
    if args.corpus:
        # Load single corpus file
        print(f"📖 Loading corpus from {args.corpus}...")
        texts, ids = load_corpus_from_jsonl(args.corpus, args.total_max_docs)
    else:
        # Load combined datasets
        print("📖 Loading combined datasets...")
        texts, ids = load_combined_datasets(args.data_dir, args.max_docs_per_dataset, args.total_max_docs)
    
    print(f"📊 Total documents loaded: {len(texts)}")
    
    if len(texts) == 0:
        print("❌ No documents found! Please check your data directory.")
        return

    print(f"🧠 Embedding {len(texts)} documents with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Process embeddings in batches to avoid memory issues
    batch_size = 1000  # Smaller batch size for memory efficiency
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        print(f"   Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch_texts)} documents)")
        batch_embeddings = model.encode(batch_texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    embeddings = np.vstack(embeddings)

    print("🔍 Building FAISS index...")
    dim = embeddings.shape[1]
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    index.add(embeddings)

    print("💾 Saving index and metadata...")
    save_index(index, embeddings, ids, args.output_dir)
    print(f"✅ Index built and saved to '{args.output_dir}'.")
    print(f"📈 Index contains {index.ntotal} documents with {dim}-dimensional embeddings")

if __name__ == "__main__":
    main()





