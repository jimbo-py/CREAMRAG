

import os
import json
import numpy as np
import faiss
import argparse
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_corpus(path):
    corpus = []
    ids = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            corpus.append(obj['text'])
            ids.append(obj['id'])
    return corpus, ids

def save_index(index, embeddings, ids, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    np.save(os.path.join(output_dir, "doc_embeddings.npy"), embeddings)
    with open(os.path.join(output_dir, "doc_ids.json"), "w") as f:
        json.dump(ids, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from corpus")
    parser.add_argument("--corpus", type=str, default="data/corpus.jsonl", 
                       help="Path to corpus file")
    parser.add_argument("--output_dir", type=str, default="index_embeddings",
                       help="Output directory for index")
    
    args = parser.parse_args()
    
    print("📖 Loading corpus...")
    texts, ids = load_corpus(args.corpus)

    print(f"🧠 Embedding {len(texts)} documents with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    print("🔍 Building FAISS index...")
    dim = embeddings.shape[1]
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    index.add(embeddings)

    print("💾 Saving index and metadata...")
    save_index(index, embeddings, ids, args.output_dir)
    print(f"✅ Index built and saved to '{args.output_dir}'.")

if __name__ == "__main__":
    main()



