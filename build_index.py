

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
CORPUS_PATH = "data/corpus.jsonl"  # Each line: {"id": ..., "text": ...}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR = "index_embeddings"

def load_corpus(path):
    corpus = []
    ids = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            corpus.append(obj['text'])
            ids.append(obj['id'])
    return corpus, ids

def save_index(index, embeddings, ids):
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss_index.bin"))
    np.save(os.path.join(INDEX_DIR, "doc_embeddings.npy"), embeddings)
    with open(os.path.join(INDEX_DIR, "doc_ids.json"), "w") as f:
        json.dump(ids, f, indent=2)

def main():
    print("📖 Loading corpus...")
    texts, ids = load_corpus(CORPUS_PATH)

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
    save_index(index, embeddings, ids)
    print(f"✅ Index built and saved to '{INDEX_DIR}'.")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()

