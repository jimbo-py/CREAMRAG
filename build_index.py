import ujson as json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import os
import numpy as np

# ---------------- Config ----------------
DATA_FILE = "./qa_data/combined_100000.jsonl"  # your combined dataset
INDEX_FILE = "./qa_data/faiss_index.bin"
EMBEDDINGS_FILE = "./qa_data/embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small & fast
BATCH_SIZE = 512

# ---------------- Load dataset ----------------
examples = []
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        # use context + question as the text for retrieval embedding
        text = (ex["context"] + " " + ex["question"]).strip()
        examples.append(text)

print(f"Loaded {len(examples)} examples")

# ---------------- Create embeddings ----------------
model = SentenceTransformer(MODEL_NAME)
all_embeddings = []

for i in tqdm(range(0, len(examples), BATCH_SIZE), desc="Embedding"):
    batch = examples[i:i+BATCH_SIZE]
    emb = model.encode(batch, show_progress_bar=False)
    all_embeddings.append(emb)

all_embeddings = np.vstack(all_embeddings).astype("float32")
print(f"Embeddings shape: {all_embeddings.shape}")

# Save embeddings for later
np.save(EMBEDDINGS_FILE, all_embeddings)
print(f"Saved embeddings to {EMBEDDINGS_FILE}")

# ---------------- Build FAISS index ----------------
dim = all_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # inner product similarity
faiss.normalize_L2(all_embeddings)  # normalize if using IP
index.add(all_embeddings)

print(f"Index has {index.ntotal} vectors")

# Save index
faiss.write_index(index, INDEX_FILE)
print(f"Saved FAISS index to {INDEX_FILE}")
