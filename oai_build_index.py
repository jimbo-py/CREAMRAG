import os
import json
import numpy as np
import faiss
from openai import OpenAI  # Import the OpenAI client
# === CONFIGURATION ===
CORPUS_PATH = "data/corpus.jsonl"  # Each line: {"id": ..., "text": ...}
EMBEDDING_MODEL = "text-embedding-3-small"  # Using a recommended OpenAI embedding model
INDEX_DIR = "index_embeddings"
# Set your OpenAI API key
# Make sure to replace "YOUR_OPENAI_API_KEY" with your actual API key
# It's recommended to load this from an environment variable for security
# E.g., OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# You'll need to set the environment variable beforehand.
OPENAI_API_KEY = ""
def load_corpus(path):
    corpus = []
    ids = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            corpus.append(obj['text'])
            ids.append(obj['id'])
    return corpus, ids
def get_openai_embeddings(texts, model_name, api_key):
    """ 
    Generates embeddings for a list of texts using the OpenAI API.
    """
    client = OpenAI(api_key=api_key)
    embeddings = []
    # OpenAI recommends batching requests for efficiency
    # You might need to adjust the batch size based on your needs and API limits
    batch_size = 500
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch_texts,
            model=model_name
        )
        # Extract embeddings and convert to a list of numpy arrays
        embeddings.extend([np.array(record.embedding) for record in response.data])  #
    return np.array(embeddings)
def save_index(index, embeddings, ids):
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "faiss_index.bin"))
    np.save(os.path.join(INDEX_DIR, "doc_embeddings.npy"), embeddings)
    with open(os.path.join(INDEX_DIR, "doc_ids.json"), "w") as f:
        json.dump(ids, f, indent=2)
def main():
    print(":book: Loading corpus...")
    texts, ids = load_corpus(CORPUS_PATH)
    print(f":brain: Embedding {len(texts)} documents with OpenAI {EMBEDDING_MODEL}...")
    embeddings = get_openai_embeddings(texts, EMBEDDING_MODEL, OPENAI_API_KEY)  #
    print(":mag: Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # Using L2 distance, which is common for OpenAI embeddings
    index.add(embeddings)
    print(":floppy_disk: Saving index and metadata...")
    save_index(index, embeddings, ids)
    print(f":white_check_mark: Index built and saved to '{INDEX_DIR}'.")
if __name__ == "__main__":
    main()


