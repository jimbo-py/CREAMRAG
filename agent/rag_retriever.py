from transformers import AutoModel, AutoTokenizer
import torch

class FlexRAGRetriever:
    def __init__(self, documents, index_embeddings, device="cpu"):
        self.device = device
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.documents = documents
        self.index_embeddings = index_embeddings

    def encode(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        return embeddings

    def retrieve(self, question, k=5):
        q_emb = self.encode(question)  # shape (1, 384)
        scores = torch.matmul(q_emb, self.index_embeddings.T)  # (1, 384) x (384, N)
        topk = torch.topk(scores, k=k, dim=-1)
        return [self.documents[i] for i in topk.indices[0].tolist()]
