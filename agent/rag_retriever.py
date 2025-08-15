"""
GPU-optimized RAG retriever using either:
- Llama hidden-state mean pooling (inner product over L2-normalized vectors), or
- SentenceTransformers embeddings (cosine via IP on normalized vectors)
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False


class LlamaRetriever:
    """Retriever with two compatible backends: 'llama' (default) or 'st' (SentenceTransformers)."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_length: int = 512,
        use_4bit: bool = False,
        use_8bit: bool = True,
        use_flash_attention: bool = False,
        backend: str = "llama",                 # "llama" or "st"
        st_model_name: Optional[str] = None,    # e.g. "intfloat/e5-base-v2"
    ):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.backend = backend.lower()
        self.index = None
        self.documents: List[str] = []
        self.doc_ids: List[Any] = []

        if self.backend not in ("llama", "st"):
            raise ValueError("backend must be 'llama' or 'st'")

        logger.info(f"Initializing retriever on {self.device} | backend={self.backend}")

        if self.backend == "llama":
            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Model
            self.model = self._load_model_with_optimizations(
                model_name, use_4bit, use_8bit, use_flash_attention
            )
            self.st_model = None
        else:
            # SentenceTransformers backend
            if not _HAS_ST:
                raise RuntimeError("sentence-transformers not installed. `pip install sentence-transformers`")
            if not st_model_name:
                raise ValueError("For backend='st', you must provide st_model_name (e.g. 'intfloat/e5-base-v2').")
            self.st_model_name = st_model_name
            self.st_model = SentenceTransformer(self.st_model_name, device=str(self.device))
            self.st_model.max_seq_length = max_length
            self.tokenizer = None
            self.model = None

        logger.info("Retriever initialized successfully.")

    # ---------------------------- LLAMA BACKEND ----------------------------

    def _load_model_with_optimizations(self, model_name: str, use_4bit: bool, use_8bit: bool, use_flash_attention: bool):
        model_kwargs = {"torch_dtype": torch.float16}
        if use_4bit or use_8bit:
            try:
                from transformers import BitsAndBytesConfig
                if use_4bit:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    model_kwargs["device_map"] = "auto"
                elif use_8bit:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    model_kwargs["device_map"] = "auto"
            except ImportError:
                logger.warning("BitsAndBytes not available; loading without quantization.")

        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using FlashAttention 2")
            except ImportError:
                logger.warning("FlashAttention 2 not available; using eager attention.")
                model_kwargs["attn_implementation"] = "eager"
        else:
            model_kwargs["attn_implementation"] = "eager"

        try:
            model = AutoModel.from_pretrained(model_name, **model_kwargs)
        except Exception as e:
            logger.warning(f"Optimized load failed: {e}. Falling back to basic load.")
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)

        if "device_map" not in model_kwargs or model_kwargs["device_map"] is None:
            model = model.to(self.device)
        return model

    def _encode_texts_llama(self, texts: List[str]) -> np.ndarray:
        """Mean-pool hidden states, then return numpy array (no normalization here)."""
        embeddings = []
        bs = 4
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            inputs = None
            try:
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    hidden_states = outputs.last_hidden_state  # [B, T, H]
                    attn = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
                    masked = hidden_states * attn
                    pooled = masked.sum(dim=1) / attn.sum(dim=1)  # [B, H]
                    embeddings.append(pooled.cpu().numpy())
            finally:
                # free VRAM ASAP
                del inputs
                torch.cuda.empty_cache()
        return np.vstack(embeddings)

    # ---------------------- SENTENCE-TRANSFORMERS BACKEND ----------------------

    def _encode_texts_st(self, texts: List[str]) -> np.ndarray:
        """SentenceTransformers encode with normalized embeddings (cosine)."""
        embs = self.st_model.encode(
            texts,
            batch_size=256,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # critical: we will use IndexFlatIP
        )
        return embs.astype("float32")

    # ----------------------------- COMMON UTILS -----------------------------

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if self.backend == "llama":
            return self._encode_texts_llama(texts)
        else:
            return self._encode_texts_st(texts)

    def load_documents(self, document_path: str) -> List[str]:
        docs: List[str] = []
        with open(document_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                if isinstance(data, dict) and "text" in data:
                    docs.append(data["text"])
                elif isinstance(data, str):
                    docs.append(data)
        logger.info(f"Loaded {len(docs)} documents")
        self.documents = docs
        # If your JSONL has 'id', keep them; otherwise enumerate:
        try:
            with open(document_path, "r", encoding="utf-8") as f:
                ids = []
                for line in f:
                    o = json.loads(line)
                    ids.append(o.get("id"))
                if any(i is None for i in ids):
                    ids = list(range(len(docs)))
        except Exception:
            ids = list(range(len(docs)))
        self.doc_ids = ids
        return docs

    # ------------------------------ INDEX I/O -------------------------------

    def build_index(self, documents: List[str], save_path: Optional[str] = None):
        """Build FAISS index from documents for the **current backend**."""
        logger.info(f"Building index ({self.backend}) for {len(documents)} docs...")
        embs = self.encode_texts(documents)

        if self.backend == "llama":
            # Use IP over normalized vectors (cosine)
            faiss.normalize_L2(embs)
            self.index = faiss.IndexFlatIP(embs.shape[1])
        else:
            # ST path already normalized in _encode_texts_st
            self.index = faiss.IndexFlatIP(embs.shape[1])

        self.index.add(embs.astype("float32"))
        self.documents = documents
        if not self.doc_ids or len(self.doc_ids) != len(documents):
            self.doc_ids = list(range(len(documents)))

        logger.info(f"Index built: {len(documents)} vectors")
        if save_path:
            self.save_index(save_path)

    def save_index(self, base_path: str):
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        faiss.write_index(self.index, f"{base_path}.faiss")
        metadata = {
            "doc_ids": self.doc_ids,
            "documents": self.documents,       # stored so we can return texts directly
            "backend": self.backend,
            "model_name": self.model_name if self.backend == "llama" else self.st_model_name,
            "max_length": self.max_length,
        }
        with open(f"{base_path}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Index saved to {base_path}.faiss (+ metadata)")

    def load_index(self, base_path: str):
        """Load index saved by `save_index` (works for both backends)."""
        self.index = faiss.read_index(f"{base_path}.faiss")
        with open(f"{base_path}_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Safety: warn if backend/model mismatch
        idx_backend = meta.get("backend", self.backend)
        if idx_backend != self.backend:
            logger.warning(f"Index backend={idx_backend} differs from current backend={self.backend}. "
                           f"Set backend='{idx_backend}' or rebuild to match.")
        self.doc_ids = meta.get("doc_ids", [])
        self.documents = meta.get("documents", [])
        logger.info(f"Index loaded: {len(self.documents)} docs")

    def load_index_from_components(self, index_dir: str, corpus_path: str):
        """
        Load FAISS + ids built by an external script that saved:
          - faiss_index.bin
          - doc_ids.json
          - (optionally) doc_embeddings.npy
        Rebuilds `documents` from corpus_path by aligning ids -> text.
        Intended for SentenceTransformers builder scripts.
        """
        faiss_path = os.path.join(index_dir, "faiss_index.bin")
        ids_path = os.path.join(index_dir, "doc_ids.json")
        if not (os.path.exists(faiss_path) and os.path.exists(ids_path) and os.path.exists(corpus_path)):
            raise FileNotFoundError("Missing faiss_index.bin, doc_ids.json, or corpus file")

        self.index = faiss.read_index(faiss_path)
        with open(ids_path, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)

        # Reconstruct text by id
        id2text: Dict[Any, str] = {}
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                id2text[o["id"]] = o["text"]
        self.documents = [id2text[i] for i in self.doc_ids]
        logger.info(f"Loaded external index with {len(self.documents)} docs")

    # ---------------------------- RETRIEVAL APIS ----------------------------

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        if self.index is None:
            raise ValueError("Index not built/loaded. Call build_index() or load_index().")

        if self.backend == "llama":
            q = self._encode_texts_llama([query]).astype("float32")
            faiss.normalize_L2(q)  # match IP on normalized
        else:
            q = self._encode_texts_st([query])  # already normalized float32

        D, I = self.index.search(q, k)
        return [self.documents[i] for i in I[0] if i < len(self.documents)]

    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("Index not built/loaded. Call build_index() or load_index().")

        if self.backend == "llama":
            q = self._encode_texts_llama([query]).astype("float32")
            faiss.normalize_L2(q)
        else:
            q = self._encode_texts_st([query])

        scores, indices = self.index.search(q, k)
        out = []
        for s, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                out.append({"document": self.documents[idx], "score": float(s), "doc_id": self.doc_ids[idx]})
        return out
