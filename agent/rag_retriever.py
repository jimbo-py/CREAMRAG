"""
GPU-optimized RAG retriever using Llama 8B for embeddings
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import faiss
import json
import os
from typing import List, Optional, Dict, Any
import logging
from huggingface_hub import login

login(token="")

logger = logging.getLogger(__name__)

class LlamaRetriever:
    """Retriever using open-source embedding model with GPU optimizations"""

    def __init__(self, 
                 model_name: str,
                 device: str = "cuda",
                 max_length: int = 512,
                 use_4bit: bool = False,
                 use_8bit: bool = True,
                 use_flash_attention: bool = False):

        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        logger.info(f"Initializing Llama retriever on {self.device}")
        logger.info(f"Model: {model_name}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        self.model = self._load_model_with_optimizations(
            model_name, use_4bit, use_8bit, use_flash_attention
        )

        # Index & document storage
        self.index = None
        self.documents = []
        self.doc_ids = []

        logger.info("Llama retriever initialized successfully")

    def _load_model_with_optimizations(self, model_name: str, use_4bit: bool, 
                                       use_8bit: bool, use_flash_attention: bool):
        """Load model with GPU optimizations"""
        
        # Start with basic model loading parameters
        model_kwargs = {
            "torch_dtype": torch.float16,
        }
        
        # Only add quantization if BitsAndBytesConfig is available
        quantization_config = None
        if use_4bit or use_8bit:
            try:
                from transformers import BitsAndBytesConfig
                
                if use_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                elif use_8bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    model_kwargs["device_map"] = "auto"
                    
            except ImportError:
                logger.warning("BitsAndBytesConfig not available, using standard loading")
        
        # Only set flash attention if explicitly requested and available
        if use_flash_attention:
            try:
                # Test if flash_attn is available
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using FlashAttention 2")
            except ImportError:
                logger.warning("FlashAttention 2 not available, using default attention")
                model_kwargs["attn_implementation"] = "eager"
        else:
            model_kwargs["attn_implementation"] = "eager"
        
        try:
            model = AutoModel.from_pretrained(model_name, **model_kwargs)
        except Exception as e:
            logger.warning(f"Failed to load with optimizations: {e}")
            logger.info("Falling back to basic model loading...")
            # Fallback to basic loading
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
        
        # Move to device if not using device_map
        if "device_map" not in model_kwargs or model_kwargs["device_map"] is None:
            model = model.to(self.device)
        
        return model
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts using Llama embeddings"""
        embeddings = []
        
        batch_size = 4
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                attention_mask = inputs['attention_mask'].unsqueeze(-1)
                masked_hidden = hidden_states * attention_mask
                embeddings_batch = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
                embeddings.append(embeddings_batch.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def load_documents(self, document_path: str) -> List[str]:
        """Load documents from JSONL file"""
        documents = []
        with open(document_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if 'text' in data:
                        documents.append(data['text'])
                    elif isinstance(data, str):
                        documents.append(data)
        logger.info(f"Loaded {len(documents)} documents")
        self.documents = documents
        return documents
    
    def build_index(self, documents: List[str], save_path: Optional[str] = None):
        """Build FAISS index from documents"""
        logger.info(f"Building index for {len(documents)} documents...")
        
        embeddings = self.encode_texts(documents)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype('float32'))
        
        self.documents = documents
        self.doc_ids = list(range(len(documents)))
        
        logger.info(f"Index built successfully with {len(documents)} documents")
        
        if save_path:
            self.save_index(save_path)
    
    def save_index(self, base_path: str):
        """Save index and documents"""
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        
        faiss.write_index(self.index, f"{base_path}.faiss")
        
        metadata = {
            "documents": self.documents,
            "doc_ids": self.doc_ids,
            "model_name": self.model_name,
            "max_length": self.max_length
        }
        
        with open(f"{base_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Index saved to {base_path}")
    
    def load_index(self, base_path: str):
        """Load saved index"""
        self.index = faiss.read_index(f"{base_path}.faiss")
        
        with open(f"{base_path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.documents = metadata["documents"]
        self.doc_ids = metadata["doc_ids"]
        
        logger.info(f"Index loaded with {len(self.documents)} documents")
    
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve top-k documents for query"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_embedding = self.encode_texts([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        retrieved_docs = []
        for idx in indices[0]:
            if idx < len(self.documents):
                retrieved_docs.append(self.documents[idx])
        
        return retrieved_docs
    
    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents with similarity scores"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        query_embedding = self.encode_texts([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "score": float(score),
                    "doc_id": self.doc_ids[idx]
                })
        
        return results
