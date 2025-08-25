"""
Consistency calculation utilities aligned with CREAM implementation.
Provides functions for calculating consistency between reward rankings.
"""

from scipy.stats import spearmanr, rankdata, kendalltau
import statistics
import numpy as np
import os
from typing import List, Dict, Optional, Literal
from enum import Enum

class ConsistencyMethod(Enum):
    SPEARMAN = "spearman"
    KENDALL = "kendall" 
    TOPORDER = "toporder"

def calc_consistency(score1: List[float], score2: List[float], method: ConsistencyMethod) -> float:
    """
    Calculate consistency between two score lists using specified method.
    
    Args:
        score1: First list of scores
        score2: Second list of scores  
        method: Consistency method (spearman, kendall, toporder)
        
    Returns:
        Consistency score in [0, 1] range
    """
    if len(score1) != len(score2) or \
       any(s is None for s in score1) or any(s is None for s in score2):
        return 0.0
    
    rank1 = rankdata(score1, method='ordinal')
    rank2 = rankdata(score2, method='ordinal')
    
    if method == ConsistencyMethod.KENDALL:
        consistency, _ = kendalltau(rank1, rank2)
        # Scale from [-1, 1] to [0, 1]
        consistency = (consistency + 1) / 2
    elif method == ConsistencyMethod.SPEARMAN:
        consistency, _ = spearmanr(rank1, rank2)
        consistency = (consistency + 1) / 2
    elif method == ConsistencyMethod.TOPORDER:
        if rank1[0] == rank2[0] and rank1[-1] == rank2[-1]:
            consistency = 1.0
        else:
            consistency = 0.0
    else:
        raise ValueError(f"Method {method} not supported")
    
    return consistency

def score_candidates_ll(model, tokenizer, prompt: str, candidates: List[str], 
                       max_length: int = 2048) -> List[float]:
    """
    Score candidates using log-likelihood.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        candidates: List of candidate responses
        max_length: Maximum sequence length
        
    Returns:
        List of log-likelihood scores
    """
    import torch
    
    scores = []
    model.eval()
    
    with torch.no_grad():
        for candidate in candidates:
            # Format full sequence
            full_text = f"{prompt}{candidate}"
            
            # Tokenize
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, 
                             max_length=max_length, padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Get logits
            outputs = model(**inputs, return_dict=True)
            logits = outputs.logits[:, :-1, :]  # Remove last token
            labels = inputs['input_ids'][:, 1:]  # Remove first token
            
            # Calculate log-likelihood
            log_probs = torch.gather(logits.log_softmax(-1), dim=2, 
                                   index=labels.unsqueeze(2)).squeeze(2)
            
            # Mask padding
            mask = (labels != tokenizer.pad_token_id).float()
            score = (log_probs * mask).sum(-1).mean().item()
            scores.append(score)
    
    return scores




