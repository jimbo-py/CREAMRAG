#!/usr/bin/env python3
"""
CUDA Optimization Script for A100 80GB Training
Provides utilities for optimizing CUDA memory usage and performance.
"""

import torch
import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def setup_cuda_optimizations():
    """Setup CUDA optimizations for A100 80GB"""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, skipping optimizations")
        return
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Enable TF32 for A100 (faster with minimal precision loss)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cudnn benchmarking for better performance
    torch.backends.cudnn.benchmark = True
    
    # Set memory fraction if needed
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
    
    if gpu_memory >= 80:  # A100 80GB
        logger.info("Detected A100 80GB - using optimized settings")
        # Set memory fraction to 95% to leave some buffer
        torch.cuda.set_per_process_memory_fraction(0.95)
    else:
        logger.warning(f"GPU memory ({gpu_memory:.1f}GB) may be insufficient")
        # Set memory fraction to 90% for smaller GPUs
        torch.cuda.set_per_process_memory_fraction(0.90)
    
    logger.info("CUDA optimizations applied successfully")

def get_optimal_batch_size(model_size_gb: float, sequence_length: int = 2048) -> int:
    """Calculate optimal batch size based on model size and GPU memory"""
    if not torch.cuda.is_available():
        return 1
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    # Rough estimation: each sample needs ~2x model size in memory
    # Plus some overhead for gradients, optimizer states, etc.
    estimated_memory_per_sample = model_size_gb * 2.5
    
    # Account for sequence length
    sequence_factor = sequence_length / 2048
    estimated_memory_per_sample *= sequence_factor
    
    # Calculate safe batch size (leave 20% buffer)
    safe_memory = gpu_memory * 0.8
    optimal_batch_size = int(safe_memory / estimated_memory_per_sample)
    
    # Ensure minimum batch size
    optimal_batch_size = max(1, optimal_batch_size)
    
    logger.info(f"Optimal batch size: {optimal_batch_size} (GPU: {gpu_memory:.1f}GB, Model: {model_size_gb:.1f}GB)")
    return optimal_batch_size

def monitor_gpu_memory():
    """Monitor GPU memory usage"""
    if not torch.cuda.is_available():
        return
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cleared")

def get_model_memory_usage(model) -> float:
    """Calculate model memory usage in GB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**3
    return size_all_mb

def optimize_model_for_training(model, use_gradient_checkpointing: bool = True, use_compile: bool = True):
    """Optimize model for training on A100"""
    # Enable gradient checkpointing for memory efficiency
    if use_gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing")
    
    # Compile model for A100 optimization
    if use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
    
    return model

if __name__ == "__main__":
    # Test the optimizations
    setup_cuda_optimizations()
    monitor_gpu_memory()
    
    # Test batch size calculation for Llama 7B
    optimal_batch = get_optimal_batch_size(7.0)  # Llama 7B
    print(f"Optimal batch size for Llama 7B: {optimal_batch}")
