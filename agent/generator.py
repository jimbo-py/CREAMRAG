"""
GPU-optimized text generator using Llama 8B
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Any, Optional
import logging
from huggingface_hub import login

login(token = "")

logger = logging.getLogger(__name__)

class LlamaGenerator:
    """GPU-optimized generator using configurable model"""

    def __init__(self, 
                 model_name: str,
                 device: str = "cuda",
                 max_length: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.1,
                 use_4bit: bool = False,  # Changed default to False to avoid issues
                 use_8bit: bool = False,  # Changed default to False to avoid memory issues  
                 use_flash_attention: bool = False):  # Changed default to False
        
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        
        logger.info(f"Initializing generator on {self.device}")
        logger.info(f"Model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations
        self.model = self._load_model_with_optimizations(
            model_name, use_4bit, use_8bit, use_flash_attention
        )
        
        logger.info("Generator initialized successfully")

    
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
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except Exception as e:
            logger.warning(f"Failed to load with optimizations: {e}")
            logger.info("Falling back to basic model loading...")
            # Fallback to basic loading
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        
        # Move to device if not using device_map
        if "device_map" not in model_kwargs or model_kwargs["device_map"] is None:
            model = model.to(self.device)
        
        return model
    
    def generate(self, prompt: str, max_new_tokens: int = 150, 
                temperature: Optional[float] = None, 
                top_p: Optional[float] = None) -> str:
        """Generate text from prompt"""
        
        # Use instance defaults if not specified
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            model_for_generate = self.model.module if hasattr(self.model, 'module') else self.model
            outputs = model_for_generate.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def generate_batch(self, prompts: List[str], max_new_tokens: int = 150,
                      temperature: Optional[float] = None,
                      top_p: Optional[float] = None) -> List[str]:
        """Generate text for multiple prompts"""
        
        # Use instance defaults if not specified
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            model_for_generate = self.model.module if hasattr(self.model, 'module') else self.model
            outputs = model_for_generate.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode responses
        generated_texts = []
        for i, output in enumerate(outputs):
            input_length = inputs['input_ids'][i].shape[0]
            generated_text = self.tokenizer.decode(
                output[input_length:], 
                skip_special_tokens=True
            )
            generated_texts.append(generated_text.strip())
        
        return generated_texts
    
    def generate_with_context(self, context: str, question: str, 
                            max_new_tokens: int = 150) -> str:
        """Generate answer given context and question"""
        
        # Create prompt
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        
        return self.generate(prompt, max_new_tokens)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "vocab_size": self.tokenizer.vocab_size,
            "model_parameters": sum(p.numel() for p in self.model.parameters())
        }


