from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMGenerator:
    def __init__(self, model_name="gpt2", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def generate(self, prompt, max_new_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


