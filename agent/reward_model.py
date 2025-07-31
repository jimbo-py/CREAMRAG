from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class RewardModel:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english", device="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.device = device

    def compute_reward(self, prompt, response):
        text = prompt + " " + response
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        return probs[0][1].item()
