import torch.nn.functional as F
from transformers import AutoTokenizer
import torch

class Consistency:
    def __init__(self, model, ref_model, tokenizer):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer

    def compute(self, prompt, response):
        text = prompt + response
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            ref_logits = self.ref_model(**inputs).logits
        curr_logits = self.model(**inputs).logits
        p_ref = F.softmax(ref_logits, dim=-1)
        p_curr = F.softmax(curr_logits, dim=-1)
        return F.kl_div(p_curr.log(), p_ref, reduction="batchmean")
