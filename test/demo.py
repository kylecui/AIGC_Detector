import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

# 加载模型
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

@torch.no_grad()
def compute_perplexity_entropy(text):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    outputs = model(input_ids, labels=input_ids)
    logits = outputs.logits

    probs = torch.nn.functional.softmax(logits, dim=-1)
    target_ids = input_ids[:, 1:]
    probs = probs[:, :-1, :]

    token_probs = probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    log_probs = torch.log(token_probs + 1e-12)
    perplexity = torch.exp(-log_probs.mean())

    entropy_per_token = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    avg_entropy = entropy_per_token.mean().item()
    std_entropy = entropy_per_token.std().item()

    return perplexity.item(), avg_entropy, std_entropy

# 示例
ai_text = "In the realm of artificial intelligence, neural networks have demonstrated remarkable capabilities in understanding natural language and generating coherent responses."
human_text = "Although I was tired after work, I took my dog for a walk. The evening breeze was refreshing, and the park was peaceful."

print("AI Generated:", compute_perplexity_entropy(ai_text))
print("Human Written:", compute_perplexity_entropy(human_text))
