# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# import numpy as np

# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")
# model.eval()

# @torch.no_grad()
# def detect_ai_likelihood(text):
#     inputs = tokenizer(text, return_tensors="pt")
#     input_ids = inputs["input_ids"]
#     outputs = model(input_ids, labels=input_ids)
#     logits = outputs.logits

#     probs = torch.nn.functional.softmax(logits, dim=-1)
#     target_ids = input_ids[:, 1:]
#     probs = probs[:, :-1, :]
#     token_probs = probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

#     log_probs = torch.log(token_probs + 1e-12)
#     perplexity = torch.exp(-log_probs.mean()).item()

#     entropy_per_token = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
#     avg_entropy = entropy_per_token.mean().item()
#     std_entropy = entropy_per_token.std().item()

#     score = 100 - (perplexity / 2 + avg_entropy * 10)
#     score = np.clip(score, 0, 100)
#     label = "Likely AI-Generated" if score > 50 else "Likely Human-Written"

#     return {
#         "Perplexity": round(perplexity, 2),
#         "Avg Entropy": round(avg_entropy, 3),
#         "Entropy Std": round(std_entropy, 3),
#         "AI Likelihood Score (0-100)": round(score, 1),
#         "Prediction": label
#     }

# # 示例调用
# if __name__ == "__main__":
#     text = input("请输入一段英文文本：\n")
#     result = detect_ai_likelihood(text)
#     for k, v in result.items():
#         print(f"{k}: {v}")

import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

@torch.no_grad()
def detect_ai_score_with_entropy_std(text):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    outputs = model(input_ids, labels=input_ids)
    logits = outputs.logits

    probs = torch.nn.functional.softmax(logits, dim=-1)
    target_ids = input_ids[:, 1:]
    probs = probs[:, :-1, :]
    token_probs = probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    log_probs = torch.log(token_probs + 1e-12)
    perplexity = torch.exp(-log_probs.mean()).item()

    entropy_per_token = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    avg_entropy = entropy_per_token.mean().item()
    std_entropy = entropy_per_token.std().item()

    # 综合评分（引入熵标准差作为风格“自然性”的鼓励因子）
    score = 100 - (perplexity / 2 + avg_entropy * 10 - std_entropy * 5)
    score = np.clip(score, 0, 100)
    label = "Likely AI-Generated" if score > 50 else "Likely Human-Written"

    return {
        "Perplexity": round(perplexity, 2),
        "Avg Entropy": round(avg_entropy, 3),
        "Entropy Std": round(std_entropy, 3),
        "AI Likelihood Score (0-100)": round(score, 1),
        "Prediction": label
    }

# 示例调用
text = input("请输入要检测的英文文本：\n")
result = detect_ai_score_with_entropy_std(text)
for k, v in result.items():
    print(f"{k}: {v}")
