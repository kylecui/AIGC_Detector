import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import math

# 加载预训练GPT-2模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# 关闭梯度计算，提高推理速度
@torch.no_grad()
def compute_perplexity_entropy(text):
    # 编码文本
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    # 获取模型输出
    outputs = model(input_ids, labels=input_ids)
    logits = outputs.logits

    # Softmax 计算概率
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # 获取目标token
    target_ids = input_ids[:, 1:]
    probs = probs[:, :-1, :]
    
    # 逐个token获取其生成概率
    token_probs = probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

    # 计算困惑度
    log_probs = torch.log(token_probs)
    perplexity = torch.exp(-log_probs.mean())

    # 计算熵
    entropy_per_token = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
    avg_entropy = entropy_per_token.mean().item()
    std_entropy = entropy_per_token.std().item()

    return {
        "perplexity": perplexity.item(),
        "avg_entropy": avg_entropy,
        "std_entropy": std_entropy,
        "token_count": input_ids.size(1)
    }

# 示例文本：一段AI生成文本和一段人写文本
texts = {
    "AI Generated": "While emerging technologies like AI, Big Data, and the IoT are driving significant gains in business productivity and enhancing our daily lives, they also introduce new vulnerabilities. The growing threat of cyberattacks, from persistent campaigns like APTs to crippling ransomware, can inflict severe damage on everything from data and financial assets to personal privacy. To counteract these evolving risks, the implementation of sophisticated cybersecurity defenses within our operational infrastructures is more crucial than ever.",
    "Human Written": "With new technologies emerging, such as AI, Big Data, IoT, and so on, our businesses are more efficient, and our life becomes more convenient. However, the concerns on cyber security come together with the benefits. The attacks on cyber security, such as APT or ransomware can damage data, privacy and secrets, assets, finances, and even more people’s lives. Therefore, more and more cyber security devices or techniques are implemented in our production environments."
}

results = {k: compute_perplexity_entropy(v) for k, v in texts.items()}
import pandas as pd
import ace_tools_open as tools

df = pd.DataFrame(results).T
df = df.rename(columns={
    "perplexity": "困惑度",
    "avg_entropy": "平均熵",
    "std_entropy": "熵标准差",
    "token_count": "Token数量"
})
tools.display_dataframe_to_user(name="AI文本检测指标对比", dataframe=df)
