# AIGC Detector：讨论整理与代码汇总

## 1. 项目背景

我们当前的目标是构建一个 **AIGC（AI Generated Content）文本检测系统**，并从最基础的统计特征出发，逐步走向可训练、可部署的检测模型。

这次讨论主要围绕以下几个方面展开：

1. **AIGC 检测的基本原理**
2. **困惑度（Perplexity）、熵（Entropy）、熵标准差（Entropy Std）的区别与作用**
3. **如何构建训练/微调数据集**
4. **如何使用本地开源模型生成 AI 样本**
5. **如何在 Windows + RTX 2080Ti + CUDA 12.9 环境下使用 `meta-llama/Meta-Llama-3-8B-Instruct`**
6. **如何以 offline mode 运行**

---

# 2. AIGC 检测的核心思路

## 2.1 初始猜想

一个很自然的猜想是：

> AI 生成的文本在逐词生成时，通常会倾向于选择“高概率词”，因此整篇文章从模型视角来看会更“平滑”、更“可预测”。

这个猜想本质上是合理的。

它可以拆成两个观察角度：

1. **困惑度（Perplexity）较低**
   - 说明语言模型更容易预测这段文本。
   - AI 生成文本通常比人类文本更符合语言模型自己的分布。

2. **熵（Entropy）与熵波动较小**
   - 熵描述的是在某个位置，模型对“下一个词”的不确定性。
   - AI 文本常常在更多位置表现出相对一致、平滑的分布。
   - 人类文本往往会在局部出现更突然、更跳跃、更难预测的表达。

---

# 3. 困惑度、熵、熵标准差：定义与区别

---

## 3.1 困惑度（Perplexity）

### 数学形式

对于一段 token 序列：

\[
x_1, x_2, ..., x_T
\]

困惑度定义为：

\[
\text{Perplexity} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log P(x_t|x_{<t})\right)
\]

### 直观含义

困惑度衡量的是：

> 模型在“正确 token”上平均有多大把握。

- 困惑度越低：模型越熟悉这段文本
- 困惑度越高：模型越“困惑”，觉得这段文本不符合自身分布

### 在 AIGC 检测中的意义

- **AI 文本**：通常困惑度更低
- **人类文本**：通常困惑度更高

---

## 3.2 熵（Entropy）

在某个位置 \(t\)，模型输出一个 softmax 分布：

\[
p_1, p_2, ..., p_V
\]

对应熵为：

\[
H_t = -\sum_{i=1}^{V} p_i \log p_i
\]

### 直观含义

熵衡量的是：

> 模型在这个位置上，对所有候选 token 的不确定性有多大。

- 熵低：模型非常偏向某个词，分布集中
- 熵高：模型对多个词都拿不准，分布分散

### 与困惑度的区别

- **困惑度**看的是：模型对“正确词”的打分
- **熵**看的是：模型对“整个候选空间”的犹豫程度

它们相关，但不相同。

---

## 3.3 熵标准差（Entropy Std）

我们可以对一段文本里每个 token 位置的熵做统计，得到：

- 平均熵 `avg_entropy`
- 熵标准差 `std_entropy`

### 含义

熵标准差表示：

> 文本中不同位置的不确定性波动程度有多大。

### 为什么它有价值

- **AI 文本**往往更平滑，某些位置不确定性变化不大
- **人类文本**常常在局部出现突兀表达、语义跳跃、风格变化，因此熵波动可能更大

所以，熵标准差是一个值得纳入 AIGC 检测的辅助特征。

---

# 4. AIGC 检测 Demo：基础思路

我们之前构想过一个简单的检测器，核心统计指标包括：

- `Perplexity`
- `Avg Entropy`
- `Entropy Std`

再基于这些指标构造一个简单的启发式评分函数：

```python
score = 100 - (perplexity / 2 + avg_entropy * 10 - std_entropy * 5)
````

其中：

* 困惑度越低，越像 AI
* 平均熵越低，越像 AI
* 熵标准差越低，越像 AI

这只是一个 **baseline**，后续真正训练模型时，更推荐把这些作为输入特征，而不是直接写死线性公式。

---

# 5. 从 Demo 走向可训练模型的路线图

## 5.1 清晰可行的路线

建议分成三个阶段：

### 阶段一：做基线特征系统

先用以下特征做 baseline：

* 困惑度
* 平均熵
* 熵标准差
* 句长分布
* 重复率
* 词汇多样性
* 标点风格
* n-gram 重复模式

用这些特征训练一个传统分类器：

* Logistic Regression
* Random Forest
* XGBoost

### 阶段二：构建监督数据集

构造三类数据：

* **H（Human）**：纯人类文本
* **A（AI）**：纯 AI 文本
* **M（Mixed）**：人机混合或 AI 改写文本

### 阶段三：微调文本分类模型

微调一个 encoder 模型作为检测器，例如：

* RoBERTa
* DeBERTa
* BERT
* 中文场景可尝试 MacBERT、RoBERTa-wwm-ext

同时可以考虑把数值特征与 encoder 输出融合。

---

# 6. 数据集构建方案

---

## 6.1 标签设计

推荐三分类：

* `H` = Human Written
* `A` = AI Generated
* `M` = Mixed / Human-edited AI / AI-edited Human

如果前期希望先简化，也可以先做二分类：

* `0` = Human
* `1` = AI

---

## 6.2 Human 文本来源

建议收集：

* 新闻
* 博客
* 论坛帖子
* 学术摘要
* 技术文档
* 英文写作材料
* 公开问答社区内容

注意：

* 去重
* 保持主题多样性
* 长度分布均衡
* 避免全部来自同一风格源

---

## 6.3 AI 文本来源

AI 文本可以通过本地大模型批量生成，建议覆盖：

* 技术说明文
* 新闻摘要
* 广告文案
* 邮件
* 法律声明
* 问答
* 散文/故事
* 诗歌
* 翻译文本

同时建议记录生成元信息：

* 模型名
* 温度
* top_p
* prompt
* 语言
* 长度

---

## 6.4 Mixed 文本构造方式

Mixed 样本非常重要，因为真实环境中最难检测的正是“经过编辑的 AI 文本”。

构造方式：

1. 人工对 AI 文本进行二次润色
2. 让 AI 对人类文本做改写
3. 回译（Back Translation）
4. 同义替换、轻微改写
5. 拼写修正、局部删改

---

# 7. 项目目录建议

```text
dataset/
├── human/
├── ai/
├── mixed/
└── meta.jsonl
```

每条样本对应一个文本文件，并在 `meta.jsonl` 中记录元信息。

例如：

```json
{
  "id": "a_000001",
  "split": "train",
  "source": "llama3-8b",
  "prompt": "Explain zero trust architecture",
  "temperature": 0.7,
  "label": "A",
  "lang": "en"
}
```

---

# 8. 为什么改为本地开源模型

一开始我们尝试过 OpenAI API 方案，但后续出现了几个现实问题：

1. `openai.ChatCompletion` 在新版 SDK 中被废弃
2. 异步 SDK 需要适配新版 `OpenAI()` / `AsyncOpenAI()`
3. 账户配额不足（`insufficient_quota`）
4. 本地批量生成样本成本较高
5. 实验数据集构造更适合使用离线本地模型

因此最终选择：

> 使用本地的 **meta-llama/Meta-Llama-3-8B-Instruct**

---

# 9. 为什么最终仍选择 Llama 3

中间我们也比较过一些无需授权的模型，例如：

* Mistral-7B-Instruct
* Falcon-7B-Instruct
* OpenChat-3.5
* Phi-2

但后来你已经拿到了：

> `meta-llama/Meta-Llama-3-8B-Instruct` 的授权

因此最终我们仍然回到 Llama 3。

原因是：

1. 模型质量更高
2. 作为生成 A 类数据的来源更强
3. 后续如果要做“检测对抗”，Llama 3 更有代表性
4. 你已经完成权限申请，条件已满足

---

# 10. 硬件与环境约束

当前环境：

* **GPU**：RTX 2080 Ti 11 GB
* **系统**：Windows
* **CUDA**：12.9

这带来几个现实约束：

1. `flash-attn` / Triton 在 Windows + Turing 上兼容性差
2. bitsandbytes 的 GPU 4bit 在 Windows 上不稳定
3. 直接跑完整 FP16 版 Llama 3 8B 会有压力
4. 最稳妥的方式是：

   * 先把模型完整下载到本地
   * 使用 `transformers + accelerate + device_map="auto"` 做 CPU/GPU offload
   * 使用 offline mode

---

# 11. Offline Mode 方案

你的思路是正确的：

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="meta-llama/Llama-2-7b-hf", repo_type="model")
```

我们将其整理进实际脚本中，变成一个完整流程：

1. **在线阶段**：

   * 使用 `snapshot_download()` 下载 `meta-llama/Meta-Llama-3-8B-Instruct`
   * 携带 `HF_TOKEN`
   * 下载到本地目录，例如 `models/llama3`

2. **离线阶段**：

   * 后续所有推理都从 `models/llama3` 加载
   * 使用 `local_files_only=True`
   * 支持 `--offline`

---

# 12. 最终版本：generate_ai_llama3_offload.py

下面是整理后的推荐脚本。

---

## 12.1 脚本功能

* 若本地没有模型，则在线下载一次
* 若本地已有模型，则直接离线加载
* 支持 `--offline`
* 使用 `meta-llama/Meta-Llama-3-8B-Instruct`
* 使用 `transformers + accelerate` 自动 offload
* 生成的文本保存到 `dataset/ai`
* 元数据追加到 `dataset/meta.jsonl`

---

## 12.2 完整代码

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate AI samples with Meta-Llama-3-8B-Instruct
- 自动下载或复用本地模型
- 支持 offline mode
- 使用 accelerate 自动做 GPU/CPU offload
"""

import os
import json
import uuid
import argparse
import pathlib

from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
LOCAL_DIR = pathlib.Path("models/llama3")

GPU_MEM = "10GiB"
CPU_MEM = "16GiB"
MAX_NEW_TOKENS = 256

parser = argparse.ArgumentParser()
parser.add_argument("--offline", action="store_true", help="完全离线模式，不联网下载")
args = parser.parse_args()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN and not args.offline:
    print("⚠️ 未检测到 HF_TOKEN。若本地模型不存在，且仓库是 gated repo，则会下载失败。")

# 1. 若模型不存在，则先下载
if not LOCAL_DIR.exists() or not any(LOCAL_DIR.iterdir()):
    if args.offline:
        raise RuntimeError(f"离线模式下未找到本地模型目录：{LOCAL_DIR}")

    print(f"⏬ 开始下载模型：{REPO_ID}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=HF_TOKEN,
    )
    print("✅ 模型下载完成")

# 2. 离线加载模型
print("🚀 正在加载 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_DIR,
    local_files_only=True,
    use_fast=True,
)

print("🚀 正在加载 model（GPU/CPU offload）...")
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_DIR,
    torch_dtype="auto",
    device_map="auto",
    max_memory={0: GPU_MEM, "cpu": CPU_MEM},
    local_files_only=True,
)

gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

prompts = [
    "Explain the core ideas of zero-trust architecture to a CTO.",
    "Write a short story (300 words) about an AI detective.",
    "Summarize the advantages of IPv6 in 5 bullet points.",
    "Generate a marketing email for a new cybersecurity SaaS.",
    "Translate the following Chinese paragraph into native English:\n零信任是一种全新的安全架构……",
    "Draft a legal disclaimer for a tech blog.",
    "Produce a poem using iambic pentameter on data privacy.",
    "Compare LSTM and Transformer models in a classroom tone.",
    "Give me 10 interview questions about cloud security.",
    "Describe the future of quantum computing for non-experts.",
]

temperatures = (0.7, 1.0)

out_dir = pathlib.Path("dataset/ai")
out_dir.mkdir(parents=True, exist_ok=True)

meta_path = pathlib.Path("dataset/meta.jsonl")
meta_path.parent.mkdir(parents=True, exist_ok=True)

with open(meta_path, "a", encoding="utf-8") as meta_fp:
    for prompt in tqdm(prompts, desc="prompts"):
        for temp in temperatures:
            result = gen(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=temp,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )

            text = result[0]["generated_text"].strip()

            uid = f"a_{uuid.uuid4().hex[:8]}"
            sample_path = out_dir / f"{uid}.txt"
            sample_path.write_text(text, encoding="utf-8")

            meta_fp.write(json.dumps({
                "id": uid,
                "split": "train",
                "source": "llama3-8b-fp16-offload",
                "prompt": prompt,
                "temperature": temp,
                "label": "A",
                "lang": "en"
            }, ensure_ascii=False) + "\n")

print("🎉 样本生成完成")
```

---

# 13. 使用方法

## 13.1 在线首次下载 + 生成

### Windows PowerShell

```powershell
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
python generate_ai_llama3_offload.py
```

---

## 13.2 离线生成

当你已经下载过模型以后：

```powershell
python generate_ai_llama3_offload.py --offline
```

这时脚本不会再访问 Hugging Face。

---

# 14. 常见问题与处理建议

---

## 14.1 Gated Repo 无法访问

典型报错：

```text
Cannot access gated repo ...
You must have access to it and be authenticated to access it.
```

原因：

* 你虽然网页上已拿到授权，但本地脚本没有带 `HF_TOKEN`

解决方式：

```powershell
$env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
```

然后再运行下载。

---

## 14.2 离线模式下找不到模型

如果使用：

```powershell
python generate_ai_llama3_offload.py --offline
```

但本地没有 `models/llama3`，就会报错。

解决方式：

* 先在线下载一次
* 确保模型目录完整存在

---

## 14.3 显存不足（OOM）

当前脚本默认：

* GPU: `10GiB`
* CPU: `16GiB`
* `max_new_tokens = 256`

如果仍然显存紧张，可以继续降低：

```python
GPU_MEM = "9GiB"
MAX_NEW_TOKENS = 128
```

---

## 14.4 速度较慢

由于 2080Ti 11 GB 无法完整容纳 Llama 3 8B，所以采用了 GPU/CPU offload。

因此速度会慢于纯 GPU 推理，这是正常现象。

优化方向：

1. 减少 `MAX_NEW_TOKENS`
2. 缩小 prompt 数量
3. 后续改到 Linux / WSL2 环境做更高效量化
4. 若只做数据构造，可接受当前速度

---

# 15. 下一步建议

在当前阶段，建议按以下顺序推进：

## 15.1 先生成 A 类样本

使用当前脚本批量生成 AI 文本。

## 15.2 同步准备 H 类样本

收集人类文本，保持领域与长度多样性。

## 15.3 再构造 M 类样本

做人机混合、AI 改写、人类润色等。

## 15.4 先做 baseline 分类器

先尝试：

* 困惑度
* 平均熵
* 熵标准差
* 其他风格特征

训练：

* Logistic Regression
* XGBoost

## 15.5 最后微调文本分类模型

数据足够后，再考虑训练：

* RoBERTa
* DeBERTa
* BERT 类 encoder 检测模型

---

# 16. 总结

本次讨论形成了以下共识：

1. **AIGC 检测可以从困惑度、熵、熵标准差等统计特征入手**
2. **你的“AI 文本更平滑、更高置信度”的猜想是合理的**
3. **真正落地时，需要构造 H / A / M 三类数据集**
4. **由于 OpenAI API 配额与环境问题，我们转向本地开源模型**
5. **最终仍选择 `meta-llama/Meta-Llama-3-8B-Instruct`**
6. **在 Windows + 2080Ti + CUDA 12.9 环境下，最稳妥的是离线下载 + accelerate offload**
7. **目前最实用的代码方案就是 `generate_ai_llama3_offload.py`**

---

# 17. 后续可扩展方向

后续我们还可以继续完善：

1. **自动构建 H / A / M 数据集脚本**
2. **困惑度/熵特征提取脚本**
3. **baseline 分类器训练脚本**
4. **RoBERTa / DeBERTa 微调脚本**
5. **可视化分析工具**
6. **混合文本检测**
7. **对抗改写鲁棒性实验**

---
