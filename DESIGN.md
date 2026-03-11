# AIGC Detector — 系统设计文档

> **版本**: 1.0  
> **日期**: 2026-03-11  
> **状态**: Draft — 待评审  
> **硬件**: NVIDIA RTX 3060 12GB / Windows  
> **语言范围**: 中文 + 英文（双语）

---

## 目录

1. [系统概述](#1-系统概述)
2. [整体架构](#2-整体架构)
3. [数据管线](#3-数据管线)
4. [检测引擎](#4-检测引擎)
5. [Web API 设计](#5-web-api-设计)
6. [前端设计](#6-前端设计)
7. [模型选型](#7-模型选型)
8. [VRAM 预算](#8-vram-预算)
9. [项目结构](#9-项目结构)
10. [依赖管理](#10-依赖管理)
11. [安全与运维](#11-安全与运维)

---

## 1. 系统概述

### 1.1 目标

构建一个 **双语（中英文）AI 生成文本检测系统**，通过 Web API + Web 页面提供服务。系统接受用户输入文本，返回：

- **二分类标签**: AI-generated / Human-written
- **置信度分数**: 0.0–1.0
- **特征分解**: 各检测子模块的独立判断结果

### 1.2 设计原则

| 原则 | 说明 |
|------|------|
| **级联 Pipeline** | 统计特征快速过滤 → 编码器精确分类 → 零样本兜底 |
| **双语路由** | 语言检测后分流到中文 / 英文独立管线，避免多语言模型精度损失 |
| **单 GPU 约束** | 所有推理在 RTX 3060 12GB 内完成，模型按需加载 |
| **渐进式构建** | 5 个 Phase，每个 Phase 可独立验证 |

### 1.3 非目标（V1 不做）

- 多 GPU / 分布式推理
- 实时流式检测（V1 为请求-响应模式）
- 对抗训练（V1 仅在评估阶段测试鲁棒性）
- 手动编辑的混合文本数据（RC 阶段引入）
- 多语言扩展（V1 仅中英文）

---

## 2. 整体架构

```
                    ┌──────────────────────────────────────┐
                    │           Web Frontend               │
                    │  (Alpine.js + HTML/CSS)               │
                    │  - 文本输入区                         │
                    │  - 检测结果展示                       │
                    │  - 特征分解可视化                     │
                    └───────────────┬──────────────────────┘
                                    │ HTTP POST /api/v1/detect
                                    ▼
                    ┌──────────────────────────────────────┐
                    │           FastAPI Server              │
                    │  - Uvicorn ASGI (1 worker)            │
                    │  - 请求队列 + Semaphore 并发控制       │
                    │  - 速率限制 (SlowAPI)                 │
                    │  - Pydantic 输入验证                  │
                    └───────────────┬──────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────┐
                    │         Language Router               │
                    │  papluca/xlm-roberta-base-            │
                    │  language-detection                   │
                    │  (~100ms, ~1GB VRAM)                  │
                    └──────┬─────────────────┬─────────────┘
                           │                 │
                ┌──────────▼───────┐ ┌───────▼──────────┐
                │  Chinese Pipeline │ │ English Pipeline  │
                │                  │ │                   │
                │ ┌──────────────┐ │ │ ┌───────────────┐ │
                │ │ 统计特征层   │ │ │ │ 统计特征层    │ │
                │ │ Qwen2-7B-Q4  │ │ │ │ GPT-2-XL      │ │
                │ │ 困惑度+熵    │ │ │ │ 困惑度+熵     │ │
                │ └──────┬───────┘ │ │ └──────┬────────┘ │
                │        │         │ │        │          │
                │ ┌──────▼───────┐ │ │ ┌──────▼────────┐ │
                │ │ 编码器分类层 │ │ │ │ 编码器分类层  │ │
                │ │ chinese-     │ │ │ │ deberta-v3-   │ │
                │ │ roberta-wwm- │ │ │ │ large         │ │
                │ │ ext-large    │ │ │ │               │ │
                │ │ (LoRA微调)   │ │ │ │ (LoRA微调)    │ │
                │ └──────┬───────┘ │ │ └──────┬────────┘ │
                │        │         │ │        │          │
                │ ┌──────▼───────┐ │ │ ┌──────▼────────┐ │
                │ │ 零样本层     │ │ │ │ 零样本层      │ │
                │ │ Binoculars   │ │ │ │ Binoculars    │ │
                │ │ (Qwen2 pair) │ │ │ │ (Falcon pair) │ │
                │ └──────┬───────┘ │ │ └──────┬────────┘ │
                │        │         │ │        │          │
                └────────┼─────────┘ └────────┼──────────┘
                         │                    │
                ┌────────▼────────────────────▼──────────┐
                │           Ensemble Aggregator           │
                │  - 加权投票                              │
                │  - 级联逻辑: 统计→编码器→零样本           │
                │  - 输出: label + confidence + breakdown  │
                └─────────────────────────────────────────┘
```

### 2.1 级联检测逻辑

```
输入文本
  │
  ├─ 语言检测 (Language Router)
  │    ├─ 中文 → Chinese Pipeline
  │    └─ 英文 → English Pipeline
  │
  ├─ Stage 1: 统计特征 (快速，<200ms)
  │    ├─ 困惑度 (Perplexity)
  │    ├─ 平均熵 (Avg Entropy)
  │    ├─ 熵标准差 (Std Entropy)
  │    └─ 结果: 如果置信度 > 0.95 → 直接返回 (跳过后续阶段)
  │
  ├─ Stage 2: 编码器分类 (中等，300-500ms)
  │    ├─ 微调后的 DeBERTa/RoBERTa
  │    ├─ 输出: P(AI) 概率
  │    └─ 结果: 与统计特征加权融合
  │
  └─ Stage 3: 零样本检测 (慢，1-3s) [仅当前两阶段不一致时]
       ├─ Binoculars score
       └─ 结果: 最终仲裁
```

**级联优势**: 90%+ 请求在 Stage 1-2 就能返回高置信结果，仅 5-10% 的困难样本需要 Stage 3。

---

## 3. 数据管线

### 3.1 数据源总览

| 类别 | 来源 | 语言 | 预期数量 |
|------|------|------|----------|
| **人类文本** | HC3-Chinese, HC3-English, Wikipedia, 新闻爬取 | 中+英 | 各 25K+ |
| **AI 生成文本** | 本地 LLM 生成 (多模型) | 中+英 | 各 25K+ |
| **混合文本** | 自动构造 (AI Completion + Sentence Insertion) | 中+英 | 各 10K+ |

### 3.2 人类文本收集

**中文来源:**
- `Hello-SimpleAI/HC3-Chinese` (HuggingFace) — ~13,000 问答对
- 中文维基百科文章
- 新闻网站爬取 (新华社, 澎湃新闻等) — 改进现有 `news_crawl.py`

**英文来源:**
- `Hello-SimpleAI/HC3` English subset
- English Wikipedia (random articles)
- 新闻网站 (The Verge, Ars Technica 等)

**质量过滤:**
```python
def filter_human_text(text: str) -> bool:
    """人类文本质量门控"""
    if len(text) < 200:           # 太短无法可靠检测
        return False
    if len(text) > 10000:         # 截断
        return False
    if detect_boilerplate(text):  # 模板文本（Cookie 声明、导航等）
        return False
    if detect_encoding_issues(text):
        return False
    return True
```

### 3.3 AI 文本生成

**多模型生成策略** — 避免单模型过拟合：

| 模型 | 语言 | HuggingFace ID | 量化 | VRAM |
|------|------|----------------|------|------|
| Mistral-7B-Instruct-v0.2 | 英文 | `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ` | GPTQ 4-bit | ~4.5GB |
| Qwen2.5-7B-Instruct | 中文+英文 | `Qwen/Qwen2.5-7B-Instruct` | BnB 4-bit | ~5.5GB |
| ChatGLM3-6B | 中文 | `THUDM/chatglm3-6b` | BnB int4 | ~5GB |
| Yi-6B-Chat | 中文+英文 | `01-ai/Yi-6B-Chat` | BnB 4-bit | ~4.5GB |
| Llama-3-8B-Instruct | 英文 | `meta-llama/Meta-Llama-3-8B-Instruct` | GPTQ 4-bit | ~5.5GB |

**关键**: 所有 Instruct 模型必须使用 `apply_chat_template()` 生成，否则输出质量严重下降。

```python
# 正确的生成方式
messages = [
    {"role": "system", "content": "你是一个专业的新闻记者。"},
    {"role": "user", "content": "请写一篇关于人工智能在医疗领域应用的文章，约500字。"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

**生成多样性控制:**
- 多种 prompt 模板 (新闻报道、议论文、科普、评论等)
- 多领域 (科技、教育、医疗、法律、金融)
- 多长度 (200-2000 字)
- `temperature` 在 [0.7, 1.0] 范围内变化
- `top_p` 在 [0.85, 0.95] 范围内变化

### 3.4 混合文本自动构造

**V1 仅使用自动化方法，手动编辑的混合文本延迟到 RC 阶段。**

#### 方法 A: AI Completion (50/50 切分)

```python
def ai_completion(human_text: str, model, tokenizer, split_ratio: float = 0.5) -> dict:
    """
    取人类文本前半段，用 LLM 续写后半段。
    最真实的混合文本构造方法。
    """
    sentences = split_sentences_bilingual(human_text)
    split_point = int(len(sentences) * split_ratio)
    
    human_part = ''.join(sentences[:split_point])  # 中文不需要空格连接
    
    prompt = f"请继续以下文本，保持相同的风格和主题：\n\n{human_part}"
    ai_part = generate(model, tokenizer, prompt)
    
    return {
        "text": human_part + ai_part,
        "label": "mixed",
        "boundary_char_idx": len(human_part),
        "human_ratio": split_ratio,
        "method": "ai_completion"
    }
```

#### 方法 B: Sentence Insertion (30% AI 句子插入)

```python
def sentence_insertion(human_text: str, model, tokenizer, insert_ratio: float = 0.3) -> dict:
    """
    在人类文本中随机位置插入 AI 生成的句子。
    基于 SenDetEX (EMNLP 2025) 方法。
    """
    sentences = split_sentences_bilingual(human_text)
    num_insertions = max(1, int(len(sentences) * insert_ratio))
    insert_positions = sorted(random.sample(range(len(sentences)), num_insertions))
    
    sentence_labels = ['human'] * len(sentences)
    
    for offset, pos in enumerate(insert_positions):
        actual_pos = pos + offset
        context = ' '.join(sentences[max(0, actual_pos-2):actual_pos+1])
        ai_sentence = generate_single_sentence(model, tokenizer, context)
        sentences.insert(actual_pos + 1, ai_sentence)
        sentence_labels.insert(actual_pos + 1, 'ai')
    
    return {
        "text": ''.join(sentences),
        "label": "mixed",
        "sentence_labels": sentence_labels,
        "method": "sentence_insertion"
    }
```

#### 中文分句处理

```python
import re

def split_sentences_bilingual(text: str) -> list[str]:
    """
    双语分句。中文按句号/问号/感叹号分割，英文用 NLTK。
    """
    if re.search(r'[\u4e00-\u9fff]', text):
        # 中文: 按标点分句，保留标点
        sentences = re.split(r'(?<=[。！？；\n])', text)
    else:
        # 英文: NLTK
        import nltk
        sentences = nltk.sent_tokenize(text)
    
    return [s for s in sentences if s.strip()]
```

### 3.5 数据集格式

```jsonl
{"id": "h_0001", "text": "...", "label": "human", "lang": "zh", "source": "hc3", "domain": "news"}
{"id": "a_0001", "text": "...", "label": "ai", "lang": "zh", "source": "qwen2.5", "domain": "news", "gen_params": {"temperature": 0.8, "top_p": 0.9}}
{"id": "m_0001", "text": "...", "label": "mixed", "lang": "zh", "method": "ai_completion", "boundary_char_idx": 342, "human_ratio": 0.5}
```

### 3.6 数据集规模目标

| 类别 | 中文 | 英文 | 合计 |
|------|------|------|------|
| Human | 25,000 | 25,000 | 50,000 |
| AI | 25,000 (5 模型 × 5,000) | 25,000 (5 模型 × 5,000) | 50,000 |
| Mixed | 10,000 | 10,000 | 20,000 |
| **总计** | **60,000** | **60,000** | **120,000** |

训练/验证/测试划分: 80% / 10% / 10%

---

## 4. 检测引擎

### 4.1 统计特征模块

**计算指标:**
- **困惑度 (Perplexity)**: 参考模型对文本的交叉熵指数
- **逐 token 熵**: 每个 token 的预测概率分布的信息熵
- **平均熵 / 熵标准差**: 聚合统计量
- **Burstiness**: 人类文本的熵波动大，AI 文本更平坦

**参考模型选择:**

| 语言 | 参考模型 | 理由 |
|------|----------|------|
| 中文 | `Qwen/Qwen2.5-7B-Instruct` (4-bit) | 最强中文 7B，官方 GPTQ/BnB 支持 |
| 英文 | `openai-community/gpt2-xl` (FP16) | 经典参考模型，1.5B 参数，~3GB VRAM |

**注意**: 参考模型必须与生成模型不同，避免循环推理。

```python
class StatisticalFeatureExtractor:
    """统计特征提取器"""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", load_in_4bit=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
    
    def extract(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits  # (1, seq_len, vocab_size)
        
        # 逐 token 概率分布
        probs = torch.softmax(logits[:, :-1, :], dim=-1)
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        
        # 逐 token 熵: H = -sum(p * log(p))
        token_entropies = -(probs * log_probs).sum(dim=-1).squeeze()
        
        # 逐 token 对数概率 (用于 perplexity)
        target_ids = inputs["input_ids"][:, 1:]
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)
        
        return {
            "perplexity": torch.exp(-token_log_probs.mean()).item(),
            "avg_entropy": token_entropies.mean().item(),
            "std_entropy": token_entropies.std().item(),
            "burstiness": self._burstiness(token_entropies),
            "max_entropy": token_entropies.max().item(),
            "min_entropy": token_entropies.min().item(),
        }
    
    def _burstiness(self, entropies: torch.Tensor) -> float:
        """Burstiness = (std - mean) / (std + mean)，衡量熵波动"""
        std = entropies.std().item()
        mean = entropies.mean().item()
        return (std - mean) / (std + mean + 1e-8)
```

### 4.2 编码器分类模块

**模型选择:**

| 语言 | 模型 | 参数量 | VRAM (FP16) |
|------|------|--------|-------------|
| 中文 | `hfl/chinese-roberta-wwm-ext-large` | 325M | ~4GB |
| 英文 | `microsoft/deberta-v3-large` | 435M | ~4GB |

**微调方式**: LoRA (Low-Rank Adaptation)
- rank = 16, alpha = 32
- 仅微调 attention 层
- VRAM 需求: ~6-7GB（含梯度）
- 训练时间: ~2h (50K 样本, batch_size=16)

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_proj", "value_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-large", num_labels=2
)
model = get_peft_model(model, lora_config)
```

### 4.3 零样本检测模块 (Binoculars)

**原理**: 对比 Observer (基座模型) 和 Performer (指令模型) 对同一文本的交叉熵差异。

**模型对:**

| 语言 | Observer | Performer | VRAM (合计) |
|------|----------|-----------|-------------|
| 中文 | `Qwen/Qwen2-7B` (4-bit) | `Qwen/Qwen2-7B-Instruct` (4-bit) | ~9GB |
| 英文 | `tiiuae/falcon-7b` (4-bit) | `tiiuae/falcon-7b-instruct` (4-bit) | ~9GB |

**⚠️ 重要**: Binoculars 的中文支持为实验性质。阈值需要在中文验证集上重新校准。

**VRAM 限制**: 两个 7B 模型无法同时加载 (9GB)。需要与统计特征模块共享 Qwen2 实例，仅额外加载 Instruct 版本。

```python
class BinocularsDetector:
    """Binoculars 零样本检测器"""
    
    # 英文阈值 (来自原论文)
    EN_ACCURACY_THRESHOLD = 0.9015
    EN_FPR_THRESHOLD = 0.8536
    
    # 中文阈值 (需要在验证集上校准)
    ZH_ACCURACY_THRESHOLD = None  # TODO: 校准
    ZH_FPR_THRESHOLD = None       # TODO: 校准
    
    def __init__(self, observer_path: str, performer_path: str):
        self.observer = AutoModelForCausalLM.from_pretrained(
            observer_path, device_map="auto", load_in_4bit=True
        )
        self.performer = AutoModelForCausalLM.from_pretrained(
            performer_path, device_map="auto", load_in_4bit=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(observer_path)
    
    def compute_score(self, text: str) -> float:
        """
        Binoculars score = CE_performer(text) / CE_observer(text)
        score < threshold → AI-generated
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            observer_loss = self.observer(**inputs, labels=inputs["input_ids"]).loss
            performer_loss = self.performer(**inputs, labels=inputs["input_ids"]).loss
        
        return (performer_loss / observer_loss).item()
```

### 4.4 集成聚合器

```python
class EnsembleAggregator:
    """
    级联 + 加权集成。
    Stage 1 (统计) → Stage 2 (编码器) → Stage 3 (零样本，仅冲突时)
    """
    
    WEIGHTS = {
        "statistical": 0.2,
        "encoder": 0.5,
        "binoculars": 0.3,
    }
    
    def predict(self, text: str, lang: str) -> dict:
        # Stage 1: 统计特征
        stat_features = self.stat_extractor.extract(text)
        stat_score = self.stat_classifier.predict(stat_features)
        
        if stat_score["confidence"] > 0.95:
            return self._format_result(stat_score, stages_used=["statistical"])
        
        # Stage 2: 编码器分类
        encoder_score = self.encoder_classifier.predict(text, lang)
        
        # 检查 Stage 1 和 2 是否一致
        if self._agree(stat_score, encoder_score):
            combined = self._weighted_combine(stat_score, encoder_score)
            return self._format_result(combined, stages_used=["statistical", "encoder"])
        
        # Stage 3: Binoculars 仲裁 (仅 Stage 1-2 冲突时)
        bino_score = self.binoculars.compute_score(text)
        final = self._weighted_combine(stat_score, encoder_score, bino_score)
        return self._format_result(final, stages_used=["statistical", "encoder", "binoculars"])
    
    def _format_result(self, score, stages_used) -> dict:
        return {
            "predicted_label": "AI-generated" if score["p_ai"] > 0.5 else "Human-written",
            "confidence": max(score["p_ai"], 1 - score["p_ai"]),
            "p_ai": score["p_ai"],
            "stages_used": stages_used,
            "breakdown": score.get("breakdown", {}),
        }
```

---

## 5. Web API 设计

### 5.1 技术栈

| 组件 | 选型 | 版本 | 理由 |
|------|------|------|------|
| Web 框架 | FastAPI | ≥0.110 | 异步优先，自动 OpenAPI 文档 |
| ASGI 服务器 | Uvicorn | ≥0.30 | Windows 兼容性最佳 |
| 速率限制 | SlowAPI | ≥0.1.9 | 基于 IP 的请求限流 |
| 验证 | Pydantic v2 | ≥2.0 | 输入验证 + 序列化 |
| 推理 | Transformers + BitsAndBytes | ≥4.40 | 模型加载和量化推理 |

### 5.2 API 端点

#### `POST /api/v1/detect`

**请求:**
```json
{
    "text": "待检测的文本内容...",
    "models": ["all"]
}
```

**约束:**
- `text`: 50–10,000 字符
- `models`: `["all"]` | `["statistical"]` | `["encoder"]` | `["binoculars"]` | 组合

**响应:**
```json
{
    "predicted_label": "AI-generated",
    "confidence": 0.87,
    "p_ai": 0.87,
    "detected_language": "zh",
    "stages_used": ["statistical", "encoder"],
    "breakdown": {
        "statistical": {
            "perplexity": 23.5,
            "avg_entropy": 3.2,
            "std_entropy": 1.1,
            "burstiness": -0.48,
            "score": 0.82
        },
        "encoder": {
            "model": "chinese-roberta-wwm-ext-large",
            "p_ai": 0.89
        },
        "binoculars": null
    },
    "processing_time_ms": 450
}
```

#### `GET /api/v1/health`

```json
{
    "status": "healthy",
    "models_loaded": ["statistical_zh", "statistical_en", "encoder_zh", "encoder_en"],
    "gpu_memory_used_mb": 8500,
    "gpu_memory_total_mb": 12288,
    "uptime_seconds": 3600
}
```

### 5.3 并发控制

```python
# 单 GPU 并发策略
MAX_CONCURRENT_REQUESTS = 2      # 同时使用 GPU 的请求数
MAX_QUEUE_SIZE = 50              # 等待队列上限
QUEUE_TIMEOUT_SECONDS = 120      # 队列超时
RATE_LIMIT = "10/minute"         # 每 IP 每分钟请求数

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

@app.post("/api/v1/detect")
@limiter.limit(RATE_LIMIT)
async def detect_text(request: Request, data: DetectionRequest):
    try:
        async with asyncio.timeout(QUEUE_TIMEOUT_SECONDS):
            async with semaphore:
                result = await run_in_threadpool(detector.predict, data.text)
                return result
    except asyncio.TimeoutError:
        raise HTTPException(503, "Server busy, please retry later")
```

### 5.4 模型生命周期

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型，关闭时释放 VRAM"""
    logger.info("Loading models...")
    
    # 按需加载策略: 优先加载高频使用的模型
    app.state.language_router = load_language_router()
    app.state.stat_zh = load_statistical_model("zh")
    app.state.stat_en = load_statistical_model("en")
    app.state.encoder_zh = load_encoder_model("zh")
    app.state.encoder_en = load_encoder_model("en")
    # Binoculars 延迟加载 (仅在需要时)
    app.state.binoculars_zh = None
    app.state.binoculars_en = None
    
    logger.info("All models loaded. Ready to serve.")
    yield
    
    # 清理
    torch.cuda.empty_cache()
    logger.info("Shutdown complete.")

app = FastAPI(lifespan=lifespan)
```

---

## 6. 前端设计

### 6.1 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| 框架 | Alpine.js + 原生 HTML/CSS | 无构建步骤，轻量，够用 |
| 样式 | 内联 CSS 或 Tailwind CDN | 不需要 npm 构建 |
| 部署 | FastAPI StaticFiles 托管 | 同一进程，无 CORS |

### 6.2 页面功能

1. **文本输入区**: `<textarea>` 支持粘贴，显示字符计数
2. **检测按钮**: 带 loading 状态
3. **结果展示**:
   - 标签 (AI-generated / Human-written) + 颜色标记
   - 置信度进度条
   - 特征分解卡片 (困惑度、编码器分数、Binoculars 分数)
4. **语言标识**: 显示检测到的语言
5. **响应时间**: 显示处理耗时

### 6.3 文件结构

```
static/
  ├── index.html      # 主页面 (含 Alpine.js 逻辑)
  ├── style.css        # 样式
  └── favicon.ico
```

---

## 7. 模型选型汇总

### 7.1 推理模型

| 用途 | 中文模型 | 英文模型 | VRAM |
|------|----------|----------|------|
| 语言检测 | `papluca/xlm-roberta-base-language-detection` | 同左 | ~1GB |
| 统计特征 (困惑度) | `Qwen/Qwen2.5-7B-Instruct` (4-bit) | `openai-community/gpt2-xl` (FP16) | ~5.5GB / ~3GB |
| 编码器分类 | `hfl/chinese-roberta-wwm-ext-large` + LoRA | `microsoft/deberta-v3-large` + LoRA | ~4GB / ~4GB |
| 零样本 (Binoculars) | Qwen2-7B + Qwen2-7B-Instruct | Falcon-7B + Falcon-7B-Instruct | ~9GB (pair) |

### 7.2 数据生成模型

| 模型 | 语言 | HuggingFace ID | 量化 |
|------|------|----------------|------|
| Mistral-7B-v0.2 | 英文 | `TheBloke/Mistral-7B-Instruct-v0.2-GPTQ` | GPTQ 4-bit |
| Qwen2.5-7B | 中+英 | `Qwen/Qwen2.5-7B-Instruct` | BnB 4-bit |
| ChatGLM3-6B | 中文 | `THUDM/chatglm3-6b` | BnB int4 |
| Yi-6B-Chat | 中+英 | `01-ai/Yi-6B-Chat` | BnB 4-bit |
| Llama-3-8B | 英文 | `meta-llama/Meta-Llama-3-8B-Instruct` | GPTQ 4-bit |

### 7.3 已有本地模型

| 路径 | 模型 | 状态 | 保留? |
|------|------|------|-------|
| `models/mistral7b-gptq/` | Mistral-7B-GPTQ | ✅ 完整 | ✅ 用于英文数据生成 |
| `models/llama3/` | Llama-3-8B FP16 | ⚠️ 不完整，有 `.incomplete` 文件 | ❌ 重新下载 GPTQ 版本 |
| `models/llama3-8b/` | Llama-3-8B | ❌ 仅 README + LICENSE | ❌ 删除 |

---

## 8. VRAM 预算

### 8.1 推理时 VRAM 分配 (12GB)

**不能同时加载所有模型。** 必须按需加载。

#### 常驻模型 (始终加载):
| 模型 | VRAM |
|------|------|
| Language Router (XLM-R base) | ~1.0 GB |
| 编码器 (中文 OR 英文, 按路由) | ~4.0 GB |
| **小计** | **~5.0 GB** |

#### 按需加载:
| 模型 | VRAM | 触发条件 |
|------|------|----------|
| 统计特征 LLM (Qwen2.5 OR GPT-2-XL) | ~5.5 / ~3.0 GB | Stage 1 |
| Binoculars (额外一个 7B 模型) | ~5.5 GB | Stage 3 (冲突时) |

#### 推理时序 (同一请求):
```
1. Language Router (常驻, 1GB)
2. 统计特征 LLM (加载, 5.5GB) → 计算 → 卸载
3. 编码器 (常驻, 4GB) → 推理
4. [仅冲突时] Binoculars (加载, 5.5GB) → 计算 → 卸载

峰值 VRAM: ~10.5GB (Language Router + 统计 LLM + 编码器)
```

### 8.2 训练时 VRAM 分配

```
编码器 LoRA 微调:
  - 模型权重 (FP16): ~4GB
  - LoRA 参数: ~0.1GB
  - 梯度 + 优化器: ~3GB
  - 激活值 (batch_size=16): ~3GB
  - 总计: ~10GB (含余量)
  
  ✅ 在 12GB 内可行
```

### 8.3 模型换入换出策略

```python
class ModelManager:
    """管理 GPU 上的模型加载/卸载"""
    
    def __init__(self, max_vram_gb: float = 11.0):
        self.max_vram = max_vram_gb * 1024 ** 3
        self.loaded_models: dict[str, Any] = {}
    
    def load(self, name: str) -> Any:
        if name in self.loaded_models:
            return self.loaded_models[name]
        
        # 检查 VRAM 是否足够
        current_usage = torch.cuda.memory_allocated()
        model_size = self._estimate_size(name)
        
        if current_usage + model_size > self.max_vram:
            # 卸载最近最少使用的模型
            self._evict_lru()
        
        model = self._load_model(name)
        self.loaded_models[name] = model
        return model
    
    def unload(self, name: str):
        if name in self.loaded_models:
            del self.loaded_models[name]
            torch.cuda.empty_cache()
            gc.collect()
```

---

## 9. 项目结构

```
AIGC_Detector/
├── pyproject.toml              # 项目元数据 + 依赖 (uv 管理)
├── uv.lock                     # 锁定文件
├── .python-version             # Python 版本 (3.11)
├── .env                        # 环境变量 (HF_TOKEN 等)
├── .env.example                # .env 模板 (不含敏感值)
├── .gitignore                  # 含 .env, .venv, dataset/, models/, __pycache__/
│
├── DESIGN.md                   # 本设计文档
├── DEVPLAN.md                  # 开发计划
├── notes.md                    # 原始思路 (历史参考)
│
├── src/
│   └── aigc_detector/
│       ├── __init__.py
│       ├── config.py           # 配置管理 (Pydantic Settings)
│       │
│       ├── api/                # Web API 层
│       │   ├── __init__.py
│       │   ├── main.py         # FastAPI app + lifespan
│       │   ├── routes.py       # 路由定义
│       │   ├── schemas.py      # Pydantic 请求/响应模型
│       │   └── middleware.py   # 限流、日志、错误处理
│       │
│       ├── detection/          # 检测引擎
│       │   ├── __init__.py
│       │   ├── pipeline.py     # 级联管线编排
│       │   ├── language.py     # 语言检测路由
│       │   ├── statistical.py  # 统计特征提取
│       │   ├── encoder.py      # 编码器分类 (DeBERTa/RoBERTa)
│       │   ├── binoculars.py   # 零样本检测
│       │   └── ensemble.py     # 集成聚合
│       │
│       ├── data/               # 数据管线
│       │   ├── __init__.py
│       │   ├── generator.py    # AI 文本生成
│       │   ├── crawler.py      # 人类文本爬取
│       │   ├── mixer.py        # 混合文本构造
│       │   ├── processor.py    # 清洗、过滤、格式化
│       │   └── splitter.py     # 数据集划分
│       │
│       ├── training/           # 模型训练
│       │   ├── __init__.py
│       │   ├── trainer.py      # LoRA 微调训练器
│       │   ├── evaluator.py    # 评估指标 (ROC-AUC, F1, etc.)
│       │   └── calibration.py  # 阈值校准
│       │
│       ├── models/             # 模型管理
│       │   ├── __init__.py
│       │   ├── manager.py      # 模型加载/卸载/VRAM 管理
│       │   └── registry.py     # 模型注册表 (路径、配置)
│       │
│       └── utils/              # 通用工具
│           ├── __init__.py
│           ├── text.py         # 文本处理 (分句、清洗)
│           └── logging.py      # 结构化日志
│
├── static/                     # 前端静态文件
│   ├── index.html
│   └── style.css
│
├── scripts/                    # 独立脚本
│   ├── download_models.py      # 模型下载脚本
│   ├── generate_dataset.py     # 数据集生成入口
│   └── train.py                # 训练入口
│
├── tests/                      # 测试
│   ├── __init__.py
│   ├── test_statistical.py
│   ├── test_encoder.py
│   ├── test_pipeline.py
│   └── test_api.py
│
├── configs/                    # 配置文件
│   ├── models.yaml             # 模型注册表
│   └── training.yaml           # 训练超参数
│
├── dataset/                    # 数据目录 (gitignore)
│   ├── raw/                    # 原始数据
│   │   ├── human/
│   │   └── ai/
│   ├── processed/              # 处理后数据
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   └── test.jsonl
│   └── meta.jsonl              # 数据集元信息
│
└── models/                     # 模型权重 (gitignore)
    ├── mistral7b-gptq/         # 已有
    ├── qwen2.5-7b/             # 待下载
    ├── encoder-zh/             # 微调后的中文编码器
    └── encoder-en/             # 微调后的英文编码器
```

---

## 10. 依赖管理

### 10.1 uv 配置

```toml
# pyproject.toml

[project]
name = "aigc-detector"
version = "0.1.0"
description = "Bilingual AI-generated text detection system"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    # Web
    "fastapi>=0.110",
    "uvicorn[standard]>=0.30",
    "slowapi>=0.1.9",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    
    # ML Core
    "transformers>=4.40",
    "tokenizers>=0.19",
    "datasets>=2.18",
    "accelerate>=0.28",
    "peft>=0.10",          # LoRA
    "bitsandbytes>=0.43",  # 量化
    "safetensors>=0.4",
    "sentencepiece>=0.2",
    
    # PyTorch (通过 index 安装)
    "torch>=2.2",
    "torchvision>=0.17",
    
    # Data
    "pandas>=2.0",
    "scikit-learn>=1.4",
    "nltk>=3.8",
    "beautifulsoup4>=4.12",
    "httpx>=0.27",
    
    # Utilities
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "rich>=13.0",         # 日志美化
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",        # 测试用 async client
    "ruff>=0.3",          # Linter + Formatter
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.3",
]

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = [
    { index = "pytorch-cu124" },
]
torchvision = [
    { index = "pytorch-cu124" },
]

[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### 10.2 环境初始化

```bash
# 安装 uv (如未安装)
pip install uv

# 创建项目
uv init
uv python pin 3.11

# 安装依赖
uv sync

# 开发模式
uv sync --dev

# 运行
uv run python scripts/download_models.py
uv run uvicorn src.aigc_detector.api.main:app --host 0.0.0.0 --port 8000
uv run pytest
```

---

## 11. 安全与运维

### 11.1 安全修复 (立即)

1. **`.env` 加入 `.gitignore`** — 当前 API 密钥已暴露
2. **轮换所有暴露的密钥** — OpenAI key 和 HF token
3. **创建 `.env.example`** — 仅含变量名模板

### 11.2 日志

```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    handler = RotatingFileHandler(
        "logs/aigc_detector.log",
        maxBytes=10_000_000,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger("aigc_detector")
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
```

### 11.3 监控端点

- `GET /api/v1/health` — 健康检查
- `GET /api/v1/metrics` — Prometheus 指标 (队列深度、延迟、GPU 使用率)

### 11.4 错误处理

```python
@app.exception_handler(torch.cuda.OutOfMemoryError)
async def oom_handler(request, exc):
    torch.cuda.empty_cache()
    gc.collect()
    logger.error(f"CUDA OOM: {exc}")
    return JSONResponse(
        status_code=503,
        content={"error": "GPU memory exhausted. Please retry."}
    )
```
