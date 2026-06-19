# 中英文AI生成文本检测系统 软件说明书

**软件名称：** 中英文AI生成文本检测系统（AIGC Detector）  
**版本号：** V1.0  
**开发完成日期：** 2026年6月  

---

## 一、软件概述

### 1.1 软件简介

中英文AI生成文本检测系统（以下简称"本系统"）是一款基于多模型集成架构的AI生成内容（AIGC）检测平台。系统通过统计语言模型特征、语言学文体特征、编码器语义特征和Binoculars交叉困惑度四个维度的融合分析，对输入文本进行AI生成概率评估，支持中英文双语检测。

### 1.2 应用场景

- **内容审核平台**：大规模用户生成内容（UGC）的AI文本筛查
- **学术诚信审查**：学术论文、作业的AI代写检测
- **新闻媒体验证**：新闻稿件来源真实性鉴别
- **企业合规审查**：内部文档、报告的AI生成比例评估

### 1.3 技术定位

| 维度 | 说明 |
|---|---|
| 检测语言 | 中文、英文（自动识别） |
| 检测维度 | 4轴融合（统计 + 语言学 + 编码器 + Binoculars） |
| 部署方式 | 本地GPU部署（单卡12GB显存） |
| 服务接口 | RESTful API + Web UI |
| 文件支持 | 纯文本粘贴 + PDF/TXT/MD文件上传 |

---

## 二、系统架构

### 2.1 总体架构

![系统总体架构](diagrams/01-system-architecture.png)

系统采用四阶段级联检测管线，每个阶段提供独立的AI生成概率评估，最终通过加权集成输出综合判定。

```
输入文本
    │
    ▼
┌─────────────────┐
│  语言检测路由器   │  → 判定中文/英文
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│Stage 1 │ │Stage 1b  │
│统计特征 │ │语言学特征 │  （并行执行，共享token对数概率）
└───┬────┘ └────┬─────┘
    │           │
    ▼           ▼
┌──────────────────┐
│  Stage 2         │
│  编码器分类器     │  → LoRA微调的RoBERTa/DeBERTa
└────────┬─────────┘
         │
    ┌────┴────┐
    │ 阶段一致? │
    └────┬────┘
     否  │  是(ZH除外)
    ┌────┘  └────┐
    ▼             ▼
┌────────┐  ┌──────────┐
│Stage 3 │  │ 直接集成  │
│Binoculars│ │ 输出结果  │
└───┬────┘  └──────────┘
    │
    ▼
┌──────────────────┐
│  加权集成 + 阈值   │  → 最终判定
└──────────────────┘
```

### 2.2 核心模块说明

#### 2.2.1 统计特征提取器（Stage 1）

使用参考语言模型（英文：GPT-2-XL，中文：Wenzhong-GPT2-110M）计算输入文本的以下特征：

| 特征 | 说明 |
|---|---|
| 困惑度（Perplexity） | 文本对参考模型的"意外程度"，AI文本通常更低 |
| 平均熵（Avg Entropy） | token预测分布的平均不确定性 |
| 熵标准差（Std Entropy） | 不确定性波动，AI文本波动更小 |
| 爆发度（Burstiness） | 熵的标准差/均值比，人类文本更高 |
| 最大熵（Max Entropy） | 最高不确定性token的熵值 |
| 最小熵（Min Entropy） | 最低不确定性token的熵值 |

特征通过XGBoost分类器转化为AI生成概率（p_ai）。

#### 2.2.2 语言学文体检测器（Stage 1b）

纯CPU运行的14维文体特征引擎，分三个层级：

**微观层（句子级，9个特征）：**
- 句长爆发度/变异系数/基尼系数（M1-M3）
- 句首token句法重复率（M4，Jaccard相似度）
- Token对数概率偏度/高频比例（M5-M6，复用Stage 1缓存）
- 模糊限制语密度（M7，中英文双语词汇表）
- 话语模板标记密度（M8）
- 标点风格评分（M9）

**中观层（段落级，2个特征）：**
- 段落长度方差（S1）
- 段落模板评分（S2，检测"背景→方法→结果→结论"模板）

**宏观层（全文级，3个特征）：**
- 词汇多样性MTLD（D1）
- 作者立场密度（D2）
- 可读性指数（D3）

#### 2.2.3 编码器分类器（Stage 2）

基于预训练语言模型的LoRA微调分类器：
- 英文：microsoft/deberta-v3-large + LoRA (r=16)
- 中文：hfl/chinese-roberta-wwm-ext-large + LoRA (r=16)

输出二分类概率（AI生成 vs 人类撰写）。

#### 2.2.4 Binoculars检测器（Stage 3）

零样本交叉困惑度检测，使用观察者模型和表演者模型的困惑度比值：
- 英文：tiiuae/falcon-7b + falcon-7b-instruct
- 中文：Qwen/Qwen2-7B + Qwen2-7B-Instruct

模型以4-bit量化加载，单卡显存占用约9GB。

#### 2.2.5 加权集成器

将各阶段结果按语言特异性权重融合：

```python
# 英文权重（语言学主导）
en_weights = {"linguistic": 0.85, "statistical": 0.15, "encoder": 0.0, "binoculars": 0.0}

# 中文权重（编码器主导）
zh_weights = {"linguistic": 0.10, "statistical": 0.10, "encoder": 0.60, "binoculars": 0.20}
```

权重系统支持动态归一化：当某些阶段不可用时，自动调整剩余权重至总和为1.0。

### 2.3 语言特异化设计

![检测管线流程](diagrams/02-pipeline-flow.png)

系统对中文和英文采用不同的决策路径：

| 决策点 | 英文 | 中文 |
|---|---|---|
| Stage 1早期退出 | 置信度>0.99时允许 | **禁止**（防止误判正式文本） |
| Stage 1&2一致时跳过Binoculars | 是 | **否**（始终运行Stage 3） |
| 仲裁块 | 无 | 统计判"人类"但编码器p_ai≥0.35时，使用编码器单独决策 |
| 决策阈值 | 0.50 | 0.47 |

---

## 三、运行环境

### 3.1 硬件要求

| 配置 | 最低要求 | 推荐 |
|---|---|---|
| GPU | NVIDIA RTX 3060 12GB | NVIDIA RTX 4090 24GB |
| 内存 | 16GB | 32GB |
| 存储 | 60GB（模型缓存） | 100GB SSD |

### 3.2 软件环境

| 组件 | 版本 |
|---|---|
| Python | ≥ 3.12 |
| CUDA | ≥ 12.0 |
| PyTorch | ≥ 2.4 |
| transformers | ≥ 4.46 |
| FastAPI | ≥ 0.115 |
| bitsandbytes | ≥ 0.43 |

### 3.3 模型依赖

| 用途 | 模型 | 大小 |
|---|---|---|
| 英文统计特征 | GPT-2-XL | 6GB |
| 中文统计特征 | Wenzhong-GPT2-110M | 0.5GB |
| 英文编码器 | DeBERTa-v3-large + LoRA | 1.5GB |
| 中文编码器 | Chinese-RoBERTa-wwm-ext-large + LoRA | 1.3GB |
| 英文Binoculars | Falcon-7B + Falcon-7B-Instruct | 28GB |
| 中文Binoculars | Qwen2-7B + Qwen2-7B-Instruct | 28GB |
| 语言检测 | XLM-RoBERTa-base | 1GB |

模型在首次启动时自动下载（后台线程），支持断点续传。

---

## 四、功能说明

### 4.1 API 接口

#### 4.1.1 文本检测

```
POST /api/v1/detect
Content-Type: application/json

{
  "text": "需要检测的文本内容",
  "include_segments": true,
  "include_diagnostics": false
}
```

#### 4.1.2 文件检测

```
POST /api/v1/detect/file
Content-Type: multipart/form-data

file: [PDF/TXT/MD文件]
include_segments: true
```

支持PDF文件（PyMuPDF + pypdf双引擎提取）、TXT、Markdown格式。

#### 4.1.3 健康检查

```
GET /api/v1/health
```

### 4.2 Web UI

浏览器访问 `http://localhost:8000` 提供双模式界面：
- **文本粘贴模式**：直接输入文本进行检测
- **文件上传模式**：拖拽上传PDF/TXT/MD文件

界面显示：综合判定（AI生成/人类撰写）、置信度、各阶段分数、分段检测结果。

### 4.3 分段检测

对长文本按句子边界分割为最多8个段落，每个段落独立检测，展示文本中AI生成概率的空间分布。

---

## 五、使用说明

### 5.1 安装

```bash
# 克隆代码库
git clone <repository_url>
cd AIGC_Detector

# 安装依赖
uv sync
```

### 5.2 启动服务

```bash
# 启动API服务（默认端口8000）
uv run uvicorn src.aigc_detector.api.main:app --host 0.0.0.0 --port 8000
```

首次启动时，系统会：
1. 加载已缓存的模型到GPU
2. 对未缓存的Binoculars模型启动后台下载线程
3. 服务以降级模式运行（不含Binoculars），下载完成后自动激活

### 5.3 预下载模型（可选）

```bash
# 检查缓存状态
uv run python scripts/prefetch_binoculars.py --check

# 预下载所有Binoculars模型
uv run python scripts/prefetch_binoculars.py
```

### 5.4 运行测试

```bash
uv run pytest tests/ -q
```

### 5.5 模型训练

```bash
# 训练编码器LoRA
uv run python scripts/train_encoder.py --lang zh

# 训练统计分类器
uv run python scripts/train_statistical.py --lang zh

# 训练语言学分类器
uv run python scripts/train_linguistic.py --lang zh --features <features.jsonl>
```

---

## 六、性能指标

### 6.1 检测准确率

| 数据集 | 语言 | 准确率 | AI召回率 | 人类召回率 |
|---|---|---|---|---|
| HC3-Chinese | 中文 | 97.7% | 99.9% | 94.6% |
| Defactify EN | 英文 | 90.4% | 88.2% | 92.3% |

### 6.2 领域自适应

针对现代LLM（GPT-4/Claude）中文教材内容的检测改进：

| 指标 | 优化前 | 优化后 |
|---|---|---|
| 教材章节p_ai | 0.0096 | 0.9980 |
| 教材召回率 | 27.3% | 100.0% |
| 总体准确率 | 95.2% | 97.7% |

### 6.3 推理延迟

| 场景 | 延迟 |
|---|---|
| 统计+语言学（Stage 1+1b） | ~200ms |
| + 编码器（Stage 2） | ~500ms |
| + Binoculars（Stage 3） | ~2000ms |
| 完整四阶段 | ~2500ms |

---

## 七、项目结构

![模块结构图](diagrams/04-module-structure.png)

```
AIGC_Detector/
├── src/aigc_detector/
│   ├── api/              — FastAPI路由、中间件、Schema
│   ├── detection/        — 检测管道（统计/编码器/Binoculars/集成/语言学）
│   ├── training/         — 模型训练、评估、校准
│   ├── data/             — 数据处理、爬取、生成、分割
│   ├── models/           — 模型管理、注册表
│   ├── utils/            — 日志、文本工具、HF缓存
│   └── config.py         — 配置加载
├── configs/              — YAML配置文件
├── scripts/              — 训练、评估、数据生成脚本
├── tests/                — pytest测试（263个用例）
├── static/               — Web前端（HTML/CSS）
├── docs/                 — 文档
├── qa/                   — QA检查清单
└── pyproject.toml        — 项目配置
```

---

## 八、版权声明

本软件由开发团队独立设计和实现，包含以下原创技术：

1. 语言感知四阶段级联检测管线
2. 14维双语语言学文体特征引擎
3. 跨阶段token对数概率缓存复用
4. 语言特异性集成权重与仲裁逻辑
5. 基于过采样的持续学习领域自适应
6. 后台可恢复模型下载与降级运行机制
7. LRU驱除VRAM预算管理器

所有源代码均为原创，未使用任何第三方专有代码。
