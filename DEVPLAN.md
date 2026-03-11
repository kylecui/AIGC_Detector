# AIGC Detector — 开发计划

> **版本**: 1.0  
> **日期**: 2026-03-11  
> **关联文档**: [DESIGN.md](./DESIGN.md)  
> **预估总周期**: 8–10 周（业余时间，每天 2-3 小时）

---

## 总览

```
Phase 0 ──→ Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4
工程基础     数据管线     检测基线     微调检测器    集成 + 部署
(1 周)      (2-3 周)    (1-2 周)    (2 周)       (2 周)
```

每个 Phase 结束时有明确的 **验收标准 (Exit Criteria)**，通过后才进入下一 Phase。

---

## Phase 0: 工程基础 (第 1 周)

**目标**: 建立项目骨架、依赖管理、安全修复。此 Phase 完成后，所有后续开发在干净的工程环境中进行。

### 任务

| # | 任务 | 优先级 | 预估 |
|---|------|--------|------|
| 0.1 | 初始化 uv 项目: `pyproject.toml`, `.python-version` (3.11), `uv sync` | P0 | 1h |
| 0.2 | 验证 PyTorch CUDA 安装: `torch.cuda.is_available()`, CUDA 版本匹配 | P0 | 1h |
| 0.3 | 创建项目目录结构 (参见 DESIGN.md §9) | P0 | 1h |
| 0.4 | 安全修复: `.env` 加入 `.gitignore`, 创建 `.env.example`, 轮换暴露的密钥 | P0 | 30min |
| 0.5 | 配置 Ruff linter/formatter, pytest | P1 | 30min |
| 0.6 | 实现 `src/aigc_detector/config.py` (Pydantic Settings, 从 .env 读取) | P1 | 1h |
| 0.7 | 实现 `src/aigc_detector/models/registry.py` (模型路径注册表, YAML) | P1 | 1h |
| 0.8 | 实现 `scripts/download_models.py` (统一模型下载脚本) | P1 | 2h |
| 0.9 | 清理历史文件: 移除 `test/` 旧 demo, `generate_ai_async.py` 等 | P2 | 30min |

### 验收标准

- [x] `uv sync` 成功，`uv run python -c "import torch; print(torch.cuda.is_available())"` 输出 `True`
- [x] 项目目录结构与 DESIGN.md §9 一致
- [x] `.env` 在 `.gitignore` 中，不被 git 追踪
- [x] `uv run ruff check src/` 无错误
- [x] `uv run pytest tests/ -v` 通过 (即使测试用例少)

---

## Phase 1: 数据管线 (第 2–4 周)

**目标**: 构建完整的数据收集、生成、混合、清洗管线。产出 120K 样本双语数据集。

### 第 1 子阶段: 人类文本收集 (Week 2)

| # | 任务 | 优先级 | 预估 |
|---|------|--------|------|
| 1.1 | 实现 `src/aigc_detector/data/crawler.py` — 改进的新闻爬虫 (中英文) | P0 | 4h |
| 1.2 | 编写 HC3 数据集下载 + 解析脚本 (中文 + 英文) | P0 | 2h |
| 1.3 | 实现 `src/aigc_detector/data/processor.py` — 文本清洗、过滤、去重 | P0 | 3h |
| 1.4 | 实现 `src/aigc_detector/utils/text.py` — 双语分句 (`split_sentences_bilingual`) | P0 | 2h |
| 1.5 | 收集中文人类文本 25K+ 条 | P0 | 持续 |
| 1.6 | 收集英文人类文本 25K+ 条 | P0 | 持续 |

### 第 2 子阶段: AI 文本生成 (Week 3)

| # | 任务 | 优先级 | 预估 |
|---|------|--------|------|
| 1.7 | 实现 `src/aigc_detector/data/generator.py` — 多模型文本生成器 | P0 | 4h |
| 1.8 | 下载所需模型: Qwen2.5-7B, ChatGLM3-6B, Yi-6B-Chat, Llama-3-8B-GPTQ | P0 | 2h (下载) |
| 1.9 | 编写多样化 prompt 模板 (5 领域 × 4 文体 × 中英) | P0 | 3h |
| 1.10 | 为每个模型实现正确的 `apply_chat_template()` 调用 | P0 | 2h |
| 1.11 | 批量生成中文 AI 文本: 5 模型 × 5000 条 = 25K | P0 | ~24h GPU |
| 1.12 | 批量生成英文 AI 文本: 5 模型 × 5000 条 = 25K | P0 | ~24h GPU |

### 第 3 子阶段: 混合文本 + 数据集组装 (Week 4)

| # | 任务 | 优先级 | 预估 |
|---|------|--------|------|
| 1.13 | 实现 `src/aigc_detector/data/mixer.py` — AI Completion 方法 | P0 | 3h |
| 1.14 | 实现 Sentence Insertion 方法 | P1 | 3h |
| 1.15 | 生成混合文本 20K 条 (中英各 10K) | P0 | ~8h GPU |
| 1.16 | 实现 `src/aigc_detector/data/splitter.py` — 训练/验证/测试集划分 (80/10/10) | P0 | 1h |
| 1.17 | 数据集统计分析: 各类别分布、长度分布、模型分布 | P1 | 2h |
| 1.18 | 编写数据管线集成测试 | P1 | 2h |

### 验收标准

- [ ] `dataset/processed/train.jsonl` 存在，包含 96K+ 条记录
- [ ] `dataset/processed/val.jsonl` 存在，包含 12K+ 条记录
- [ ] `dataset/processed/test.jsonl` 存在，包含 12K+ 条记录
- [ ] 数据包含 human / ai / mixed 三类标签
- [ ] 数据包含 zh / en 两种语言
- [ ] AI 文本来源覆盖 5+ 个不同模型
- [ ] 所有文本长度 ≥ 200 字符
- [ ] `uv run pytest tests/test_data.py -v` 通过

---

## Phase 2: 检测基线 (第 5–6 周)

**目标**: 实现统计特征基线和零样本基线。建立评估框架。在测试集上报告初始指标。

### 任务

| # | 任务 | 优先级 | 预估 |
|---|------|--------|------|
| 2.1 | 实现 `src/aigc_detector/detection/language.py` — 语言路由 | P0 | 2h |
| 2.2 | 实现 `src/aigc_detector/detection/statistical.py` — 统计特征提取 | P0 | 4h |
| 2.3 | 实现统计特征 → XGBoost/LR 分类器 (基线) | P0 | 3h |
| 2.4 | 实现 `src/aigc_detector/training/evaluator.py` — 评估框架 (ROC-AUC, F1, Precision, Recall, 混淆矩阵) | P0 | 3h |
| 2.5 | 在测试集上评估统计特征基线 | P0 | 2h |
| 2.6 | 实现 `src/aigc_detector/detection/binoculars.py` — Binoculars 零样本检测 | P1 | 4h |
| 2.7 | 英文 Binoculars 评估 (Falcon pair, 使用论文阈值) | P1 | 2h |
| 2.8 | 中文 Binoculars 实验 (Qwen2 pair, 校准阈值) | P1 | 4h |
| 2.9 | 实现 `src/aigc_detector/training/calibration.py` — 阈值校准工具 | P1 | 2h |
| 2.10 | 编写基线评估报告 (各方法在测试集上的表现) | P1 | 2h |

### 验收标准

- [ ] 统计特征基线在二分类 (human vs. ai) 上 ROC-AUC > 0.80
- [ ] Binoculars 英文在二分类上 ROC-AUC > 0.90 (对标论文)
- [ ] Binoculars 中文有初步 ROC-AUC 指标 (即使低于英文)
- [ ] 评估框架可复现 (相同数据 + 相同阈值 = 相同结果)
- [ ] `uv run pytest tests/test_statistical.py -v` 通过

---

## Phase 3: 微调检测器 (第 7–8 周)

**目标**: LoRA 微调 DeBERTa (英文) 和 chinese-roberta (中文)。显著超越基线。

### 任务

| # | 任务 | 优先级 | 预估 |
|---|------|--------|------|
| 3.1 | 实现 `src/aigc_detector/detection/encoder.py` — 编码器分类器 (推理) | P0 | 3h |
| 3.2 | 实现 `src/aigc_detector/training/trainer.py` — LoRA 微调训练器 | P0 | 4h |
| 3.3 | 实现 `configs/training.yaml` — 训练超参数配置 | P0 | 1h |
| 3.4 | 训练英文 DeBERTa-v3-large + LoRA (binary classification) | P0 | ~4h GPU |
| 3.5 | 训练中文 chinese-roberta-wwm-ext-large + LoRA | P0 | ~4h GPU |
| 3.6 | 超参数调优 (lr, batch_size, epochs, LoRA rank) | P1 | ~8h GPU |
| 3.7 | 在测试集上评估微调编码器 | P0 | 2h |
| 3.8 | 对比分析: 编码器 vs. 统计基线 vs. Binoculars | P0 | 2h |
| 3.9 | 交叉模型泛化测试 (训练用模型 A 生成的数据，测试用模型 B 生成的) | P1 | 4h |
| 3.10 | 编写微调模型评估报告 | P1 | 2h |

### 验收标准

- [ ] 英文 DeBERTa 在二分类上 ROC-AUC > 0.95
- [ ] 中文 RoBERTa 在二分类上 ROC-AUC > 0.93
- [ ] 微调后的编码器显著优于统计特征基线 (ROC-AUC 差 > 0.05)
- [ ] 交叉模型泛化: ROC-AUC > 0.85 (在未见过的生成模型上)
- [ ] 模型保存在 `models/encoder-zh/` 和 `models/encoder-en/`
- [ ] `uv run pytest tests/test_encoder.py -v` 通过

---

## Phase 4: 集成 + Web 部署 (第 9–10 周)

**目标**: 实现级联集成管线、Web API、前端页面。系统可端到端运行。

### 第 1 子阶段: 集成管线 (Week 9)

| # | 任务 | 优先级 | 预估 |
|---|------|--------|------|
| 4.1 | 实现 `src/aigc_detector/detection/pipeline.py` — 级联管线编排 | P0 | 4h |
| 4.2 | 实现 `src/aigc_detector/detection/ensemble.py` — 加权集成聚合 | P0 | 3h |
| 4.3 | 实现 `src/aigc_detector/models/manager.py` — 模型加载/卸载/VRAM 管理 | P0 | 4h |
| 4.4 | 集成管线端到端测试 (输入文本 → 检测结果) | P0 | 3h |
| 4.5 | 集成管线在测试集上的最终评估 | P0 | 2h |

### 第 2 子阶段: Web API + 前端 (Week 10)

| # | 任务 | 优先级 | 预估 |
|---|------|--------|------|
| 4.6 | 实现 `src/aigc_detector/api/main.py` — FastAPI app + lifespan | P0 | 3h |
| 4.7 | 实现 `src/aigc_detector/api/routes.py` — API 路由 (detect, health) | P0 | 2h |
| 4.8 | 实现 `src/aigc_detector/api/schemas.py` — Pydantic 模型 | P0 | 1h |
| 4.9 | 实现 `src/aigc_detector/api/middleware.py` — 限流、错误处理、日志 | P1 | 2h |
| 4.10 | 实现前端页面 `static/index.html` + `static/style.css` | P0 | 4h |
| 4.11 | 端到端 Web 集成测试 (启动服务器 → 前端发请求 → 看结果) | P0 | 2h |
| 4.12 | 编写 API 测试 `tests/test_api.py` | P1 | 2h |
| 4.13 | 性能测试: 单请求延迟、并发吞吐量 | P1 | 2h |
| 4.14 | 编写用户使用说明 | P2 | 2h |

### 鲁棒性评估 (贯穿 Phase 4)

| # | 任务 | 优先级 | 预估 |
|---|------|--------|------|
| 4.15 | 释义攻击测试 (用 LLM 重写 AI 文本后再检测) | P1 | 4h |
| 4.16 | 短文本测试 (<100 tokens) | P1 | 2h |
| 4.17 | 混合文本检测准确率 | P1 | 2h |
| 4.18 | 编写最终评估报告 (含鲁棒性) | P1 | 3h |

### 验收标准

- [ ] `uv run uvicorn src.aigc_detector.api.main:app` 启动成功
- [ ] 浏览器访问 `http://localhost:8000` 显示检测页面
- [ ] 粘贴中文文本 → 返回检测结果 (含中文管线特征)
- [ ] 粘贴英文文本 → 返回检测结果 (含英文管线特征)
- [ ] API 响应时间 < 5 秒 (单请求，含所有阶段)
- [ ] GPU VRAM 峰值 < 12GB
- [ ] `GET /api/v1/health` 返回 200
- [ ] 集成管线在测试集上的 ROC-AUC 优于任何单独子模块
- [ ] `uv run pytest tests/ -v` 全部通过

---

## 风险与缓解

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|----------|
| Binoculars 中文效果差 | 零样本层在中文上不可靠 | 中 | 降级为仅英文使用；中文管线仅用统计+编码器 |
| 数据量不足 (收集慢) | 模型泛化能力差 | 低 | 优先使用 HC3 现成数据集，减少爬取依赖 |
| VRAM 不够 (模型换入换出慢) | 推理延迟高 | 中 | 常驻编码器 + 按需加载 LLM；牺牲 Stage 3 延迟 |
| bitsandbytes Windows 兼容性 | 量化推理失败 | 低-中 | 使用社区 Windows fork 或 GPTQ 替代 BnB |
| 交叉模型泛化差 | 对未见模型生成的文本检测率低 | 中 | 增加训练数据中的模型多样性；集成多种检测方法 |
| 中文分句质量差 | 混合文本构造不准确 | 低 | 使用正则 + 规则结合，覆盖中文标点全集 |

---

## 依赖关系图

```
Phase 0 (工程基础)
  │
  ├──→ Phase 1 (数据管线)
  │      │
  │      ├──→ Phase 2 (检测基线)
  │      │      │
  │      │      ├──→ Phase 3 (微调检测器)
  │      │      │      │
  │      │      │      └──→ Phase 4 (集成 + 部署)
  │      │      │              ↑
  │      │      └──────────────┘ (基线结果作为对比基准)
  │      │
  │      └──→ Phase 4 可以与 Phase 3 并行开始 (API 骨架不依赖微调模型)
  │
  └──→ Phase 4 前端开发可以在 Phase 2 后开始 (Mock API)
```

### 可并行的任务

- Phase 1: 人类文本爬取 (CPU) 可与模型下载 (网络) 并行
- Phase 1: AI 文本生成 (GPU) 可过夜批量运行
- Phase 2: 统计基线 和 Binoculars 可独立开发
- Phase 4: 前端开发 (纯 HTML/JS) 不依赖后端模型

---

## 里程碑总结

| 里程碑 | 目标日期 | 交付物 |
|--------|----------|--------|
| **M0: 项目初始化** | Week 1 | uv 项目、目录结构、安全修复 |
| **M1: 数据集就绪** | Week 4 | 120K 样本双语数据集 (train/val/test) |
| **M2: 基线指标** | Week 6 | 统计+零样本基线 ROC-AUC 报告 |
| **M3: 微调模型** | Week 8 | LoRA 微调编码器 + 评估报告 |
| **M4: V1 发布** | Week 10 | Web API + 前端，端到端可用 |
