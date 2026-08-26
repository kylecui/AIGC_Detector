# AIGC_Detector

## Overview

中英文 AI 生成文本检测系统，通过统计模型、编码器模型和 Binoculars 方法的多模型集成提供检测能力。本系统的定位是**可核查的检测仪器与研究平台**：每个判定附带分段证据、置信度与语域提示，已知盲区主动声明——我们不承诺高准确率，承诺的是判定的可辩护性（见 [能力边界与使用指引](docs/capability-statement.md)）。

## Core Features

1. **多模型集成检测** — 统计/编码器/Binoculars 三路投票，综合评分输出
2. **中英文双语支持** — 自动语言检测，针对中英文分别优化检测策略
3. **FastAPI REST API** — 高性能异步服务，支持批量检测与限流
4. **模型训练与评估** — 完整的训练、评估、校准流程，支持 LoRA 微调
5. **数据集生成管道** — 自动化数据爬取、生成、分割与混合

## Target Users

- 平台运营方（内容审核与 AIGC 识别）
- 内容审核团队（大规模文本筛查）
- 学术研究者（AIGC 检测方法研究）

## Known Limitations（能力边界与推荐用法）

本系统追求**判定的可核查性**：结果附带分段证据与置信度，已知局限主动声明。以下场景可靠性较低，建议人工复核或结合其他证据使用：

- **中文正式文书**（声明、公告、承诺书、情况说明等固定格式文本）：此类文本的用词和结构高度规范化，无论人写还是AI写都很相似，系统整体判定的可靠性显著下降——既可能漏判AI代写的文书，也可能把格式规整的人工文书误判为AI。请结合分段证据（`segment_highlights` 字段）人工复核，重要决策建议采用语料级检测或过程证据。
- **英文法律/专利类文本**：学术研究（arXiv:2607.13044）表明此类受严格格式约束的英文文本上，主流检测方法普遍不可靠，本系统同样将其列为已知盲区，建议人工复核。
- **过短文本**（低于系统最低字符要求）不适用单文档判定。

**使用原则**：检测分数是辅助参考，不是定性证据；在上述盲区场景中，请以人工复核为准。完整的能力正面清单、盲区边界、审核决策流程与FAQ见 **[docs/capability-statement.md](docs/capability-statement.md)**；实测数字与校准报告见 [reports/](reports/)。正式文书语域的检测结果会自动附带警示说明（`caveat` 字段）与分段证据（`segment_highlights` 字段）。

## Quick Start

```bash
# 安装依赖
uv sync

# 启动 API 服务（首次启动会从 HuggingFace 下载基础模型，见下方 Model Bootstrap）
uv run uvicorn aigc_detector.api.main:app --host 0.0.0.0 --port 8000

# 运行测试
uv run pytest
```

服务就绪后：WebUI 在 `http://localhost:8000/`，交互式 API 文档在 `/docs`，健康检查在 `/api/v1/health`（Binoculars 大模型对约 14GB/语言会在后台断点续传下载，就绪前服务以降级模式运行）。

### Model Bootstrap

完整部署需要本地模型文件（LoRA 适配器与校准工件已随仓库分发，基础模型需下载）：

```bash
# 下载基础模型（gpt2-xl / Wenzhong-GPT2 / DeBERTa-v3 / Chinese-RoBERTa / XLM-RoBERTa，约 4-8 GB）
uv run python scripts/download_models.py

# （可选，检测能力增强）预取 Binoculars 观察者/表演者模型对（falcon-7b 系 / Qwen2-7B 系，约 28 GB）
uv run python scripts/prefetch_binoculars.py
```

不运行任何下载脚本也可启动：基础模型在首次请求时自动按需下载；显存预算按 RTX 3060 12GB 设计（4-bit 量化 + LRU 逐出，`src/aigc_detector/models/manager.py`）。

### License Note

本项目代码以 Apache-2.0 发布（见 [LICENSE](LICENSE)）。依赖 `pymupdf` 为 AGPL-3.0 许可：若你的分发场景需要规避 AGPL，可卸载它（`uv remove pymupdf`）——PDF 提取会自动回退到 `pypdf`（Apache/BSD 系许可），功能保持可用。

## Project Structure

详见 [AGENTS.md](AGENTS.md) 和 [初始化报告](initialization-report.md)。

## Development

- **Python**: >=3.12
- **包管理**: uv
- **测试**: pytest
- **Lint**: ruff
- **配置**: configs/ 下 YAML 文件

## Quality Control

使用 `qa/` 下的检查清单，在合并、发布或交付前进行质量审核。

## MCP Integration

MCP 配置模板见 `mcp/`。
