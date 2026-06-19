# AIGC_Detector

## Overview

中英文 AI 生成文本检测系统，通过统计模型、编码器模型和 Binoculars 方法的多模型集成，提供高精度的 AIGC 内容识别能力。

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

## Quick Start

```bash
# 安装依赖
uv sync

# 启动 API 服务
uv run python main.py

# 运行测试
uv run pytest
```

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
