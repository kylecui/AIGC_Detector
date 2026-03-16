# AIGC_Detector Deployment Guide

> **Version**: 1.0  
> **Date**: 2026-03-14  
> **Status**: Ready for local deployment / internal handoff

---

## 1. Scope

This document describes how to deploy and operate **AIGC_Detector** in the current project state.

Current validated scope:

- **Languages**: Chinese + English
- **Interface**: FastAPI Web API + static web page
- **Local runtime**: Windows + RTX 3060 12GB
- **Training path**: local preprocessing + cloud GPU LoRA training

---

## 2. Runtime Architecture

At startup, the service builds a three-stage detection stack:

1. **Language Router**
   - model: `papluca/xlm-roberta-base-language-detection`
   - output: `zh` / `en`

2. **Statistical Detector**
   - EN reference LM: `openai-community/gpt2-xl`
   - ZH reference LM: `IDEA-CCNL/Wenzhong-GPT2-110M`
   - output: perplexity / entropy-based features + classifier prediction

3. **Encoder Detector**
   - EN base: `microsoft/deberta-v3-large`
   - ZH base: `hfl/chinese-roberta-wwm-ext-large`
   - LoRA adapters loaded from:
     - `models/encoder-en/`
     - `models/encoder-zh/`

4. **Zero-shot Binoculars**
   - only used when earlier stages do not exit early

The pipeline loads wrappers at startup and loads heavy weights lazily on first request.

---

## 3. Required Local Directory Layout

The following directories are expected under the project root:

```text
models/
├─ statistical-en/
│  ├─ classifier.joblib
│  ├─ calibration.json
│  └─ eval_*.json
├─ statistical-zh/
│  ├─ classifier.joblib
│  ├─ calibration.json
│  └─ eval_*.json
├─ encoder-en/
│  ├─ adapter_config.json
│  ├─ adapter_model.safetensors
│  ├─ tokenizer_config.json
│  └─ tokenizer files
├─ encoder-zh/
│  ├─ adapter_config.json
│  ├─ adapter_model.safetensors
│  ├─ tokenizer_config.json
│  └─ tokenizer files
├─ base/
│  ├─ deberta-v3-large/
│  └─ chinese-roberta-wwm-ext-large/
└─ eval_reports/
```

### Important

`models/base/` is now the preferred offline base-model source for encoder evaluation/runtime.

If it is absent, the code may fall back to online Hugging Face loading.

---

## 4. Environment Requirements

## 4.1 Python / package management

- Python: **3.12**
- package manager: **uv**

Initialize environment:

```bash
uv sync
```

## 4.2 GPU runtime

Validated local target:

- Windows
- NVIDIA RTX 3060 12GB
- CUDA-compatible PyTorch via uv index in `pyproject.toml`

Check CUDA visibility:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 5. Configuration

Runtime settings are loaded from `src/aigc_detector/config.py`.

Current fields:

- `hf_token`
- `openai_api_key`
- `model_dir` (default `models`)
- `dataset_dir` (default `dataset`)
- `log_dir` (default `logs`)
- `device` (default `cuda`)
- `max_vram_gb` (default `11.0`)

Recommended `.env` example:

```env
HF_TOKEN=
OPENAI_API_KEY=
MODEL_DIR=models
DATASET_DIR=dataset
LOG_DIR=logs
DEVICE=cuda
MAX_VRAM_GB=11.0
```

---

## 6. Start the API

## 6.1 Direct startup

```bash
uv run python -m uvicorn aigc_detector.api.main:app --host 127.0.0.1 --port 8000
```

## 6.2 Health check

```bash
uv run python -c "import httpx; print(httpx.get('http://127.0.0.1:8000/api/v1/health').text)"
```

Expected response shape:

```json
{
  "status": "ok",
  "models_loaded": ["xlm-roberta-lang-detect"],
  "gpu_memory_used_mb": 1060.8,
  "gpu_memory_total_mb": 12287.5,
  "uptime_seconds": 49.0
}
```

## 6.3 Detect endpoint

Request:

```http
POST /api/v1/detect
```

Body:

```json
{
  "text": "A sufficiently long text sample...",
  "models": ["all"]
}
```

Constraints:

- `text` length: **50–10000 chars**
- `models` default: `['all']`

---

## 7. Evaluation

Run the complete bilingual evaluation:

```bash
uv run python scripts/evaluate.py
```

Artifacts are written to:

```text
models/eval_reports/
```

Current validated metrics:

| Detector | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| statistical-en | 0.9933 | 0.9913 | 0.9990 |
| encoder-en | 0.9990 | 0.9987 | 0.9993 |
| statistical-zh | 0.9657 | 0.9688 | 0.9947 |
| encoder-zh | 0.9665 | 0.9704 | 0.9997 |

---

## 8. Cloud Training Path

Cloud training is performed using helper scripts:

- `scripts/remote_cmd.py`
- `scripts/upload_cloud.py`
- `scripts/train_cloud.py`
- `scripts/download_cloud.py`

Notes:

- training was validated on a cloud V100-32GB machine
- cloud training currently uses **fp32** to avoid AMP instability
- downloaded adapters are portable to local runtime

---

## 9. Operational Notes

### 9.1 First request latency

The API uses lazy model loading, so:

- first request may be much slower than warm requests
- `/health` only proves service startup, not all detectors are warm

### 9.2 Model storage policy

Do not rely on Hugging Face online availability in production.

Recommended production policy:

- keep encoder adapters in `models/encoder-*`
- keep encoder base models in `models/base/*`
- keep statistical classifiers in `models/statistical-*`

### 9.3 Secrets

- never commit `.env`
- use `.env.example` or deployment-side secret injection

### 9.4 Windows note

Hugging Face cache may emit a symlink warning on Windows. This is not fatal.

---

## 10. Recommended Deployment Sequence

1. `uv sync`
2. verify CUDA
3. ensure `models/` tree is populated
4. ensure `models/base/` contains EN/ZH base models
5. run `uv run python scripts/evaluate.py`
6. start API
7. verify `/api/v1/health`
8. verify `/api/v1/detect`

---

## 11. Remaining Engineering Gaps

These do not block current internal deployment, but should be improved later:

- add `.env.example`
- add dedicated production startup wrapper
- add model integrity checksums
- add offline model bootstrap script for `models/base/`
- add structured logging / request tracing
