# Development Guide

## Environment Setup

The project uses [uv](https://docs.astral.sh/uv/) with Python ≥ 3.12.

```bash
# Install dependencies (creates .venv from the lockfile)
uv sync

# Copy env template and fill values as needed
cp .env.example .env

# Verify CUDA is visible (GPU is the default device)
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Environment variables (`.env.example`)

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | empty | Optional; generation scripts (not the detector). |
| `HF_TOKEN` | empty | Hugging Face access; required for license-gated `falcon-7b` (EN Binoculars). |
| `MODEL_DIR` | `models` | Local artifacts root (classifiers, LoRA adapters, `calibration/`). |
| `DATASET_DIR` | `dataset` | Training/evaluation data. |
| `LOG_DIR` | `logs` | Service log output (sanitized). |
| `DEVICE` | `cuda` | Torch device; `cpu` works for smoke tests only. |
| `MAX_VRAM_GB` | `11.0` | ModelManager VRAM budget (12 GB envelope). |

## Running

```bash
uv run uvicorn aigc_detector.api.main:app --host 0.0.0.0 --port 8000
# or equivalently:
uv run aigc-detector serve
```

WebUI at `http://localhost:8000/`, API docs at `/docs`. First-run model
bootstrap: `scripts/download_models.py` (base models) and
`scripts/prefetch_binoculars.py` (optional 7B pairs) — see README.

## Testing

```bash
uv run pytest                 # full suite (310 tests, no GPU required)
uv run pytest tests/test_api.py -k health   # focused run
```

The suite mocks all GPU operations; tests run on CPU-only machines.

### Adding a test

Follow the stub pattern used across `tests/` (see `tests/test_api.py` for
the canonical example): build the FastAPI app with `create_app()`, replace
`app.state.pipeline` (and `app.state.model_manager`) with `MagicMock` or a
small fake object exposing `detect()` returning the fields the route
reads, then drive it with `fastapi.testclient.TestClient`. Detection-stage
tests similarly use fake extractors/classifiers (dict-returning stubs)
instead of real model weights — keep new tests GPU-free so CI stays fast.
`pytest.ini_options` sets `asyncio_mode = "auto"`; async tests need no marker.

## Linting

```bash
uv run ruff check src scripts tests   # lint (E/F/I/N/W/UP, line length 120)
uv run ruff format --check src        # formatting (double quotes, spaces)
```

## Building the Wheel

```bash
uv build        # dist/aigc_detector-0.1.0-py3-none-any.whl
```

The wheel is **self-contained**: `static/` and `configs/` are
force-included into `aigc_detector/` (see `[tool.hatch.build]` in
`pyproject.toml`), so installed layouts resolve assets without the repo.

### Clean-venv doctor verification

After building, verify packaging from a clean environment (no repo
checkout on `PYTHONPATH`):

```bash
uv run aigc-detector doctor
```

`doctor` checks that static assets, the model registry, and calibration
artifacts resolve from the **installed package layout** — it reports
`[OK] static`, `[OK] model registry`, and the calibration artifact paths.
On a dev checkout it falls back to the repo layout and says so.

## Release Process

1. All changes merged; `uv run pytest` and `uv run ruff check` green.
2. Run the machine-checkable release gate:
   `pwsh scripts/verify_release.ps1 -Batch v0.1` — 10 checks (LICENSE,
   README entry, wheel self-containment, CLI doctor, calibration
   artifacts, model pins, full test suite, …); all must PASS.
3. Update `CHANGELOG.md`: move items from `[Unreleased]` into a dated
   version section.
4. Tag: `git tag v0.1.0 && git push origin v0.1.0`.
5. For container distribution, build per `docs/deploy-docker.md`.

## Project Layout

See `AGENTS.md` for the directory map; `docs/architecture.md` for the
detection pipeline design; `DETECTOR_NOTES_*.md` for the experimental
lab notebook behind every threshold and weight decision.
