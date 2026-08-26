# API Reference

REST API for the AIGC Detector service (FastAPI). Interactive docs are
served at `/docs` (Swagger) and `/redoc` when the service is running.

- **Base URL**: `http://localhost:8000` (default; see `docs/development.md`)
- **Content type**: `application/json` except where noted
- **Interactive exploration**: WebUI at `/`, OpenAPI at `/docs`
- **Field-level schema authority**: `src/aigc_detector/api/schemas.py`

## Authentication

The service ships as a single-tenant, self-hosted instrument and runs
**without authentication by default** (see `docs/sla-statement.md`). When
API-key auth is configured at deployment, every `/api/v1/*` request must
carry an `X-API-Key` header; missing or invalid keys are rejected with
`401`. Do not expose an unauthenticated instance beyond localhost —
see `docs/deploy-docker.md` for network guidance.

## POST /api/v1/detect

Detect whether submitted text is AI-generated or human-written. Rate
limited to **10 requests/minute per IP**; GPU inference is serialized by
a concurrency semaphore (max 2) with a 120-second queue timeout.

**Request body**

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | required | Text to analyze (50–10,000 characters). |
| `models` | string[] | `["all"]` | Which detection models to use. Default `'all'` runs the full pipeline. |
| `include_segments` | bool | `false` | If true, also return segment-level detection results. |
| `include_diagnostics` | bool | `false` | If true, also return linguistic-stylistic diagnostics (micro/meso/macro scores). |

**Response fields** (all responses from `/detect` and `/detect/file`)

| Field | Type | Description |
|---|---|---|
| `predicted_label` | string | `'AI-generated'` or `'Human-written'`. |
| `confidence` | float | Calibrated confidence in [0, 1] (may be compressed in formal-register texts — see `calibration`). |
| `p_ai` | float | Probability of AI generation. |
| `detected_language` | string | ISO-639 code: `'zh'` or `'en'`. |
| `stages_used` | string[] | Detection stages that actually ran. |
| `breakdown` | object | Per-stage result details. |
| `processing_time_ms` | float | Total processing time. |
| `segments` | object[] | Optional segment-level results (when `include_segments=true`). |
| `segment_highlights` | object \| null | Strongest local AI traces: `{max_p_ai, top_k_segments, n_segments}`. An auxiliary review aid only — it may disagree with the document-level verdict by design (see `docs/capability-statement.md`). |
| `caveat` | object \| null | Register caveat: present when the text hits the formal-document register (声明/公告/承诺书…, or the EN formal register), where overall verdict reliability is reduced. Carries `{code, message, action_guidance}`. |
| `calibration` | object \| null | Present when register-conditioned confidence calibration was applied (formal register only): `{method, register, T, confidence_raw, note}`. Displayed confidence is the calibrated value; verdict and ranking are unchanged. |
| `decision_rule` | object \| null | Present when the register-gated binoculars-floor OR-rule fired: verdict upgraded to AI-generated because the binoculars stage exceeded its cutoff in the formal register. Carries `{rule, cutoff, binoculars_p_ai, note}`. |
| `linguistic_diagnostics` | object \| null | Linguistic-stylistic diagnostics (only when `include_diagnostics=true`). |

```bash
curl -s -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"text": "<at least 50 characters of text...>", "include_segments": true}'
```

## POST /api/v1/detect/file

Upload a file, extract text server-side, and run the same detection
pipeline. Multipart form (`multipart/form-data`); same response schema and
rate limit as `/detect`.

- **Form field**: `file` (required)
- **Query parameters**: `include_segments` (default **true**), `include_diagnostics` (default `false`)
- **Formats**: `.pdf` (PyMuPDF with pypdf fallback), `.txt`, `.md`
- **Limits**: 20 MB max file size; extracted text truncated at 50,000 characters; extracted text must be ≥ 50 characters

```bash
curl -s -X POST "http://localhost:8000/api/v1/detect/file?include_segments=true" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@document.pdf"
```

## GET /api/v1/health

Liveness + status probe. Always cheap (no inference).

| Field | Type | Description |
|---|---|---|
| `status` | string | `"ok"` when the process is serving. |
| `models_loaded` | string[] | Models currently resident (lazy-loaded on first use). |
| `gpu_memory_used_mb` / `gpu_memory_total_mb` | float | CUDA memory usage. |
| `uptime_seconds` | float | Process uptime. |

```bash
curl -s http://localhost:8000/api/v1/health
```

## GET /api/v1/ready

Readiness probe (added in this release batch): returns `200` when the
detection pipeline is initialized and the service can accept detection
requests, `503` while models are still loading (cold start) or when the
service runs in degraded mode (e.g. Binoculars still downloading in the
background). Use this — not `/health` — to gate traffic in orchestrators,
container healthchecks, and load balancers.

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/ready
```

## GET /metrics

Prometheus exposition endpoint (added in this release batch): request
counters and latency statistics for the API in Prometheus text format.
Scrape-compatible with standard Prometheus setups; intended for the
operator-side monitoring described in `docs/sla-statement.md`.

```bash
curl -s http://localhost:8000/metrics | head -20
```

## Error Codes

| Code | When it occurs |
|---|---|
| `401` | API-key auth is configured and the `X-API-Key` header is missing or invalid. |
| `413` | Uploaded file exceeds 20 MB. |
| `415` | Unsupported file type (allowed: `.pdf`, `.txt`, `.md`). |
| `422` | Request validation failed: text shorter than 50 / longer than 10,000 characters; empty file; undecodable text file; PDF with no extractable text (e.g. scanned image — OCR it first); extracted text below the 50-character minimum. |
| `429` | Rate limit exceeded (10 requests/minute per IP). |
| `503` | Detection pipeline not initialized; inference queue timeout (120 s, server busy); or CUDA out-of-memory — retry later or use shorter text. |

## Rate Limiting & Concurrency

All detection endpoints are limited to **10 requests/minute per client IP**
(slowapi). GPU inference runs under a semaphore of 2 concurrent requests;
further requests queue up to 120 seconds before returning `503`.
Formal-register Chinese texts may take an extra ~17–23 s when the
binoculars-floor rule triggers a forced Binoculars run (see
`docs/sla-statement.md` for the measured latency envelope).
