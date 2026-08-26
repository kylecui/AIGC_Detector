# Documentation Index

| Document | Contents |
|---|---|
| [capability-statement.md](capability-statement.md) | Reliability boundaries: where verdicts are dependable, known blind spots (ZH/EN formal registers), reviewer decision flow, FAQ. |
| [sla-statement.md](sla-statement.md) | Operating envelope: single-tenant self-hosted positioning, measured latency/VRAM specs, what v0.1 explicitly is not. |
| [api.md](api.md) | REST API reference: auth, `/api/v1/detect`, `/api/v1/detect/file`, `/api/v1/health`, `/api/v1/ready`, `/metrics`, error codes, rate limits, curl examples. |
| [architecture.md](architecture.md) | Four-stage ensemble cascade, language routing weights, lazy loading + LRU VRAM manager, register gates, calibration, W15 floor rule, artifact layout. |
| [development.md](development.md) | Dev setup (uv, env vars), running tests and ruff, wheel build, clean-venv doctor verification, release process, adding tests. |
| [deploy-docker.md](deploy-docker.md) | GPU container build/run guide, HF cache volume, first-run behavior, known trade-offs. |
| [adr/](adr/) | Architecture decision records (ADR-0001: formal-register encoder retraining deferral). |
| [research/](research/) | Experimental dossier: direction validation, FN-1 root cause, W13 paper drafts and figures. |

Additional project material: patent disclosure (`patent-disclosure.md`),
software copyright registration package (`software-copyright/`), system
diagrams (`diagrams/`, draw.io sources), and the experimental lab
notebook (`../DETECTOR_NOTES_*.md` at repo root). Evaluation reports and
calibration evidence live in [`../reports/`](../reports/).
