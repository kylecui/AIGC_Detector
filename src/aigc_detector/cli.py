"""Console entry point: `aigc-detector` (installed via [project.scripts]).

Two responsibilities:
1. Serve the API with sensible defaults:  `aigc-detector serve [--host H] [--port P]`
2. Report packaging health:               `aigc-detector doctor`
   (checks static assets + model registry + calibration artifacts resolve
   from the INSTALLED package layout, not the repo checkout)
"""

from __future__ import annotations

import argparse
import sys


def _package_root():
    return __import__("pathlib").Path(__file__).resolve().parent


def _run_doctor() -> int:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = _package_root()
    ok = True

    static = root / "static"
    print(f"[{'OK' if static.is_dir() else 'MISS'}] static assets: {static}")
    ok &= static.is_dir()

    registry = root / "configs" / "models.yaml"
    print(f"[{'OK' if registry.is_file() else 'MISS'}] model registry: {registry}")
    ok &= registry.is_file()

    # repo-layout fallback: static/configs at repo root (dev checkout)
    # root = <pkg>/aigc_detector -> parent=src|site-packages -> parent=repo root
    if not ok:
        repo_root = root.parent.parent
        alt_static = repo_root / "static"
        alt_reg = repo_root / "configs" / "models.yaml"
        if alt_static.is_dir() and alt_reg.is_file():
            print(f"[INFO] repo-layout fallback found at {repo_root} (dev checkout)")
            return 0

    from aigc_detector.config import settings

    calib = settings.model_dir / "calibration"
    t_file = calib / "global_temperature.json"
    f_file = calib / "binoculars_floor.json"
    print(f"[{'OK' if t_file.is_file() else 'WARN'}] calibration/temperature: {t_file}")
    print(f"[{'OK' if f_file.is_file() else 'WARN'}] calibration/floor: {f_file}")
    print(f"[INFO] device={settings.device} max_vram_gb={settings.max_vram_gb}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="aigc-detector",
        description="Bilingual AI-generated text detection system (self-hosted instrument)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="start the API server (uvicorn)")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    sub.add_parser("doctor", help="verify packaging: static/configs/calibration resolution")

    args = ap.parse_args(argv)
    if args.cmd == "doctor":
        return _run_doctor()
    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("aigc_detector.api.main:app", host=args.host, port=args.port)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
