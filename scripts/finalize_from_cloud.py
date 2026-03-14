"""Wait for cloud training artifacts, download them, evaluate locally, and smoke-test the API.

Usage:
  uv run python scripts/finalize_from_cloud.py
  uv run python scripts/finalize_from_cloud.py --poll-seconds 300 --timeout-minutes 600
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.remote_cmd import run as remote_run


def _remote_artifacts_ready() -> bool:
    check_cmd = (
        "test -f /data/aigc/models/encoder-en/adapter_config.json && "
        "test -f /data/aigc/models/encoder-zh/adapter_config.json && "
        "echo READY || echo WAIT"
    )
    code, out, _ = remote_run(
        check_cmd,
        timeout=20,
    )
    return code == 0 and "READY" in out


def _log(message: str) -> None:
    print(message, flush=True)


def _run_local(cmd: list[str]) -> None:
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _api_smoke_test() -> None:
    detect_script = (
        "import httpx, json; "
        "payload={"
        "'text':'This is a sufficiently long English passage for endpoint smoke testing. "
        "It exists only to verify that the real detection route accepts input, runs the "
        "pipeline, and returns a structured response without crashing during post-training "
        "finalization.', "
        "'models':['all']}; "
        "r=httpx.post('http://127.0.0.1:8000/api/v1/detect', json=payload, timeout=120); "
        "print(r.status_code); "
        "print(r.text); "
        "data=r.json(); "
        "ok=(r.status_code==200 and 'predicted_label' in data and 'detected_language' in data); "
        "raise SystemExit(0 if ok else 1)"
    )
    proc = subprocess.Popen(
        [
            "uv",
            "run",
            "python",
            "-m",
            "uvicorn",
            "aigc_detector.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
    )
    try:
        time.sleep(15)
        _run_local(
            [
                "uv",
                "run",
                "python",
                "-c",
                (
                    "import httpx; "
                    "r=httpx.get('http://127.0.0.1:8000/api/v1/health', timeout=30); "
                    "print(r.status_code, r.text); "
                    "raise SystemExit(0 if r.status_code==200 else 1)"
                ),
            ]
        )
        _run_local(
            [
                "uv",
                "run",
                "python",
                "-c",
                detect_script,
            ]
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize local work after cloud training completes")
    parser.add_argument(
        "--poll-seconds", type=int, default=300, help="Polling interval while waiting for cloud artifacts"
    )
    parser.add_argument("--timeout-minutes", type=int, default=600, help="Maximum wait time for cloud artifacts")
    args = parser.parse_args()

    deadline = time.time() + args.timeout_minutes * 60
    while time.time() < deadline:
        if _remote_artifacts_ready():
            break
        _log("Cloud artifacts not ready yet; waiting...")
        time.sleep(args.poll_seconds)
    else:
        print("Timed out waiting for cloud artifacts.", file=sys.stderr, flush=True)
        return 1

    _log("Cloud artifacts ready. Downloading...")
    _run_local(
        ["uv", "run", "python", "scripts/download_cloud.py", "--remote", "/data/aigc/models", "--local", "models"]
    )

    _log("Running evaluation...")
    _run_local(["uv", "run", "python", "scripts/evaluate.py"])

    _log("Running API smoke test...")
    _api_smoke_test()

    _log("Finalization complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
