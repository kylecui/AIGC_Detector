"""Download encoder base models into models/base for offline local runtime.

Usage:
  uv run python scripts/bootstrap_base_models.py
  uv run python scripts/bootstrap_base_models.py --lang zh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))


BASE_MODELS = {
    "en": {
        "repo_id": "microsoft/deberta-v3-large",
        "local_dir": Path("models/base/deberta-v3-large"),
    },
    "zh": {
        "repo_id": "hfl/chinese-roberta-wwm-ext-large",
        "local_dir": Path("models/base/chinese-roberta-wwm-ext-large"),
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download local encoder base models for offline runtime")
    parser.add_argument("--lang", choices=["en", "zh"], help="Only bootstrap one language")
    args = parser.parse_args()

    console = Console()
    selected = [args.lang] if args.lang else ["en", "zh"]

    for lang in selected:
        cfg = BASE_MODELS[lang]
        local_dir = cfg["local_dir"]
        repo_id = cfg["repo_id"]

        if local_dir.exists() and any(local_dir.iterdir()):
            console.print(f"[yellow]⊘[/yellow] {lang}: already present at {local_dir}")
            continue

        console.print(f"[cyan]↓[/cyan] Downloading {repo_id} -> {local_dir}")
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
        console.print(f"[green]✓[/green] {lang}: ready")

    console.print("[bold green]Base model bootstrap complete.[/bold green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
