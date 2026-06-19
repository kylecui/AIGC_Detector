"""Pre-download Binoculars detector models before starting the service.

Default mode prioritizes RELIABLE RESUME over speed:
  - Standard Python downloader (not hf_transfer) — properly resumes from
    .incomplete files after Ctrl+C or network failure
  - Sequential download (max_workers=1) — each file completes fully before
    the next starts, minimizing restart blast radius
  - Read timeout (30s) — prevents indefinite hangs on stalled connections
  - Per-repo retry with exponential backoff

Use --fast for speed (hf_transfer + concurrent), but note: hf_transfer does
NOT support resume — Ctrl+C will lose all in-progress downloads.

Usage:
    # Check what's missing (no download)
    uv run python scripts/prefetch_binoculars.py --check

    # Download all missing Binoculars models (reliable, resumable)
    uv run python scripts/prefetch_binoculars.py

    # Download only ZH pair
    uv run python scripts/prefetch_binoculars.py --lang zh

    # Fast mode: hf_transfer + concurrent (NO resume on interruption)
    uv run python scripts/prefetch_binoculars.py --fast

    # Use mirror (only if huggingface.co is blocked)
    uv run python scripts/prefetch_binoculars.py --mirror

References:
    - DETECTOR_NOTES_2026-06.md (Binoculars download optimization)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from aigc_detector.utils.hf_cache import (
    get_cache_size,
    get_incomplete_size,
    is_model_cached,
)

console = Console()

BINO_CONFIGS = {
    "en": ("tiiuae/falcon-7b", "tiiuae/falcon-7b-instruct"),
    "zh": ("Qwen/Qwen2-7B", "Qwen/Qwen2-7B-Instruct"),
}

IGNORE_PATTERNS = [
    "*.bin",
    "*.pt",
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "*.gguf",
    "original/*",
    "tf_model*",
    "flax_model*",
    "pytorch_model*",
]


def show_status(langs: list[str]) -> Table:
    """Build a status table showing cached / missing / partial models."""
    table = Table(title="Binoculars Model Cache Status", show_header=True)
    table.add_column("Lang", style="cyan", width=6)
    table.add_column("Repo", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Size (GB)", justify="right", style="dim")
    table.add_column("Resumable", justify="right", style="yellow")

    for lang in langs:
        observer, performer = BINO_CONFIGS[lang]
        for repo_id, role in [(observer, "observer"), (performer, "performer")]:
            if is_model_cached(repo_id):
                size = get_cache_size(repo_id)
                status = "[green]CACHED[/green]"
                partial = ""
            else:
                size = get_cache_size(repo_id)
                partial_size = get_incomplete_size(repo_id)
                status = "[red]MISSING[/red]"
                partial = f"{partial_size:.2f} GB" if partial_size > 0 else ""
            label = f"{repo_id} ({role})"
            table.add_row(lang, label, status, f"{size:.2f}", partial)

    return table


def download_repo(
    repo_id: str,
    use_mirror: bool,
    max_retries: int = 3,
) -> bool:
    """Download a single repo with retry. Returns True on success."""
    from huggingface_hub import snapshot_download

    for attempt in range(1, max_retries + 1):
        try:
            console.print(f"  [cyan]Downloading[/cyan] {repo_id} (attempt {attempt}/{max_retries})...")
            snapshot_download(
                repo_id,
                ignore_patterns=IGNORE_PATTERNS,
                max_workers=1,  # Sequential: each file completes fully
            )
            console.print(f"  [green]OK[/green]    {repo_id}")
            return True
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** (attempt - 1) * 5  # 5s, 10s, 20s
                console.print(
                    f"  [yellow]retry[/yellow] {repo_id} failed ({e}), "
                    f"waiting {wait}s before retry..."
                )
                time.sleep(wait)
            else:
                console.print(f"  [red]FAIL[/red]  {repo_id}: {e}")
                return False
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-download Binoculars detector models (reliable, resumable)."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only show cache status, do not download.",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh", "all"],
        default="all",
        help="Which language pair to download (default: all).",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Use HF mirror (hf-mirror.com). Only if huggingface.co is blocked.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable hf_transfer + concurrent downloads for speed. "
        "WARNING: no resume support — Ctrl+C loses all in-progress downloads.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Max retries per repo on failure (default: 3).",
    )
    args = parser.parse_args()

    langs = ["en", "zh"] if args.lang == "all" else [args.lang]

    # --- Show status ---
    console.print()
    console.print(show_status(langs))
    console.print()

    # --- Check-only mode ---
    if args.check:
        missing = []
        for lang in langs:
            obs, perf = BINO_CONFIGS[lang]
            if not is_model_cached(obs):
                missing.append((lang, obs))
            if not is_model_cached(perf):
                missing.append((lang, perf))
        if missing:
            partial = sum(1 for _, r in missing if get_incomplete_size(r) > 0)
            console.print(
                f"[yellow]{len(missing)} model(s) need downloading.[/yellow] "
                f"({partial} have resumable partial downloads) "
                f"Run without --check to download."
            )
        else:
            console.print("[green]All Binoculars models are cached.[/green]")
        return 0

    # --- Collect repos to download ---
    repos_to_download: list[str] = []
    for lang in langs:
        obs, perf = BINO_CONFIGS[lang]
        for repo_id in (obs, perf):
            if not is_model_cached(repo_id):
                repos_to_download.append(repo_id)

    if not repos_to_download:
        console.print("[green]All requested models are already cached.[/green]")
        return 0

    # --- Configure download mode ---
    # Default: reliable resume (standard downloader, sequential, read timeout)
    # --fast: hf_transfer + concurrent (speed, no resume)
    if args.fast:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        download_mode = "fast (hf_transfer, no resume)"
        warning = "\n  [yellow]WARNING: --fast disables resume. "
        warning += "Ctrl+C will lose in-progress downloads.[/yellow]"
    else:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        download_mode = "reliable (resumable, sequential)"
        warning = ""

    if args.mirror:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        try:
            import huggingface_hub.constants as _const
            _const.ENDPOINT = os.environ["HF_ENDPOINT"]
        except Exception:
            pass
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        download_mode = "mirror (hf-mirror.com, no hf_transfer)"

    # Prevent indefinite hangs: timeout if no data received for 30 seconds
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

    console.print(
        Panel.fit(
            f"[bold]Will download {len(repos_to_download)} model(s)[/bold]\n"
            f"  Mode: {download_mode}\n"
            f"  Retries per repo: {args.retries}\n"
            f"  Read timeout: {os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT', 'default')}s\n"
            f"  Format filter: safetensors-only (skip .bin/.pt/.onnx)"
            f"{warning}",
            title="Download Plan",
            border_style="cyan",
        )
    )
    console.print()

    # --- Download sequentially ---
    t0 = time.perf_counter()
    success: list[str] = []
    failed: list[str] = []

    for i, repo_id in enumerate(repos_to_download, 1):
        console.print(f"[bold]Repo {i}/{len(repos_to_download)}[/bold]")
        ok = download_repo(repo_id, use_mirror=args.mirror, max_retries=args.retries)
        if ok:
            success.append(repo_id)
        else:
            failed.append(repo_id)
        console.print()

    elapsed = time.perf_counter() - t0

    # --- Summary ---
    console.print(
        Panel.fit(
            f"Downloaded: [green]{len(success)}[/green]  "
            f"Failed: [red]{len(failed)}[/red]  "
            f"Time: {elapsed:.0f}s",
            title="Result",
            border_style="green" if not failed else "red",
        )
    )

    # --- Show final status ---
    console.print()
    console.print(show_status(langs))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
