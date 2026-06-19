"""HuggingFace Hub cache inspection utilities.

Provides testable functions to check whether a model repo is fully cached,
partially downloaded, or absent — without requiring network access.

All functions accept an optional ``cache_root`` parameter so tests can use
a temporary directory instead of the real HF cache.
"""
from __future__ import annotations

import os
from pathlib import Path


def _default_cache_root() -> Path:
    """Return the HF Hub cache root from environment or default location."""
    return Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ) / "hub"


def _model_dir(repo_id: str, cache_root: Path | None = None) -> Path:
    """Return the cache directory path for a given repo_id."""
    root = cache_root or _default_cache_root()
    folder = repo_id.replace("/", "--")
    return root / f"models--{folder}"


def is_model_cached(repo_id: str, cache_root: Path | None = None) -> bool:
    """Check if a model repo has COMPLETE weights in the HF Hub cache.

    Returns ``False`` if:
    - The cache directory does not exist
    - Any ``.incomplete`` blob files exist (partial download in progress)
    - No snapshot directory contains weight files (``*.safetensors`` or ``*.bin``)

    Returns ``True`` only when the repo directory exists, has no incomplete
    blobs, and at least one snapshot contains weight files.
    """
    model_dir = _model_dir(repo_id, cache_root)
    if not model_dir.exists():
        return False

    # If any .incomplete files exist, the model is not fully downloaded
    blobs_dir = model_dir / "blobs"
    if blobs_dir.exists() and any(blobs_dir.glob("*.incomplete")):
        return False

    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return False
    for snap in snapshots.iterdir():
        if snap.is_dir() and any(snap.iterdir()):
            has_weights = any(snap.rglob("*.safetensors")) or any(
                snap.rglob("*.bin")
            )
            if has_weights:
                return True
    return False


def get_incomplete_size(repo_id: str, cache_root: Path | None = None) -> float:
    """Return the total size of ``.incomplete`` files for a repo in GB.

    Returns ``0.0`` if no incomplete files exist or the repo is not cached.
    """
    blobs_dir = _model_dir(repo_id, cache_root) / "blobs"
    if not blobs_dir.exists():
        return 0.0
    total = sum(
        f.stat().st_size
        for f in blobs_dir.glob("*.incomplete")
        if f.is_file()
    )
    return total / (1024**3)


def get_cache_size(repo_id: str, cache_root: Path | None = None) -> float:
    """Return the total on-disk size of a cached model in GB.

    Includes both complete blobs and ``.incomplete`` files.
    Returns ``0.0`` if the repo is not cached.
    """
    model_dir = _model_dir(repo_id, cache_root)
    if not model_dir.exists():
        return 0.0
    total = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    return total / (1024**3)
