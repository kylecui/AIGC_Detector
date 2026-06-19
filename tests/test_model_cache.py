"""Tests for HuggingFace Hub cache inspection utilities.

Tests cover:
- is_model_cached: complete / incomplete / absent detection
- get_incomplete_size: partial download size measurement
- get_cache_size: total cache size measurement
- Edge cases: missing dirs, empty snapshots, mixed states
- Download lifecycle: not-downloaded -> partial -> complete transitions

All tests use tmp_path to create isolated mock cache structures. No network
access or real HF cache is touched. File sizes are MB-scale for speed.
"""
from __future__ import annotations

from pathlib import Path

from aigc_detector.utils.hf_cache import (
    get_cache_size,
    get_incomplete_size,
    is_model_cached,
)

REPO_ID = "Qwen/Qwen2-7B-Instruct"
MB = 1024 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(
    cache_root: Path,
    repo_id: str,
    snapshot_files: list[str] | None = None,
    incomplete_sizes: list[int] | None = None,
    complete_blobs: list[int] | None = None,
    create_blobs_dir: bool = True,
    create_snapshots_dir: bool = True,
) -> Path:
    """Create a mock HF cache entry for a repo."""
    folder = repo_id.replace("/", "--")
    model_dir = cache_root / f"models--{folder}"

    if create_snapshots_dir:
        snap_dir = model_dir / "snapshots" / "abc123def"
        snap_dir.mkdir(parents=True, exist_ok=True)
        for fname in snapshot_files or []:
            (snap_dir / fname).write_bytes(b"x")

    if create_blobs_dir:
        blobs_dir = model_dir / "blobs"
        blobs_dir.mkdir(parents=True, exist_ok=True)
        for i, size in enumerate(complete_blobs or []):
            (blobs_dir / f"blob_complete_{i}").write_bytes(b"\x00" * size)
        for i, size in enumerate(incomplete_sizes or []):
            (blobs_dir / f"blob_partial_{i}.incomplete").write_bytes(b"\x00" * size)

    return model_dir


# ---------------------------------------------------------------------------
# is_model_cached -- basic states
# ---------------------------------------------------------------------------

class TestIsModelCached:

    def test_returns_false_when_cache_dir_missing(self, tmp_path: Path):
        assert is_model_cached(REPO_ID, cache_root=tmp_path) is False

    def test_returns_false_when_no_snapshots_dir(self, tmp_path: Path):
        _make_repo(tmp_path, REPO_ID, create_snapshots_dir=False)
        assert is_model_cached(REPO_ID, cache_root=tmp_path) is False

    def test_returns_false_when_snapshots_empty(self, tmp_path: Path):
        _make_repo(tmp_path, REPO_ID, snapshot_files=[])
        assert is_model_cached(REPO_ID, cache_root=tmp_path) is False

    def test_returns_false_when_only_config_files(self, tmp_path: Path):
        _make_repo(
            tmp_path, REPO_ID,
            snapshot_files=["config.json", "tokenizer.json"],
        )
        assert is_model_cached(REPO_ID, cache_root=tmp_path) is False

    def test_returns_true_when_safetensors_exist(self, tmp_path: Path):
        _make_repo(
            tmp_path, REPO_ID,
            snapshot_files=["config.json", "model-00001-of-00004.safetensors"],
        )
        assert is_model_cached(REPO_ID, cache_root=tmp_path) is True

    def test_returns_true_when_bin_files_exist(self, tmp_path: Path):
        _make_repo(
            tmp_path, REPO_ID,
            snapshot_files=["pytorch_model.bin"],
        )
        assert is_model_cached(REPO_ID, cache_root=tmp_path) is True

    def test_returns_true_with_single_safetensors(self, tmp_path: Path):
        _make_repo(tmp_path, REPO_ID, snapshot_files=["model.safetensors"])
        assert is_model_cached(REPO_ID, cache_root=tmp_path) is True


# ---------------------------------------------------------------------------
# is_model_cached -- .incomplete detection (the critical bug fix)
# ---------------------------------------------------------------------------

class TestIsModelCachedIncomplete:
    """Tests for .incomplete file detection.

    Bug context: snapshot symlinks existed but underlying blobs were
    .incomplete. Old code reported CACHED, causing download to be skipped.
    """

    def test_returns_false_when_incomplete_blobs_exist(self, tmp_path: Path):
        _make_repo(
            tmp_path, REPO_ID,
            snapshot_files=["model.safetensors"],
            incomplete_sizes=[5 * MB],
        )
        assert is_model_cached(REPO_ID, cache_root=tmp_path) is False

    def test_returns_false_with_multiple_incomplete(self, tmp_path: Path):
        _make_repo(
            tmp_path, REPO_ID,
            snapshot_files=["model.safetensors"],
            incomplete_sizes=[10 * MB, 5 * MB, 2 * MB],
        )
        assert is_model_cached(REPO_ID, cache_root=tmp_path) is False

    def test_returns_true_when_complete_blobs_no_incomplete(self, tmp_path: Path):
        _make_repo(
            tmp_path, REPO_ID,
            snapshot_files=["model.safetensors"],
            complete_blobs=[4 * MB],
        )
        assert is_model_cached(REPO_ID, cache_root=tmp_path) is True

    def test_returns_false_when_mixed_complete_and_incomplete(self, tmp_path: Path):
        _make_repo(
            tmp_path, REPO_ID,
            snapshot_files=["model.safetensors"],
            complete_blobs=[2 * MB],
            incomplete_sizes=[1 * MB],
        )
        assert is_model_cached(REPO_ID, cache_root=tmp_path) is False

    def test_zero_byte_incomplete_still_blocks(self, tmp_path: Path):
        _make_repo(
            tmp_path, REPO_ID,
            snapshot_files=["model.safetensors"],
            incomplete_sizes=[0],
        )
        assert is_model_cached(REPO_ID, cache_root=tmp_path) is False


# ---------------------------------------------------------------------------
# is_model_cached -- edge cases
# ---------------------------------------------------------------------------

class TestIsModelCachedEdgeCases:

    def test_repo_id_with_dashes(self, tmp_path: Path):
        _make_repo(
            tmp_path, "tiiuae/falcon-7b-instruct",
            snapshot_files=["model.safetensors"],
        )
        assert is_model_cached("tiiuae/falcon-7b-instruct", cache_root=tmp_path) is True

    def test_multiple_snapshots_one_has_weights(self, tmp_path: Path):
        model_dir = _make_repo(
            tmp_path, REPO_ID,
            snapshot_files=["config.json"],
        )
        snap2 = model_dir / "snapshots" / "def456"
        snap2.mkdir(parents=True, exist_ok=True)
        (snap2 / "model.safetensors").write_bytes(b"x")
        assert is_model_cached(REPO_ID, cache_root=tmp_path) is True

    def test_multiple_repos_no_interference(self, tmp_path: Path):
        _make_repo(tmp_path, "tiiuae/falcon-7b", snapshot_files=["model.safetensors"])
        _make_repo(tmp_path, "Qwen/Qwen2-7B", snapshot_files=[])
        assert is_model_cached("tiiuae/falcon-7b", cache_root=tmp_path) is True
        assert is_model_cached("Qwen/Qwen2-7B", cache_root=tmp_path) is False


# ---------------------------------------------------------------------------
# get_incomplete_size
# ---------------------------------------------------------------------------

class TestGetIncompleteSize:

    def test_zero_when_no_cache(self, tmp_path: Path):
        assert get_incomplete_size(REPO_ID, cache_root=tmp_path) == 0.0

    def test_zero_when_no_blobs_dir(self, tmp_path: Path):
        _make_repo(tmp_path, REPO_ID, create_blobs_dir=False,
                    snapshot_files=["model.safetensors"])
        assert get_incomplete_size(REPO_ID, cache_root=tmp_path) == 0.0

    def test_zero_when_no_incomplete_files(self, tmp_path: Path):
        _make_repo(tmp_path, REPO_ID, complete_blobs=[4 * MB])
        assert get_incomplete_size(REPO_ID, cache_root=tmp_path) == 0.0

    def test_correct_size_single_file(self, tmp_path: Path):
        _make_repo(tmp_path, REPO_ID, incomplete_sizes=[10 * MB])
        result = get_incomplete_size(REPO_ID, cache_root=tmp_path)
        expected = 10 * MB / (1024**3)
        assert abs(result - expected) < 0.001

    def test_correct_size_multiple_files(self, tmp_path: Path):
        sizes = [10 * MB, 5 * MB, 2 * MB]
        _make_repo(tmp_path, REPO_ID, incomplete_sizes=sizes)
        result = get_incomplete_size(REPO_ID, cache_root=tmp_path)
        expected = sum(sizes) / (1024**3)
        assert abs(result - expected) < 0.001

    def test_ignores_complete_blobs(self, tmp_path: Path):
        _make_repo(
            tmp_path, REPO_ID,
            complete_blobs=[4 * MB],
            incomplete_sizes=[5 * MB],
        )
        result = get_incomplete_size(REPO_ID, cache_root=tmp_path)
        expected = 5 * MB / (1024**3)
        assert abs(result - expected) < 0.001


# ---------------------------------------------------------------------------
# get_cache_size
# ---------------------------------------------------------------------------

class TestGetCacheSize:

    def test_zero_when_no_cache(self, tmp_path: Path):
        assert get_cache_size(REPO_ID, cache_root=tmp_path) == 0.0

    def test_includes_complete_blobs(self, tmp_path: Path):
        _make_repo(
            tmp_path, REPO_ID,
            complete_blobs=[2 * MB],
            snapshot_files=["config.json"],
        )
        result = get_cache_size(REPO_ID, cache_root=tmp_path)
        assert result >= 2 * MB / (1024**3)

    def test_includes_incomplete_files(self, tmp_path: Path):
        _make_repo(tmp_path, REPO_ID, incomplete_sizes=[1 * MB])
        result = get_cache_size(REPO_ID, cache_root=tmp_path)
        assert result >= 1 * MB / (1024**3)

    def test_includes_both(self, tmp_path: Path):
        _make_repo(
            tmp_path, REPO_ID,
            complete_blobs=[2 * MB],
            incomplete_sizes=[1 * MB],
        )
        result = get_cache_size(REPO_ID, cache_root=tmp_path)
        assert result >= 3 * MB / (1024**3)


# ---------------------------------------------------------------------------
# Download lifecycle simulation
# ---------------------------------------------------------------------------

class TestDownloadLifecycle:

    def test_lifecycle_not_downloaded_to_complete(self, tmp_path: Path):
        folder = REPO_ID.replace("/", "--")
        model_dir = tmp_path / f"models--{folder}"

        # Stage 1: nothing exists
        assert is_model_cached(REPO_ID, cache_root=tmp_path) is False

        # Stage 2: download in progress, .incomplete files exist
        _make_repo(
            tmp_path, REPO_ID,
            snapshot_files=["config.json"],
            incomplete_sizes=[5 * MB],
        )
        assert is_model_cached(REPO_ID, cache_root=tmp_path) is False
        assert get_incomplete_size(REPO_ID, cache_root=tmp_path) > 0

        # Stage 3: download complete -- .incomplete removed, weights present
        for f in (model_dir / "blobs").glob("*.incomplete"):
            f.unlink()
        snap = model_dir / "snapshots" / "abc123def"
        (snap / "model.safetensors").write_bytes(b"complete")
        assert is_model_cached(REPO_ID, cache_root=tmp_path) is True
        assert get_incomplete_size(REPO_ID, cache_root=tmp_path) == 0.0
