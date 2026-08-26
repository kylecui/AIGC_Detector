"""P0-2: pin every runtime model to its exact HF revision + hash registry.

Writes models/calibration/model_pins.json — the reproducibility contract:
T=5.645 and floor=0.46 are calibrated against THESE revisions; any drift
silently invalidates them. Resolution order mirrors runtime: local models/
override (configs/models.yaml local_path) -> HF cache snapshot commit hash.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml  # noqa: E402
from huggingface_hub import scan_cache_dir  # noqa: E402

ROOT = Path(__file__).parent.parent
REGISTRY = ROOT / "configs/models.yaml"
OUT = ROOT / "models/calibration/model_pins.json"


def main() -> int:
    reg = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["models"]
    cache = {r.repo_id: r for r in scan_cache_dir().repos}

    pins: dict[str, dict] = {}
    missing: list[str] = []
    for name, info in sorted(reg.items()):
        repo = info.get("hf_id") or info.get("model_name") or name
        entry = {"registry_name": name}
        if info.get("local_path"):
            entry["source"] = f"local:{info['local_path']}"
        if repo in cache:
            r = cache[repo]
            entry["hf_id"] = repo
            # newest local snapshot's commit hash (CachedRepoInfo.revisions)
            revs = sorted(r.revisions, key=lambda v: getattr(v, "last_modified", "") or "")
            entry["revision"] = revs[-1].commit_hash if revs else "unknown"
            entry["size_on_disk_gb"] = round(r.size_on_disk / 1e9, 2)
        elif not entry.get("source", "").startswith("local:"):
            missing.append(f"{name} ({repo})")
        pins[name] = entry

    OUT.write_text(json.dumps({
        "_doc": ("Reproducibility pins: calibration (T=5.645, floor=0.46) and "
                 "probe baselines are valid ONLY against these revisions. "
                 "Regenerate via scripts/pin_model_versions.py after any model update."),
        "pinned_at": "2026-08-21",
        "models": pins,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"pinned {len(pins)} models -> {OUT}")
    for k, v in pins.items():
        rev = v.get("revision", "-")[:12]
        print(f"  {k:<28} {v.get('hf_id', v.get('source','-')):<52} {rev}")
    if missing:
        print(f"\nNOT IN CACHE (resolved at runtime, unpinnable now): {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
