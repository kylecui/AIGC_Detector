"""Background wrapper for AI text generation.

Runs the generation pipeline and logs progress to a file.
Usage: uv run python scripts/run_generate.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("dataset/generate_log.txt", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def main() -> int:
    start = time.time()
    log.info("Starting AI text generation pipeline")
    log.info("Models: qwen2.5-7b, chatglm3-6b, yi-6b-chat")
    log.info("Num per prompt: 10")

    try:
        from src.aigc_detector.data.generator import generate_all

        output_path = generate_all(
            num_per_prompt=10,
            models=["qwen2.5-7b", "chatglm3-6b", "yi-6b-chat"],
            prompt_config_path="configs/prompts.yaml",
        )
        elapsed = time.time() - start
        log.info(f"Generation complete in {elapsed / 3600:.1f} hours")
        log.info(f"Output: {output_path}")

        # Count records
        import json

        counts: dict[str, int] = {}
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    k = f"{r.get('label', '?')}/{r.get('lang', '?')}"
                    counts[k] = counts.get(k, 0) + 1
        log.info(f"Total records: {sum(counts.values())}")
        for k, v in sorted(counts.items()):
            log.info(f"  {k}: {v}")

    except Exception:
        log.exception("Generation failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
