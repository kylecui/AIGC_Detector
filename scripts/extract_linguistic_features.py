"""Extract linguistic features from a JSONL dataset.

For each record, computes the 14 stylistic features and writes an augmented
JSONL. Token log-prob features (M5/M6) are left NaN — they require an LM and
are computed separately if needed.

Usage:
    uv run python scripts/extract_linguistic_features.py \\
        --input dataset/processed/train.jsonl \\
        --output dataset/processed/train.linguistic.jsonl

Resume support: if --output exists, count completed records and skip them.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from aigc_detector.detection.linguistic import LinguisticFeatureExtractor
from aigc_detector.utils.text import is_chinese

logger = logging.getLogger(__name__)


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a file."""
    n = 0
    with open(path, encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract 14 linguistic-stylistic features per record from a JSONL dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL path (one record per line).")
    parser.add_argument("--output", type=Path, required=True, help="Output augmented JSONL path.")
    parser.add_argument("--text-key", default="text", help="Record key holding the text content.")
    parser.add_argument(
        "--lang-key",
        default="lang",
        help="Record key holding the language ('en'/'zh'). Absent => inferred via is_chinese().",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=200,
        help="Minimum chars to compute features; shorter texts get all-NaN features.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N input records (for debugging).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    input_path: Path = args.input
    output_path: Path = args.output
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: count existing output records.
    already_done = 0
    if output_path.exists():
        already_done = _count_lines(output_path)
        if already_done > 0:
            logger.info("Resume: skipping %d already-processed records in %s", already_done, output_path)

    total_input = _count_lines(input_path)
    total = total_input if args.limit is None else min(total_input, args.limit)
    logger.info("Input:  %s (total_lines=%d, will_process=%d)", input_path, total_input, total)
    logger.info("Output: %s", output_path)
    logger.info("Params: text_key=%s lang_key=%s min_text_chars=%d", args.text_key, args.lang_key, args.min_text_chars)

    extractor = LinguisticFeatureExtractor(min_text_chars=args.min_text_chars)

    processed = already_done
    errors = 0
    skipped_no_text = 0
    inferred_lang_count = 0

    mode = "a" if already_done > 0 else "w"
    seen_after_resume = 0
    with open(input_path, encoding="utf-8") as fin, open(output_path, mode, encoding="utf-8") as fout:
        for i, raw_line in enumerate(fin):
            # Skip already-processed records on resume.
            if i < already_done:
                continue
            if args.limit is not None and seen_after_resume >= args.limit:
                break
            seen_after_resume += 1

            stripped = raw_line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning("Malformed JSON on line %d, skipping", i + 1)
                errors += 1
                continue

            text = record.get(args.text_key)
            if not isinstance(text, str) or not text.strip():
                skipped_no_text += 1
                errors += 1
                continue

            # Language resolution: explicit lang key wins; otherwise infer.
            lang = record.get(args.lang_key)
            if not lang:
                lang = "zh" if is_chinese(text) else "en"
                inferred_lang_count += 1

            try:
                feats = extractor.extract(text, lang=lang, token_log_probs=None)
            except Exception:  # extractor is pure-Python but we never crash the batch run
                logger.warning("Feature extraction failed on line %d", i + 1, exc_info=True)
                errors += 1
                continue

            record["linguistic_features"] = feats.to_dict()
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

            if processed % 500 == 0 and processed > 0:
                logger.info("Progress: %d/%d (errors=%d)", processed, total, errors)

    summary = {
        "processed": processed,
        "errors": errors,
        "skipped_no_text": skipped_no_text,
        "inferred_lang": inferred_lang_count,
        "total_input_lines": total_input,
        "target_total": total,
        "output_path": str(output_path),
    }
    logger.info("Done: %s", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
