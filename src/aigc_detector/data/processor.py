"""Text processing pipeline: cleaning, filtering, and deduplication.

Reads raw JSONL records (from crawler or generator), applies quality filters,
truncates overly long texts at sentence boundaries, removes duplicates,
and writes cleaned JSONL output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from tqdm import tqdm

from aigc_detector.utils.text import clean_text, is_chinese, split_sentences_bilingual

console = Console()

# Boilerplate patterns (English)
EN_BOILERPLATE = re.compile(
    r"(?i)"
    r"terms of service|privacy policy|all rights reserved|cookie|"
    r"click here|subscribe|sign up|log in|"
    r"advertisement|sponsored content|"
    r"\[edit\]|references\s*$|see also\s*$|external links\s*$|"
    r"©\s*\d{4}|DMCA"
)

# Boilerplate patterns (Chinese)
ZH_BOILERPLATE = re.compile(
    r"点击查看|版权所有|免责声明|广告|"
    r"关注我们|扫码关注|订阅|"
    r"用户协议|隐私政策|"
    r"©\s*\d{4}|备案号"
)

# URL pattern
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

# Mojibake / encoding issue patterns
MOJIBAKE_PATTERN = re.compile(
    r"[\ufffd]"  # Unicode replacement character
    r"|Ã[\x80-\xbf]"  # Common UTF-8 → Latin-1 mojibake
    r"|â\x80[\x90-\x9f]"  # More mojibake
)


def detect_boilerplate(text: str) -> bool:
    """Check if text contains significant boilerplate content.

    Returns True if boilerplate is detected (text should be rejected).
    """
    # Check boilerplate phrase density
    if is_chinese(text):
        matches = ZH_BOILERPLATE.findall(text)
    else:
        matches = EN_BOILERPLATE.findall(text)

    if len(matches) >= 3:
        return True

    # Check URL density — more than 5 URLs per 1000 chars is suspicious
    urls = URL_PATTERN.findall(text)
    if len(text) > 0 and (len(urls) / max(len(text), 1)) * 1000 > 5:
        return True

    # Check if text is mostly special characters / non-alphabetic
    alnum_count = sum(1 for c in text if c.isalnum())
    if len(text) > 50 and alnum_count / len(text) < 0.5:
        return True

    return False


def detect_encoding_issues(text: str) -> bool:
    """Check if text has encoding problems.

    Returns True if encoding issues are detected (text should be rejected).
    """
    # Unicode replacement character
    if "\ufffd" in text:
        return True

    # Mojibake patterns
    if MOJIBAKE_PATTERN.search(text):
        return True

    # Excessive non-printable characters (control chars except newline/tab)
    non_printable = sum(1 for c in text if ord(c) < 32 and c not in "\n\r\t")
    if len(text) > 0 and non_printable / len(text) > 0.01:
        return True

    return False


def filter_text(text: str, min_chars: int = 200, max_chars: int = 10000) -> tuple[bool, str]:
    """Apply quality filters to a text sample.

    Args:
        text: The text to evaluate.
        min_chars: Minimum character count.
        max_chars: Maximum character count (will be truncated separately, not rejected).

    Returns:
        Tuple of (passes_filter, rejection_reason).
        If passes_filter is True, rejection_reason is empty string.
    """
    if len(text) < min_chars:
        return False, "too_short"

    if detect_encoding_issues(text):
        return False, "encoding_issues"

    if detect_boilerplate(text):
        return False, "boilerplate"

    # Check if text is mostly repetitive (same sentence repeated)
    sentences = split_sentences_bilingual(text)
    if len(sentences) >= 3:
        unique_sentences = set(s.strip().lower() for s in sentences)
        if len(unique_sentences) / len(sentences) < 0.5:
            return False, "repetitive"

    return True, ""


def truncate_text(text: str, max_chars: int = 10000) -> str:
    """Truncate text to max_chars at a sentence boundary.

    If text is shorter than max_chars, returns it unchanged.
    Otherwise, finds the last complete sentence within the limit.
    """
    if len(text) <= max_chars:
        return text

    sentences = split_sentences_bilingual(text)
    result = []
    total_len = 0

    for sentence in sentences:
        if total_len + len(sentence) > max_chars:
            break
        result.append(sentence)
        total_len += len(sentence)

    if not result:
        # Single sentence longer than max_chars — hard truncate
        return text[:max_chars]

    # Join: Chinese sentences don't need spaces, English do
    if is_chinese(text):
        return "".join(result)
    return " ".join(result)


def _normalize_for_hash(text: str) -> str:
    """Normalize text for deduplication hashing."""
    # Lowercase, collapse whitespace, strip
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _char_ngrams(text: str, n: int = 5) -> set[str]:
    """Extract character n-grams from text."""
    text = _normalize_for_hash(text)
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def deduplicate(records: list[dict], similarity_threshold: float = 0.9) -> list[dict]:
    """Remove exact and near-duplicate records.

    Phase 1: Exact dedup via normalized text hash.
    Phase 2: Near-dedup via character n-gram Jaccard similarity.

    For performance, near-dedup only compares records within the same
    (label, lang) group, and uses a simplified approach for large datasets.

    Args:
        records: List of JSONL record dicts (must have "text" field).
        similarity_threshold: Jaccard threshold for near-duplicate detection.

    Returns:
        Deduplicated list of records.
    """
    if not records:
        return records

    # Phase 1: Exact dedup
    seen_hashes: set[str] = set()
    phase1: list[dict] = []
    exact_dupes = 0

    for record in records:
        text_hash = hashlib.md5(_normalize_for_hash(record["text"]).encode("utf-8")).hexdigest()
        if text_hash in seen_hashes:
            exact_dupes += 1
            continue
        seen_hashes.add(text_hash)
        phase1.append(record)

    if exact_dupes > 0:
        console.print(f"[dim]Exact duplicates removed: {exact_dupes}[/]")

    # Phase 2: Near-dedup within (label, lang) groups
    # For large datasets, this is O(n²) within each group — acceptable for our scale
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(phase1):
        group_key = f"{record.get('label', 'unknown')}_{record.get('lang', 'unknown')}"
        groups[group_key].append(idx)

    # Pre-compute n-grams
    ngrams_cache: dict[int, set[str]] = {}
    remove_indices: set[int] = set()
    near_dupes = 0

    for group_key, indices in groups.items():
        # Skip near-dedup for very large groups (> 50K) — too slow
        if len(indices) > 50000:
            console.print(f"[yellow]Group {group_key} has {len(indices)} records, skipping near-dedup.[/]")
            continue

        for i, idx_a in enumerate(indices):
            if idx_a in remove_indices:
                continue

            if idx_a not in ngrams_cache:
                ngrams_cache[idx_a] = _char_ngrams(phase1[idx_a]["text"])

            # Only compare with subsequent records in the same group
            for idx_b in indices[i + 1 :]:
                if idx_b in remove_indices:
                    continue

                if idx_b not in ngrams_cache:
                    ngrams_cache[idx_b] = _char_ngrams(phase1[idx_b]["text"])

                sim = _jaccard_similarity(ngrams_cache[idx_a], ngrams_cache[idx_b])
                if sim >= similarity_threshold:
                    remove_indices.add(idx_b)
                    near_dupes += 1

    if near_dupes > 0:
        console.print(f"[dim]Near-duplicates removed: {near_dupes}[/]")

    result = [r for idx, r in enumerate(phase1) if idx not in remove_indices]
    return result


def process_records(
    input_path: Path,
    output_path: Path,
    min_chars: int = 200,
    max_chars: int = 10000,
    dedup_threshold: float = 0.9,
) -> dict:
    """Main processing pipeline: read → clean → filter → truncate → dedup → write.

    Args:
        input_path: Path to input JSONL file.
        output_path: Path to output JSONL file.
        min_chars: Minimum text length to keep.
        max_chars: Maximum text length (longer texts are truncated at sentence boundary).
        dedup_threshold: Jaccard similarity threshold for near-duplicate removal.

    Returns:
        Stats dict with processing summary.
    """
    # Read input
    records: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total_input = len(records)
    console.print(f"[bold blue]Processing:[/] {total_input} records from {input_path}")

    # Clean + Filter + Truncate
    filtered: list[dict] = []
    rejected_reasons: dict[str, int] = defaultdict(int)

    for record in tqdm(records, desc="Filtering", unit="rec"):
        # Clean text
        record["text"] = clean_text(record["text"])

        # Filter
        passes, reason = filter_text(record["text"], min_chars=min_chars, max_chars=max_chars)
        if not passes:
            rejected_reasons[reason] += 1
            continue

        # Truncate
        record["text"] = truncate_text(record["text"], max_chars=max_chars)

        filtered.append(record)

    passed_filter = len(filtered)
    console.print(f"[dim]Passed filter: {passed_filter}/{total_input}[/]")

    # Deduplicate
    deduped = deduplicate(filtered, similarity_threshold=dedup_threshold)
    after_dedup = len(deduped)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in deduped:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = {
        "total_input": total_input,
        "passed_filter": passed_filter,
        "after_dedup": after_dedup,
        "rejected_reasons": dict(rejected_reasons),
    }

    # Print summary
    console.rule("[bold]Processing Summary")
    console.print(f"Input records:    {total_input}")
    console.print(f"Passed filter:    {passed_filter}")
    console.print(f"After dedup:      {after_dedup}")
    console.print(f"Total removed:    {total_input - after_dedup}")
    if rejected_reasons:
        console.print("Rejection reasons:")
        for reason, count in sorted(rejected_reasons.items(), key=lambda x: -x[1]):
            console.print(f"  {reason}: {count}")
    console.print(f"Output: {output_path}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and clean text data for AIGC detection")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file path")
    parser.add_argument("--min-chars", type=int, default=200, help="Minimum text length (default: 200)")
    parser.add_argument("--max-chars", type=int, default=10000, help="Maximum text length (default: 10000)")
    parser.add_argument(
        "--dedup-threshold", type=float, default=0.9, help="Near-dedup Jaccard threshold (default: 0.9)"
    )
    args = parser.parse_args()

    process_records(
        input_path=args.input,
        output_path=args.output,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        dedup_threshold=args.dedup_threshold,
    )
