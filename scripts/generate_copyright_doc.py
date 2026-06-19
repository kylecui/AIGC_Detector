"""Generate software copyright source code document (软著源代码文档).

Chinese software copyright registration requires:
- First 30 pages + last 30 pages of source code (60 pages total)
- ~50 lines per page
- Page header: software name + version + page number
- Sequential page numbering
- No content removed or modified

Output: docs/software-copyright-source-code.txt
"""
import sys
from pathlib import Path

SOFTWARE_NAME = "中英文AI生成文本检测系统"
VERSION = "V1.0"
LINES_PER_PAGE = 50
FIRST_PAGES = 30
LAST_PAGES = 30

# Source files in logical order (dependency: foundation → core → API → utilities)
SOURCE_FILES = [
    # Foundation
    "src/aigc_detector/__init__.py",
    "src/aigc_detector/config.py",
    # Detection core
    "src/aigc_detector/detection/__init__.py",
    "src/aigc_detector/detection/language.py",
    "src/aigc_detector/detection/statistical.py",
    "src/aigc_detector/detection/linguistic.py",
    "src/aigc_detector/detection/encoder.py",
    "src/aigc_detector/detection/binoculars.py",
    "src/aigc_detector/detection/ensemble.py",
    "src/aigc_detector/detection/pipeline.py",
    # Training
    "src/aigc_detector/training/__init__.py",
    "src/aigc_detector/training/trainer.py",
    "src/aigc_detector/training/evaluator.py",
    "src/aigc_detector/training/calibration.py",
    # API layer
    "src/aigc_detector/api/__init__.py",
    "src/aigc_detector/api/schemas.py",
    "src/aigc_detector/api/middleware.py",
    "src/aigc_detector/api/routes.py",
    "src/aigc_detector/api/main.py",
    # Model management
    "src/aigc_detector/models/__init__.py",
    "src/aigc_detector/models/registry.py",
    "src/aigc_detector/models/manager.py",
    # Utilities
    "src/aigc_detector/utils/__init__.py",
    "src/aigc_detector/utils/text.py",
    "src/aigc_detector/utils/logging.py",
    "src/aigc_detector/utils/hf_cache.py",
    # Data pipeline
    "src/aigc_detector/data/__init__.py",
    "src/aigc_detector/data/processor.py",
    "src/aigc_detector/data/splitter.py",
    "src/aigc_detector/data/crawler.py",
    "src/aigc_detector/data/generator.py",
    "src/aigc_detector/data/mixer.py",
]


def collect_all_lines() -> list[str]:
    """Collect all source lines in order, with file separators."""
    all_lines: list[str] = []
    project_root = Path(__file__).parent.parent

    for filepath in SOURCE_FILES:
        full_path = project_root / filepath
        if not full_path.exists():
            print(f"  SKIP (not found): {filepath}")
            continue

        # File separator header
        all_lines.append(f"# {'=' * 60}")
        all_lines.append(f"# File: {filepath}")
        all_lines.append(f"# {'=' * 60}")
        all_lines.append("")

        # File contents
        content = full_path.read_text(encoding="utf-8")
        lines = content.split("\n")
        all_lines.extend(lines)

        # Ensure file ends with newline
        if lines and lines[-1].strip():
            all_lines.append("")

        all_lines.append("")  # Blank line between files

    return all_lines


def generate_document() -> None:
    """Generate the paginated source code document."""
    output_path = Path(__file__).parent.parent / "docs" / "software-copyright-source-code.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_lines = collect_all_lines()
    total_lines = len(all_lines)
    total_pages = (total_lines + LINES_PER_PAGE - 1) // LINES_PER_PAGE

    print(f"Total source lines: {total_lines}")
    print(f"Total pages: {total_pages}")
    print(f"Need: first {FIRST_PAGES} + last {LAST_PAGES} = {FIRST_PAGES + LAST_PAGES} pages")

    # Select first N and last N pages
    first_end_line = FIRST_PAGES * LINES_PER_PAGE
    last_start_line = max(first_end_line, total_lines - LAST_PAGES * LINES_PER_PAGE)

    first_chunk = all_lines[:first_end_line]
    last_chunk = all_lines[last_start_line:]

    print(f"First chunk: lines 1-{first_end_line} ({len(first_chunk)} lines, {FIRST_PAGES} pages)")
    print(f"Last chunk: lines {last_start_line + 1}-{total_lines} ({len(last_chunk)} lines)")
    actual_last_pages = (len(last_chunk) + LINES_PER_PAGE - 1) // LINES_PER_PAGE
    print(f"Last chunk pages: {actual_last_pages}")

    # Generate document with page headers
    output_lines: list[str] = []
    page_num = 0

    def add_page(lines: list[str], is_last_section: bool = False) -> None:
        nonlocal page_num
        page_num += 1
        header = f"{SOFTWARE_NAME} {VERSION}                                              第 {page_num} 页"
        output_lines.append(header)
        output_lines.append("-" * 72)
        # Pad to exactly LINES_PER_PAGE content lines
        for i in range(LINES_PER_PAGE):
            if i < len(lines):
                output_lines.append(lines[i])
            else:
                output_lines.append("")  # blank padding
        output_lines.append("")

    # First pages
    for p in range(FIRST_PAGES):
        start = p * LINES_PER_PAGE
        end = start + LINES_PER_PAGE
        page_lines = first_chunk[start:end]
        if not page_lines:
            break
        add_page(page_lines, is_last_section=False)

    # Separator
    output_lines.append("")
    output_lines.append(f"{'=' * 72}")
    output_lines.append(f"# (中间部分省略，以下为源代码后 {LAST_PAGES} 页)")
    output_lines.append(f"{'=' * 72}")
    output_lines.append("")

    # Last pages
    for p in range(actual_last_pages):
        start = p * LINES_PER_PAGE
        end = start + LINES_PER_PAGE
        page_lines = last_chunk[start:end]
        if not page_lines:
            break
        add_page(page_lines, is_last_section=True)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"\nOutput: {output_path}")
    print(f"Total pages in document: {page_num}")
    print(f"Total lines in document: {len(output_lines)}")
    file_size = output_path.stat().st_size / 1024
    print(f"File size: {file_size:.1f} KB")


if __name__ == "__main__":
    generate_document()
