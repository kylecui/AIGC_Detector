"""Bilingual human text crawler: Wikipedia API + HC3 dataset loader.

Collects human-written text from:
1. Wikipedia API (zh + en) — random article extraction via MediaWiki API
2. HC3 dataset (HuggingFace) — human answers from the HC3-Chinese and HC3 datasets

Outputs JSONL records to dataset/raw/human_raw.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

import httpx
from rich.console import Console
from tqdm import tqdm

from aigc_detector.config import settings
from aigc_detector.utils.text import clean_text

console = Console()

WIKIPEDIA_API = "https://{lang}.wikipedia.org/w/api.php"
WIKIPEDIA_PARAMS = {
    "action": "query",
    "generator": "random",
    "grnnamespace": "0",
    "grnlimit": "10",
    "prop": "extracts",
    "explaintext": "true",
    "format": "json",
}

HC3_DATASETS = {
    "zh": "Hello-SimpleAI/HC3-Chinese",
    "en": "Hello-SimpleAI/HC3",
}


def _make_record(text: str, lang: str, source: str, domain: str = "general") -> dict:
    """Create a JSONL record for a human text sample."""
    return {
        "id": f"h_{uuid.uuid4().hex[:8]}",
        "text": text,
        "label": "human",
        "lang": lang,
        "source": source,
        "domain": domain,
    }


class WikipediaCrawler:
    """Async Wikipedia article crawler using MediaWiki API.

    Fetches random articles in plain text (no HTML parsing needed)
    with rate limiting via semaphore and politeness delays.
    """

    def __init__(self, max_concurrent: int = 5, delay_seconds: float = 1.0):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._delay = delay_seconds

    async def _fetch_batch(self, client: httpx.AsyncClient, lang: str, max_retries: int = 3) -> list[dict]:
        """Fetch a single batch of random articles from Wikipedia API."""
        url = WIKIPEDIA_API.format(lang=lang)
        data: dict | None = None
        async with self._semaphore:
            for attempt in range(max_retries):
                try:
                    resp = await client.get(url, params=WIKIPEDIA_PARAMS, timeout=30.0)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except (httpx.HTTPError, json.JSONDecodeError) as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    console.print(f"[red]Wikipedia API error ({lang}):[/] {e}")
                    return []

        if data is None:
            return []

        pages = data.get("query", {}).get("pages", {})
        records = []
        for page in pages.values():
            text = page.get("extract", "")
            text = clean_text(text)
            if len(text) < 200:
                continue
            records.append(_make_record(text, lang, "wikipedia"))

        return records

    async def crawl(self, lang: str, num_articles: int = 15000) -> list[dict]:
        """Crawl random Wikipedia articles for a given language.

        Args:
            lang: Language code ("en" or "zh").
            num_articles: Target number of articles to collect.

        Returns:
            List of JSONL-compatible record dicts.
        """
        collected: list[dict] = []
        batch_size = 10  # Wikipedia API returns up to 10 random articles per request
        num_batches = (num_articles // batch_size) + 1
        stall_count = 0
        max_stalls = 200  # Stop if too many consecutive empty batches

        console.print(f"[bold blue]Crawling Wikipedia ({lang}):[/] target={num_articles} articles")

        async with httpx.AsyncClient(
            headers={"User-Agent": "AIGCDetector/1.0 (research; contact@example.com)"},
            follow_redirects=True,
        ) as client:
            pbar = tqdm(total=num_articles, desc=f"Wikipedia ({lang})", unit="article")

            for _ in range(num_batches):
                if len(collected) >= num_articles:
                    break

                batch = await self._fetch_batch(client, lang)

                if not batch:
                    stall_count += 1
                    if stall_count >= max_stalls:
                        console.print(f"[yellow]Too many empty batches for {lang}, stopping early.[/]")
                        break
                else:
                    stall_count = 0
                    collected.extend(batch)
                    pbar.update(len(batch))

                await asyncio.sleep(self._delay)

            pbar.close()

        # Trim to exact target
        collected = collected[:num_articles]
        console.print(f"[bold green]Wikipedia ({lang}):[/] collected {len(collected)} articles")
        return collected


class HC3Loader:
    """Loader for HC3 (Human ChatGPT Comparison) dataset from HuggingFace.

    Extracts human_answers from the dataset. Each row may have multiple
    human answers — all are extracted as separate records.

    Uses the auto-converted parquet revision since the original HC3
    loading scripts are no longer supported in datasets v3.x+.
    """

    # Map source field to domain
    SOURCE_DOMAIN_MAP = {
        "finance": "finance",
        "medicine": "healthcare",
        "open_qa": "general",
        "reddit_eli5": "general",
        "wiki_csai": "technology",
        "baike": "general",
        "law": "general",
        "nlpcc_dbqa": "general",
        "psychology": "healthcare",
    }

    def load(self, lang: str) -> list[dict]:
        """Load HC3 dataset for a language and extract human answers.

        Args:
            lang: Language code ("en" or "zh").

        Returns:
            List of JSONL-compatible record dicts.
        """
        from datasets import load_dataset

        dataset_name = HC3_DATASETS.get(lang)
        if dataset_name is None:
            console.print(f"[yellow]No HC3 dataset for lang={lang}, skipping.[/]")
            return []

        console.print(f"[bold blue]Loading HC3 ({lang}):[/] {dataset_name}")

        try:
            ds = load_dataset(
                dataset_name,
                revision="refs/convert/parquet",
            )
        except Exception as e:
            console.print(f"[red]Failed to load HC3 ({lang}):[/] {e}")
            return []

        records: list[dict] = []

        # Find the data split
        split_data = ds.get("train", None)
        if split_data is None:
            for split_name in ds:
                split_data = ds[split_name]
                break

        if split_data is None:
            console.print(f"[yellow]No data splits found in HC3 ({lang}).[/]")
            return []

        for row in tqdm(split_data, desc=f"HC3 ({lang})", unit="row"):
            human_answers = row.get("human_answers", [])
            source = row.get("source", "unknown")
            domain = self.SOURCE_DOMAIN_MAP.get(source, "general")

            for answer in human_answers:
                if not isinstance(answer, str):
                    continue
                text = clean_text(answer)
                if len(text) < 200:
                    continue
                records.append(_make_record(text, lang, "hc3", domain))

        console.print(f"[bold green]HC3 ({lang}):[/] extracted {len(records)} human answers")
        return records


def _write_records(records: list[dict], output_path: Path) -> None:
    """Write records to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def collect_human_texts(
    output_dir: Path | None = None,
    num_wiki_per_lang: int = 15000,
    hc3: bool = True,
) -> Path:
    """Orchestrate human text collection from all sources.

    Collects Wikipedia articles (async) and HC3 answers (sync) for both
    Chinese and English, then writes everything to a single JSONL file.

    Args:
        output_dir: Directory for output. Defaults to settings.dataset_dir / "raw".
        num_wiki_per_lang: Number of Wikipedia articles per language.
        hc3: Whether to include HC3 dataset.

    Returns:
        Path to the output JSONL file.
    """
    if output_dir is None:
        output_dir = settings.dataset_dir / "raw"

    output_path = output_dir / "human_raw.jsonl"
    all_records: list[dict] = []

    console.rule("[bold]Human Text Collection Pipeline")

    # Wikipedia crawling (async)
    crawler = WikipediaCrawler(max_concurrent=5, delay_seconds=1.0)
    for lang in ["zh", "en"]:
        wiki_records = await crawler.crawl(lang, num_wiki_per_lang)
        all_records.extend(wiki_records)

    # HC3 loading (sync, but wrapped in the async context)
    if hc3:
        loader = HC3Loader()
        for lang in ["zh", "en"]:
            hc3_records = loader.load(lang)
            all_records.extend(hc3_records)

    # Write output
    _write_records(all_records, output_path)

    # Summary
    console.rule("[bold]Collection Summary")
    source_counts: dict[str, int] = {}
    lang_counts: dict[str, int] = {}
    for r in all_records:
        src = r["source"]
        lng = r["lang"]
        source_counts[src] = source_counts.get(src, 0) + 1
        lang_counts[lng] = lang_counts.get(lng, 0) + 1

    console.print(f"Total records: {len(all_records)}")
    console.print(f"By source: {source_counts}")
    console.print(f"By language: {lang_counts}")
    console.print(f"Output: {output_path}")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect human-written text from Wikipedia and HC3")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: dataset/raw)")
    parser.add_argument("--num-wiki", type=int, default=15000, help="Wikipedia articles per language (default: 15000)")
    parser.add_argument("--no-hc3", action="store_true", help="Skip HC3 dataset loading")
    args = parser.parse_args()

    asyncio.run(
        collect_human_texts(
            output_dir=args.output_dir,
            num_wiki_per_lang=args.num_wiki,
            hc3=not args.no_hc3,
        )
    )
