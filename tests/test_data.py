"""Tests for Phase 1 data pipeline: processor, splitter, crawler, generator (unit tests only).

GPU-dependent tests (generator, mixer) are marked with pytest.mark.slow and skipped by default.
"""

import json
from pathlib import Path

import pytest

from aigc_detector.data.processor import (
    deduplicate,
    detect_boilerplate,
    detect_encoding_issues,
    filter_text,
    process_records,
    truncate_text,
)
from aigc_detector.data.splitter import print_split_stats, split_dataset

# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------


def _sample_records(n: int = 20, lang: str = "en", label: str = "human") -> list[dict]:
    """Generate sample JSONL records for testing.

    Each record contains diverse sentences to pass the repetitive-text filter.
    """
    sentence_pool = [
        "Technology continues to advance at a rapid pace in modern society.",
        "Researchers have discovered new methods for analyzing complex data.",
        "Education systems around the world are undergoing significant reform.",
        "Climate change remains one of the most pressing global challenges.",
        "Artificial intelligence is transforming healthcare diagnostics.",
        "The global economy shows signs of recovery after recent downturns.",
        "Space exploration has entered a new era of commercial ventures.",
        "Renewable energy adoption is accelerating across developing nations.",
        "Urbanization is reshaping how cities plan infrastructure and services.",
        "Advances in biotechnology promise breakthroughs in disease treatment.",
        "Digital privacy concerns are driving new legislation worldwide.",
        "Ocean conservation efforts have gained momentum among policymakers.",
        "The rise of remote work has changed corporate culture fundamentally.",
        "Agricultural innovation is needed to feed a growing population.",
        "Quantum computing may revolutionize cryptography and drug discovery.",
    ]
    records = []
    for i in range(n):
        # Rotate through sentences to create unique, diverse texts per record
        start = i % len(sentence_pool)
        sentences = [sentence_pool[(start + j) % len(sentence_pool)] for j in range(10)]
        sentences[0] = f"This is sample record number {i}. " + sentences[0]
        text = " ".join(sentences)
        records.append(
            {
                "id": f"h_{i:04d}",
                "text": text,
                "label": label,
                "lang": lang,
                "source": "test",
                "domain": "general",
            }
        )
    return records


def _sample_records_zh(n: int = 10) -> list[dict]:
    """Generate sample Chinese records with diverse sentences."""
    sentence_pool = [
        "人工智能技术的发展正在深刻改变人类社会的面貌。",
        "教育改革需要适应数字时代的新要求和挑战。",
        "气候变化问题已成为全球各国共同关注的焦点。",
        "医疗技术的进步为疾病治疗带来了新的希望。",
        "经济全球化对发展中国家产生了深远的影响。",
        "城市化进程推动了基础设施建设的快速发展。",
        "可再生能源的应用正在全球范围内加速推广。",
        "数据隐私保护已成为信息时代的重要议题。",
        "航天科技的突破为人类探索宇宙开辟了新篇章。",
        "生物技术的创新有望在医药领域取得重大突破。",
    ]
    records = []
    for i in range(n):
        start = i % len(sentence_pool)
        sentences = [sentence_pool[(start + j) % len(sentence_pool)] for j in range(8)]
        sentences[0] = f"这是第{i}个中文测试文本。" + sentences[0]
        text = "".join(sentences)
        records.append(
            {
                "id": f"h_zh_{i:04d}",
                "text": text,
                "label": "human",
                "lang": "zh",
                "source": "test",
                "domain": "general",
            }
        )
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ===========================================================================
# Processor Tests
# ===========================================================================


class TestDetectBoilerplate:
    def test_clean_text_passes(self):
        text = "The rapid advancement of artificial intelligence has transformed numerous industries. " * 5
        assert not detect_boilerplate(text)

    def test_boilerplate_english(self):
        text = (
            "Click here to subscribe. Terms of Service apply. "
            "Privacy Policy updated. All rights reserved. Cookie notice accepted."
        )
        assert detect_boilerplate(text)

    def test_boilerplate_chinese(self):
        text = "点击查看更多内容。版权所有，侵权必究。免责声明：本文仅代表作者观点。" * 2
        assert detect_boilerplate(text)

    def test_high_url_density(self):
        text = "Visit https://example.com and https://test.com and https://foo.com for more"
        assert detect_boilerplate(text)


class TestDetectEncodingIssues:
    def test_clean_text_passes(self):
        assert not detect_encoding_issues("Hello world, this is a normal text.")

    def test_replacement_char(self):
        assert detect_encoding_issues("Hello \ufffd world")

    def test_clean_chinese(self):
        assert not detect_encoding_issues("中文文本没有编码问题。")


class TestFilterText:
    def test_passes_good_text(self):
        text = (
            "A well-written article about technology explores many facets of innovation. "
            "Modern computing has transformed how we interact with information daily. "
            "Artificial intelligence represents one of the most significant recent advances. "
            "Cloud platforms enable businesses to scale operations efficiently worldwide. "
            "Open source software continues to drive collaboration across the industry. "
            "Cybersecurity threats evolve constantly and require vigilant defenses. "
            "Machine learning models improve with access to larger training datasets. "
            "The semiconductor industry faces new challenges in chip miniaturization. "
        )
        passes, reason = filter_text(text)
        assert passes
        assert reason == ""

    def test_rejects_too_short(self):
        passes, reason = filter_text("Short text.")
        assert not passes
        assert reason == "too_short"

    def test_rejects_boilerplate(self):
        text = (
            "Click here to subscribe. Terms of Service apply. "
            "Privacy Policy updated. All rights reserved. Cookie notice."
        ) * 3
        passes, reason = filter_text(text, min_chars=10)
        assert not passes
        assert reason == "boilerplate"

    def test_rejects_encoding_issues(self):
        text = "Some text with encoding issues \ufffd\ufffd\ufffd. " * 20
        passes, reason = filter_text(text)
        assert not passes
        assert reason == "encoding_issues"

    def test_rejects_repetitive(self):
        text = "The same sentence repeated. " * 50
        passes, reason = filter_text(text)
        assert not passes
        assert reason == "repetitive"


class TestTruncateText:
    def test_short_text_unchanged(self):
        text = "Short text that doesn't need truncation."
        assert truncate_text(text, max_chars=1000) == text

    def test_long_text_truncated(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        result = truncate_text(text, max_chars=40)
        assert len(result) <= 40
        assert result.endswith(".")

    def test_chinese_truncation(self):
        text = "第一句话。第二句话。第三句话。第四句话。第五句话。"
        result = truncate_text(text, max_chars=20)
        assert len(result) <= 20


class TestDeduplicate:
    def test_exact_dedup(self):
        records = [
            {"text": "Exact same text here. " * 10, "label": "human", "lang": "en"},
            {"text": "Exact same text here. " * 10, "label": "human", "lang": "en"},
            {"text": "Different text here. " * 10, "label": "human", "lang": "en"},
        ]
        result = deduplicate(records)
        assert len(result) == 2

    def test_no_duplicates(self):
        records = [{"text": f"Unique text number {i}. " * 10, "label": "human", "lang": "en"} for i in range(5)]
        result = deduplicate(records)
        assert len(result) == 5

    def test_near_dedup(self):
        base = (
            "Machine learning models require large datasets for effective training. "
            "Deep neural networks have transformed computer vision applications. "
            "Natural language processing enables machines to understand human text. "
            "Transfer learning reduces the need for task-specific training data. "
            "Reinforcement learning agents can master complex strategic games. "
            "Gradient descent optimization forms the backbone of model training. "
            "Convolutional networks excel at recognizing patterns in images. "
            "Attention mechanisms have revolutionized sequence modeling tasks. "
            "Regularization techniques help prevent overfitting in neural networks. "
            "Distributed computing accelerates the training of very large models. "
        )
        records = [
            {"text": base, "label": "human", "lang": "en"},
            {"text": base + " Extra word.", "label": "human", "lang": "en"},  # Near-duplicate
            {"text": "Completely different content. " * 10, "label": "human", "lang": "en"},
        ]
        result = deduplicate(records, similarity_threshold=0.9)
        assert len(result) == 2

    def test_empty_input(self):
        assert deduplicate([]) == []


class TestProcessRecords:
    def test_basic_pipeline(self, tmp_path):
        records = _sample_records(10)
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        _write_jsonl(input_path, records)

        stats = process_records(input_path, output_path)

        assert output_path.exists()
        assert stats["total_input"] == 10
        assert stats["after_dedup"] > 0

        # Verify output is valid JSONL
        with open(output_path, encoding="utf-8") as f:
            output_records = [json.loads(line) for line in f if line.strip()]
        assert len(output_records) == stats["after_dedup"]

    def test_filters_short_text(self, tmp_path):
        records = [
            {
                "id": "h_0001",
                "text": "Too short",
                "label": "human",
                "lang": "en",
                "source": "test",
                "domain": "general",
            },
        ]
        input_path = tmp_path / "input.jsonl"
        output_path = tmp_path / "output.jsonl"
        _write_jsonl(input_path, records)

        stats = process_records(input_path, output_path, min_chars=200)

        assert stats["total_input"] == 1
        assert stats["after_dedup"] == 0
        assert stats["rejected_reasons"]["too_short"] == 1


# ===========================================================================
# Splitter Tests
# ===========================================================================


class TestSplitDataset:
    def test_basic_split(self, tmp_path):
        records = _sample_records(100)
        input_path = tmp_path / "all.jsonl"
        output_dir = tmp_path / "splits"
        _write_jsonl(input_path, records)

        stats = split_dataset(input_path, output_dir)

        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "val.jsonl").exists()
        assert (output_dir / "test.jsonl").exists()

        assert stats["train_count"] == 80
        assert stats["val_count"] == 10
        assert stats["test_count"] == 10

    def test_stratified_split(self, tmp_path):
        # Mix human + AI + two languages
        records = (
            _sample_records(40, lang="en", label="human")
            + _sample_records(40, lang="en", label="ai")
            + _sample_records_zh(20)
        )
        input_path = tmp_path / "mixed.jsonl"
        output_dir = tmp_path / "strat"
        _write_jsonl(input_path, records)

        stats = split_dataset(input_path, output_dir, stratify_by=["label", "lang"])

        total = stats["train_count"] + stats["val_count"] + stats["test_count"]
        assert total == 100

        # Check that both labels and languages appear in each split
        assert "label_distribution" in stats
        assert "lang_distribution" in stats

    def test_invalid_ratios(self, tmp_path):
        records = _sample_records(10)
        input_path = tmp_path / "data.jsonl"
        _write_jsonl(input_path, records)

        with pytest.raises(ValueError, match="Ratios must sum to 1.0"):
            split_dataset(input_path, tmp_path / "out", train_ratio=0.5, val_ratio=0.1, test_ratio=0.1)

    def test_custom_ratios(self, tmp_path):
        records = _sample_records(100)
        input_path = tmp_path / "data.jsonl"
        _write_jsonl(input_path, records)

        stats = split_dataset(
            input_path,
            tmp_path / "custom",
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        assert stats["train_count"] == 70
        assert stats["val_count"] == 15
        assert stats["test_count"] == 15

    def test_reproducible(self, tmp_path):
        records = _sample_records(50)
        input_path = tmp_path / "data.jsonl"
        _write_jsonl(input_path, records)

        stats1 = split_dataset(input_path, tmp_path / "run1", seed=42)
        stats2 = split_dataset(input_path, tmp_path / "run2", seed=42)

        assert stats1["train_count"] == stats2["train_count"]

    def test_print_stats_no_crash(self, tmp_path, capsys):
        records = _sample_records(20)
        input_path = tmp_path / "data.jsonl"
        _write_jsonl(input_path, records)

        stats = split_dataset(input_path, tmp_path / "out")
        # Should not raise
        print_split_stats(stats)


# ===========================================================================
# Crawler Tests (unit only — no network calls)
# ===========================================================================


class TestCrawlerUnit:
    def test_make_record(self):
        from aigc_detector.data.crawler import _make_record

        record = _make_record("Some text here", "en", "wikipedia")
        assert record["label"] == "human"
        assert record["lang"] == "en"
        assert record["source"] == "wikipedia"
        assert record["id"].startswith("h_")
        assert len(record["id"]) == 10  # "h_" + 8 hex chars

    def test_make_record_chinese(self):
        from aigc_detector.data.crawler import _make_record

        record = _make_record("中文文本", "zh", "hc3", "finance")
        assert record["lang"] == "zh"
        assert record["domain"] == "finance"

    def test_hc3_loader_domain_map(self):
        from aigc_detector.data.crawler import HC3Loader

        loader = HC3Loader()
        assert loader.SOURCE_DOMAIN_MAP["finance"] == "finance"
        assert loader.SOURCE_DOMAIN_MAP["medicine"] == "healthcare"


# ===========================================================================
# Generator Tests (unit only — no GPU needed)
# ===========================================================================


class TestGeneratorUnit:
    def test_load_prompts(self):
        from aigc_detector.data.generator import load_prompts

        config = load_prompts("configs/prompts.yaml")
        assert "domains" in config
        assert "styles" in config
        assert "system_prompts" in config
        assert "prompts" in config
        assert "generation" in config

    def test_prompt_domains(self):
        from aigc_detector.data.generator import load_prompts

        config = load_prompts("configs/prompts.yaml")
        domains = config["domains"]
        assert len(domains) >= 5
        for key in ["technology", "education", "healthcare", "law", "finance"]:
            assert key in domains

    def test_prompt_styles(self):
        from aigc_detector.data.generator import load_prompts

        config = load_prompts("configs/prompts.yaml")
        styles = config["styles"]
        for key in ["news_report", "argumentative", "popular_science", "commentary"]:
            assert key in styles

    def test_prompt_has_bilingual_templates(self):
        from aigc_detector.data.generator import load_prompts

        config = load_prompts("configs/prompts.yaml")
        tech_news = config["prompts"]["technology"]["news_report"]
        assert "zh" in tech_news
        assert "en" in tech_news
        assert len(tech_news["zh"]) >= 1
        assert len(tech_news["en"]) >= 1

    def test_generation_config(self):
        from aigc_detector.data.generator import load_prompts

        config = load_prompts("configs/prompts.yaml")
        gen = config["generation"]
        assert "temperatures" in gen
        assert "top_p_values" in gen
        assert "length_ranges" in gen
        assert len(gen["temperatures"]) >= 2

    def test_resolve_model_source_name(self):
        from aigc_detector.data.generator import _resolve_model_source_name

        assert _resolve_model_source_name("qwen2.5-7b") == "qwen2.5"
        assert _resolve_model_source_name("mistral-7b-gptq") == "mistral7b"
        assert _resolve_model_source_name("unknown-model") == "unknown-model"

    def test_select_prompts_for_model(self):
        from aigc_detector.data.generator import _select_prompts_for_model, load_prompts

        config = load_prompts("configs/prompts.yaml")
        tasks = _select_prompts_for_model(config, ["zh"])
        assert len(tasks) > 0
        for task in tasks:
            assert task["lang"] == "zh"
            assert "system_prompt" in task
            assert "user_prompt" in task

    def test_select_prompts_bilingual(self):
        from aigc_detector.data.generator import _select_prompts_for_model, load_prompts

        config = load_prompts("configs/prompts.yaml")
        tasks = _select_prompts_for_model(config, ["zh", "en"])
        langs = {t["lang"] for t in tasks}
        assert "zh" in langs
        assert "en" in langs
