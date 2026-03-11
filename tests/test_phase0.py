"""Tests for Phase 0 setup: config, registry, text utils."""

from pathlib import Path

from aigc_detector.config import Settings
from aigc_detector.models.registry import ModelInfo, load_registry
from aigc_detector.utils.text import clean_text, is_chinese, split_sentences_bilingual, text_stats


class TestConfig:
    def test_settings_defaults(self):
        s = Settings(hf_token="", openai_api_key="")
        assert s.device == "cuda"
        assert s.max_vram_gb == 11.0
        assert isinstance(s.model_dir, Path)
        assert isinstance(s.dataset_dir, Path)

    def test_settings_override(self):
        s = Settings(hf_token="test", openai_api_key="test", device="cpu", max_vram_gb=8.0)
        assert s.device == "cpu"
        assert s.max_vram_gb == 8.0


class TestRegistry:
    def test_load_registry(self):
        registry = load_registry("configs/models.yaml")
        assert len(registry) >= 10
        assert "mistral-7b-gptq" in registry
        assert "deberta-v3-large" in registry

    def test_model_info_fields(self):
        registry = load_registry("configs/models.yaml")
        mistral = registry["mistral-7b-gptq"]
        assert isinstance(mistral, ModelInfo)
        assert mistral.hf_id == "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
        assert mistral.purpose == "generation"
        assert mistral.language == "en"
        assert mistral.vram_gb == 4.5
        assert mistral.local_path == "models/mistral7b-gptq"

    def test_filter_by_purpose(self):
        from aigc_detector.models.registry import get_models_by_purpose, _registry

        # Reset global cache to load fresh
        import aigc_detector.models.registry as reg

        reg._registry = None
        gen_models = get_models_by_purpose("generation")
        assert len(gen_models) >= 5
        for m in gen_models:
            assert m.purpose == "generation"

    def test_filter_by_language(self):
        from aigc_detector.models.registry import get_models_by_language

        import aigc_detector.models.registry as reg

        reg._registry = None
        zh_models = get_models_by_language("zh")
        assert len(zh_models) >= 1
        for m in zh_models:
            assert m.language == "zh"


class TestTextUtils:
    def test_is_chinese_true(self):
        assert is_chinese("我爱自然语言处理和机器学习")

    def test_is_chinese_false(self):
        assert not is_chinese("I love natural language processing")

    def test_is_chinese_mixed(self):
        # Majority Chinese
        assert is_chinese("我爱NLP和AI技术在中国的发展")

    def test_split_sentences_english(self):
        text = "Hello world. This is a test. How are you?"
        sentences = split_sentences_bilingual(text)
        assert len(sentences) == 3

    def test_split_sentences_chinese(self):
        text = "我喜欢编程。你呢？我也是！"
        sentences = split_sentences_bilingual(text)
        assert len(sentences) == 3

    def test_clean_text(self):
        text = "  Hello   world  \n\n  test  "
        cleaned = clean_text(text)
        assert cleaned == "Hello world test"

    def test_text_stats(self):
        text = "Hello world. This is a test."
        stats = text_stats(text)
        assert stats["char_count"] > 0
        assert stats["word_count"] == 6
        assert stats["sentence_count"] == 2
