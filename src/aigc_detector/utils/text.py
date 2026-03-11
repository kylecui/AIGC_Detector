import re

import nltk


def _ensure_punkt():
    """Ensure NLTK punkt tokenizer is available."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def is_chinese(text: str) -> bool:
    """Check if text contains predominantly Chinese characters."""
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    total_alpha = len(re.findall(r"[\w]", text))
    if total_alpha == 0:
        return False
    return chinese_chars / total_alpha > 0.3


def split_sentences_bilingual(text: str) -> list[str]:
    """
    Split text into sentences. Auto-detects language.
    Chinese: split on sentence-ending punctuation, preserving punctuation.
    English: use NLTK punkt tokenizer.
    """
    if is_chinese(text):
        # Chinese: split after sentence-ending punctuation, keeping the punctuation attached
        sentences = re.split(r"(?<=[。！？；\n])", text)
    else:
        _ensure_punkt()
        sentences = nltk.sent_tokenize(text)

    return [s.strip() for s in sentences if s.strip()]


def clean_text(text: str) -> str:
    """Basic text cleaning: normalize whitespace, strip boilerplate."""
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def text_stats(text: str) -> dict:
    """Compute basic text statistics."""
    words = text.split()
    chars = len(text)
    sentences = split_sentences_bilingual(text)
    return {
        "char_count": chars,
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
    }
