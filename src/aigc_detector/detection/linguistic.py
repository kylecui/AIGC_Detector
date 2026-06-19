"""Linguistic-stylistic feature extraction for AI text detection.

A CPU-only, dependency-free feature engine that captures "human writing
noise" signals orthogonal to the LM-probability features produced by
:class:`StatisticalFeatureExtractor`. No LM/GPU calls are made from this
module — :meth:`LinguisticFeatureExtractor.extract` accepts an optional
``token_log_probs`` array (typically produced upstream by
``StatisticalFeatureExtractor``) for the two token-probability features.

Feature groups
--------------
- **M1-M6**: micro / sentence-level signals (length burstiness, repetition,
  token-logprob distribution shape).
- **M7-M9**: micro / lexical signals (hedging, discourse templating,
  punctuation style).
- **S1-S2**: meso / paragraph-level signals (paragraph-length variance,
  template-structure score).
- **D1-D3**: macro / document-level signals (lexical diversity MTLD,
  authorial stance, readability).

References
----------
- DESIGN.md §4.x for the linguistic-feature specification.
- McCarthy & Jarvis (2010) for the MTLD algorithm.
- Flesch (1948) for Reading Ease.

All per-1000-char features are robust to the empty-text edge case
(return 0.0 rather than raising).
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from aigc_detector.utils.text import is_chinese, split_sentences_bilingual

logger = logging.getLogger(__name__)

# ======================================================================
# Lexicons (case-insensitive substring match)
# ======================================================================

# fmt: off
HEDGING_WORDS_EN: list[str] = [
    "appears to", "tends to", "seems to", "somewhat", "in our experience",
    "we found", "we find", "somewhat unexpectedly", "interestingly",
    "in practice", "to some extent", "in our observations",
    "a closer examination", "presumably", "arguably", "likely",
    "may", "might", "could", "possibly", "perhaps",
]

HEDGING_WORDS_ZH: list[str] = [
    "似乎", "往往", "我们发现", "出人意料", "实践中", "某种程度上",
    "据我们观察", "更仔细的观察", "可能", "也许", "或许", "大概",
    "不一定", "未必", "似乎可以",
]

AI_TEMPLATE_MARKERS_EN: list[str] = [
    "furthermore", "moreover", "in addition", "additionally",
    "therefore", "thus", "consequently", "as a result",
    "in conclusion", "to summarize", "firstly", "secondly", "finally",
]

AI_TEMPLATE_MARKERS_ZH: list[str] = [
    "此外", "另外", "因此", "综上", "总而言之", "首先", "其次", "最后",
    "由此可见", "不难看出",
]

PUNCTUATION_EN: list[str] = ["\u2014", "\u2013", "(", ")", ";"]  # em-dash, en-dash, parens, semicolon
PUNCTUATION_ZH: list[str] = ["\u2014\u2014", "\u2026\u2026", "\u3001", "\uff1b"]  # ——, ……, 、, ;

# Template-y openers for paragraph_template_score (S2)
TEMPLATE_OPENERS_EN: tuple[str, ...] = ("the", "this", "these", "we", "in", "a")
TEMPLATE_OPENERS_ZH: tuple[str, ...] = ("本文", "本研究", "通过", "为了", "基于")

# First-person / opinion markers for authorial_stance_score (D2)
FIRST_PERSON_EN: list[str] = ["we", "our", "us", "i", "my", "me"]
FIRST_PERSON_ZH: list[str] = ["我们", "笔者", "本人", "我"]
# fmt: on


# ======================================================================
# Feature names (canonical order — keep in sync with LinguisticFeatures)
# ======================================================================

FEATURE_NAMES: list[str] = [
    "sentence_length_burstiness",
    "sentence_length_cv",
    "sentence_length_gini",
    "syntactic_repetition_rate",
    "token_logprob_skew",
    "token_logprob_high_prob_frac",
    "hedging_density",
    "discourse_templating",
    "punctuation_style",
    "paragraph_length_variance",
    "paragraph_template_score",
    "lexical_diversity_mtld",
    "authorial_stance_score",
    "readability_index",
]

# Common English stopwords used by syntactic_repetition_rate (M4).
# Small, conservative list — we only need a rough "opener token" filter.
_STOPWORDS_EN: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "of", "in", "on", "at",
        "to", "for", "with", "by", "from", "as", "is", "are", "was",
        "were", "be", "been", "being", "that", "this", "these", "those",
        "it", "its", "i", "we", "you", "they", "he", "she",
        "has", "have", "had", "do", "does", "did", "will", "would",
        "can", "could", "should", "may", "might", "must", "shall",
        "not", "no", "so", "than", "then", "there", "here",
    }
)


# ======================================================================
# LinguisticFeatures dataclass
# ======================================================================


@dataclass
class LinguisticFeatures:
    """Container for the 14 linguistic-stylistic features.

    Some fields may be ``float("nan")`` when the input text is too short
    or when a required input (e.g. ``token_log_probs``) is not provided.
    """

    # Micro / sentence-level
    sentence_length_burstiness: float       # M1: (std-mean)/(std+mean) of sentence lengths
    sentence_length_cv: float               # M2: std/mean of sentence lengths
    sentence_length_gini: float             # M3: Gini coefficient of sentence lengths
    syntactic_repetition_rate: float        # M4: avg pairwise Jaccard of sentence opener token sets
    token_logprob_skew: float               # M5: skewness of per-token log-probs (NaN if no probs)
    token_logprob_high_prob_frac: float     # M6: fraction of tokens with log-prob > -1.0 (NaN if no probs)
    hedging_density: float                  # M7: hedging word hits per 1000 chars
    discourse_templating: float             # M8: AI-template discourse marker hits per 1000 chars
    punctuation_style: float                # M9: combined em-dash/parenthetical/semicolon per 1000 chars
    # Meso / paragraph-level
    paragraph_length_variance: float        # S1: log-variance of paragraph char counts
    paragraph_template_score: float         # S2: heuristic 0-1 score for "Bg->Method->Result->Conclusion" symmetry
    # Macro / document-level
    lexical_diversity_mtld: float           # D1: MTLD (Measure of Textual Lexical Diversity)
    authorial_stance_score: float           # D2: first-person + opinion + hedging combined density
    readability_index: float                # D3: Flesch-Kincaid (en) / avg-sentence-length proxy (zh)

    def to_array(self) -> np.ndarray:
        """Return features as a 1-D float64 numpy array in canonical order.

        NaN-safe: callers (e.g. the classifier pipeline) are expected to
        handle NaNs via ``SimpleImputer``.
        """
        return np.array(
            [
                self.sentence_length_burstiness,
                self.sentence_length_cv,
                self.sentence_length_gini,
                self.syntactic_repetition_rate,
                self.token_logprob_skew,
                self.token_logprob_high_prob_frac,
                self.hedging_density,
                self.discourse_templating,
                self.punctuation_style,
                self.paragraph_length_variance,
                self.paragraph_template_score,
                self.lexical_diversity_mtld,
                self.authorial_stance_score,
                self.readability_index,
            ],
            dtype=np.float64,
        )

    def to_dict(self) -> dict:
        """Return features as a plain dict (preserves field order)."""
        return asdict(self)


def _nan_features() -> LinguisticFeatures:
    """Build a LinguisticFeatures with every field set to NaN."""
    nan = float("nan")
    return LinguisticFeatures(
        sentence_length_burstiness=nan,
        sentence_length_cv=nan,
        sentence_length_gini=nan,
        syntactic_repetition_rate=nan,
        token_logprob_skew=nan,
        token_logprob_high_prob_frac=nan,
        hedging_density=nan,
        discourse_templating=nan,
        punctuation_style=nan,
        paragraph_length_variance=nan,
        paragraph_template_score=nan,
        lexical_diversity_mtld=nan,
        authorial_stance_score=nan,
        readability_index=nan,
    )


# ======================================================================
# LinguisticFeatureExtractor
# ======================================================================


class LinguisticFeatureExtractor:
    """Extract the 14 linguistic-stylistic features from text.

    Pure-CPU, dependency-free. No LM call is performed inside this class;
    the two token-probability features (M5/M6) require an optional
    ``token_log_probs`` argument typically supplied by
    :class:`StatisticalFeatureExtractor`.

    Parameters
    ----------
    min_text_chars : int
        Inputs shorter than this threshold return an all-NaN
        :class:`LinguisticFeatures` (the signals are too noisy on short
        text). Default 200.
    """

    def __init__(self, min_text_chars: int = 200):
        self.min_text_chars = int(min_text_chars)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        lang: str | None = None,
        token_log_probs: list[float] | np.ndarray | None = None,
    ) -> LinguisticFeatures:
        """Extract linguistic features from *text*.

        Parameters
        ----------
        text : str
            Input text (paragraph or full document).
        lang : str or None
            ``"en"`` / ``"zh"``; if None, inferred via :func:`is_chinese`.
        token_log_probs : sequence of float or None
            Per-token log-probabilities from a reference LM. If None,
            M5/M6 are set to NaN.

        Returns
        -------
        LinguisticFeatures
            14-field feature container (some fields NaN for short text).
        """
        # Empty / whitespace-only text: never crash.
        if text is None or len(text.strip()) == 0:
            return _nan_features()

        if lang is None:
            lang = "zh" if is_chinese(text) else "en"

        # Short text guard: all-NaN result (matches spec).
        if len(text) < self.min_text_chars:
            return _nan_features()

        sentences = split_sentences_bilingual(text)
        char_count = len(text)

        m1, m2, m3 = self._sentence_length_stats(sentences)
        m4 = self._syntactic_repetition_rate(sentences, lang)
        m5, m6 = self._token_logprob_features(token_log_probs)
        m7 = self._density_from_lexicon(text, char_count, lang, hedging=True)
        m8 = self._density_from_lexicon(text, char_count, lang, hedging=False, templating=True)
        m9 = self._punctuation_density(text, char_count, lang)
        s1 = self._paragraph_length_variance(text)
        s2 = self._paragraph_template_score(text, lang)
        d1 = self._lexical_diversity_mtld(text, lang)
        # Hedging count (raw) reused for the stance composite.
        hedge_count = self._count_lexicon(text, lang, hedging=True)
        d2 = self._authorial_stance_density(text, char_count, lang, hedge_count)
        d3 = self._readability_index(sentences, text, lang)

        return LinguisticFeatures(
            sentence_length_burstiness=m1,
            sentence_length_cv=m2,
            sentence_length_gini=m3,
            syntactic_repetition_rate=m4,
            token_logprob_skew=m5,
            token_logprob_high_prob_frac=m6,
            hedging_density=m7,
            discourse_templating=m8,
            punctuation_style=m9,
            paragraph_length_variance=s1,
            paragraph_template_score=s2,
            lexical_diversity_mtld=d1,
            authorial_stance_score=d2,
            readability_index=d3,
        )

    # ------------------------------------------------------------------
    # M1-M3: sentence-length statistics
    # ------------------------------------------------------------------

    def _sentence_length_stats(self, sentences: list[str]) -> tuple[float, float, float]:
        """Return (burstiness, cv, gini) of sentence char-lengths."""
        if len(sentences) < 2:
            return float("nan"), float("nan"), float("nan")

        lengths = np.array([len(s) for s in sentences], dtype=np.float64)
        mean = float(lengths.mean())
        std = float(lengths.std())  # population std, matches StatisticalFeatures convention

        # M1 burstiness: (std - mean) / (std + mean + eps) — same as StatisticalFeatures._burstiness.
        burstiness = (std - mean) / (std + mean + 1e-8)

        # M2 coefficient of variation.
        cv = std / (abs(mean) + 1e-8) if abs(mean) > 1e-8 else 0.0

        # M3 Gini coefficient.
        gini = self._gini(lengths)

        return burstiness, cv, gini

    @staticmethod
    def _gini(values: np.ndarray) -> float:
        """Standard Gini coefficient over non-negative values.

        ``G = (2 * sum(i * x_i) / (n * sum(x_i))) - (n + 1) / n``
        after sorting ascending. Returns NaN if the total is zero.
        """
        values = np.sort(np.asarray(values, dtype=np.float64))
        n = values.size
        if n == 0:
            return float("nan")
        total = float(values.sum())
        if total <= 0:
            return float("nan")
        weighted = float(np.sum(np.arange(1, n + 1, dtype=np.float64) * values))
        return (2.0 * weighted / (n * total)) - (n + 1.0) / n

    # ------------------------------------------------------------------
    # M4: syntactic_repetition_rate (avg pairwise Jaccard of openers)
    # ------------------------------------------------------------------

    def _syntactic_repetition_rate(self, sentences: list[str], lang: str) -> float:
        """Average pairwise Jaccard similarity of per-sentence opener token sets."""
        if len(sentences) < 2:
            return float("nan")

        opener_sets: list[set[str]] = []
        for sent in sentences:
            tokens = self._opener_tokens(sent, lang, k=5)
            if tokens:
                opener_sets.append(set(tokens))

        if len(opener_sets) < 2:
            return float("nan")

        total = 0.0
        pairs = 0
        for i in range(len(opener_sets)):
            for j in range(i + 1, len(opener_sets)):
                a, b = opener_sets[i], opener_sets[j]
                union = len(a | b)
                if union == 0:
                    continue
                total += len(a & b) / union
                pairs += 1

        return total / pairs if pairs > 0 else 0.0

    def _opener_tokens(self, sentence: str, lang: str, k: int = 5) -> list[str]:
        """First *k* non-stopword tokens of a sentence, lowercased.

        English: whitespace tokens, stopwords filtered.
        Chinese: per-character tokens (no whitespace), all kept — Chinese
        has no equivalent stopword set in this conservative implementation.
        """
        if lang == "zh":
            # Drop punctuation, keep CJK and ASCII letters/digits.
            chars = re.findall(r"[\u4e00-\u9fff a-zA-Z0-9]", sentence.lower())
            tokens = [c for c in chars if c.strip()]
            return tokens[:k]
        # English
        raw = re.findall(r"[a-zA-Z'][a-zA-Z'-]*", sentence.lower())
        tokens = [t for t in raw if t not in _STOPWORDS_EN]
        return tokens[:k]

    # ------------------------------------------------------------------
    # M5, M6: token log-prob distribution shape
    # ------------------------------------------------------------------

    def _token_logprob_features(
        self,
        token_log_probs: list[float] | np.ndarray | None,
    ) -> tuple[float, float]:
        """Return (skewness, high-prob fraction) or (NaN, NaN) if not provided."""
        if token_log_probs is None:
            return float("nan"), float("nan")

        arr = np.asarray(token_log_probs, dtype=np.float64)
        if arr.size == 0:
            return float("nan"), float("nan")

        # M5 skewness. Prefer scipy.stats.skew (bias=True for population skew).
        skew_val = self._skewness(arr)

        # M6 fraction of tokens with log-prob > -1.0 (i.e. prob > e^-1 ~= 0.368).
        high_prob_frac = float((arr > -1.0).mean())

        return skew_val, high_prob_frac

    @staticmethod
    def _skewness(arr: np.ndarray) -> float:
        """Population skewness (Fisher-Pearson with bias correction off).

        Uses scipy.stats.skew when available; falls back to a manual
        computation matching scipy's default (bias=True) definition.
        """
        try:
            from scipy.stats import skew as _scipy_skew

            return float(_scipy_skew(arr, bias=True))
        except Exception:  # pragma: no cover — scipy ships with sklearn dep tree
            n = arr.size
            if n < 3:
                return float("nan")
            mean = arr.mean()
            std = arr.std()
            if std < 1e-12:
                return 0.0
            return float(np.mean(((arr - mean) / std) ** 3))

    # ------------------------------------------------------------------
    # M7, M8: lexicon densities
    # ------------------------------------------------------------------

    def _count_lexicon(self, text: str, lang: str, hedging: bool, templating: bool = False) -> int:
        """Count substring occurrences of the requested lexicon.

        Exactly one of ``hedging`` / ``templating`` must be True.
        Case-insensitive for English; substring match for Chinese.
        """
        if hedging:
            words = HEDGING_WORDS_EN if lang == "en" else HEDGING_WORDS_ZH
        elif templating:
            words = AI_TEMPLATE_MARKERS_EN if lang == "en" else AI_TEMPLATE_MARKERS_ZH
        else:
            return 0

        if lang == "en":
            haystack = text.lower()
            return sum(haystack.count(w) for w in words)
        return sum(text.count(w) for w in words)

    def _density_from_lexicon(
        self,
        text: str,
        char_count: int,
        lang: str,
        hedging: bool,
        templating: bool = False,
    ) -> float:
        """Lexicon hits per 1000 characters (0.0 on empty text)."""
        if char_count == 0:
            return 0.0
        count = self._count_lexicon(text, lang, hedging=hedging, templating=templating)
        return count * 1000.0 / char_count

    # ------------------------------------------------------------------
    # M9: punctuation style density
    # ------------------------------------------------------------------

    @staticmethod
    def _punctuation_density(text: str, char_count: int, lang: str) -> float:
        """Combined em-dash / parenthetical / semicolon count per 1000 chars."""
        if char_count == 0:
            return 0.0
        marks = PUNCTUATION_EN if lang == "en" else PUNCTUATION_ZH
        count = sum(text.count(m) for m in marks)
        return count * 1000.0 / char_count

    # ------------------------------------------------------------------
    # S1: paragraph length variance (log1p of variance)
    # ------------------------------------------------------------------

    @staticmethod
    def _paragraph_length_variance(text: str) -> float:
        """log1p(var(paragraph_char_counts)); NaN if fewer than 2 paragraphs."""
        paragraphs = LinguisticFeatureExtractor._split_paragraphs(text)
        if len(paragraphs) < 2:
            return float("nan")
        lengths = np.array([len(p) for p in paragraphs], dtype=np.float64)
        var = float(np.var(lengths))
        return float(math.log1p(max(var, 0.0)))

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Split on double-newline, falling back to single-newline."""
        if "\n\n" in text:
            parts = text.split("\n\n")
        else:
            parts = text.split("\n")
        return [p.strip() for p in parts if p.strip()]

    # ------------------------------------------------------------------
    # S2: paragraph template score (heuristic 0-1)
    # ------------------------------------------------------------------

    def _paragraph_template_score(self, text: str, lang: str) -> float:
        """Crude 0-1 score for Background->Method->Result->Conclusion symmetry.

        Returns 0.0 when there are fewer than 3 paragraphs (insufficient
        signal). Higher scores indicate more templated / AI-like structure.
        """
        paragraphs = self._split_paragraphs(text)
        if len(paragraphs) < 3:
            return 0.0

        openers = TEMPLATE_OPENERS_EN if lang == "en" else TEMPLATE_OPENERS_ZH

        # Per-paragraph: does it start with a template-y opener?
        opener_hits = 0
        signatures: list[frozenset[str]] = []
        for para in paragraphs:
            first_sentence = self._first_sentence(para)
            opener_tokens = self._opener_tokens(first_sentence, lang, k=3)
            signatures.append(frozenset(opener_tokens))
            if opener_tokens and opener_tokens[0] in openers:
                opener_hits += 1

        # Topic signature similarity between consecutive paragraphs.
        similar_pairs = 0
        for i in range(len(signatures) - 1):
            a, b = signatures[i], signatures[i + 1]
            union = len(a | b)
            if union == 0:
                continue
            jacc = len(a & b) / union
            if jacc >= 0.5:  # TODO: calibrate threshold against labelled data.
                similar_pairs += 1

        n = len(paragraphs)
        # Blend opener hit-rate and consecutive-similarity rate, both in [0, 1].
        opener_rate = opener_hits / n
        similarity_rate = similar_pairs / max(n - 1, 1)
        return float(0.5 * opener_rate + 0.5 * similarity_rate)

    @staticmethod
    def _first_sentence(paragraph: str) -> str:
        """Best-effort extraction of the first sentence of a paragraph."""
        sentences = split_sentences_bilingual(paragraph)
        return sentences[0] if sentences else paragraph.strip()

    # ------------------------------------------------------------------
    # D1: lexical diversity (MTLD)
    # ------------------------------------------------------------------

    def _lexical_diversity_mtld(self, text: str, lang: str) -> float:
        """MTLD (McCarthy, 2005). NaN if fewer than 10 tokens."""
        tokens = self._tokenize(text, lang)
        if len(tokens) < 10:
            return float("nan")
        # Cap for performance on very long inputs.
        if len(tokens) > 5000:
            tokens = tokens[:5000]
        forward = self._mtld_directional(tokens)
        backward = self._mtld_directional(tokens[::-1])
        return float((forward + backward) / 2.0)

    def _mtld_directional(self, tokens: list[str], factor_threshold: float = 0.72) -> float:
        """One-direction MTLD (McCarthy & Jarvis, 2010).

        Walks *tokens*, counting "factors" — runs whose type-token ratio
        drops from 1.0 to or below *factor_threshold*. A trailing partial
        factor is converted to a fractional factor via the linearised
        progress of TTR between 1.0 and the threshold. MTLD is then
        ``total_tokens / (n_factors + partial_proportion)``.

        ``factor_threshold`` default 0.72 follows the original paper.
        """
        factors = 0
        total_tokens = 0
        current_factor_len = 0
        types: set[str] = set()

        for token in tokens:
            types.add(token)
            current_factor_len += 1
            total_tokens += 1
            ttr = len(types) / current_factor_len
            if ttr <= factor_threshold:
                factors += 1
                types = set()
                current_factor_len = 0

        # No tokens at all (should not happen — guarded upstream).
        if total_tokens == 0:
            return 0.0

        # Trailing partial factor: convert to fractional progress.
        if current_factor_len == 0:
            # Ended exactly on a factor boundary.
            if factors == 0:
                return float(total_tokens)  # TTR never dropped below threshold.
            return float(total_tokens / factors)

        ttr_final = len(types) / current_factor_len
        # Linearised progress from TTR=1.0 (start of factor) to TTR=threshold.
        progress = (1.0 - ttr_final) / (1.0 - factor_threshold)
        progress = max(0.0, min(progress, 1.0))

        denominator = factors + progress
        if denominator < 1e-8:
            return float(total_tokens)
        return float(total_tokens / denominator)

    @staticmethod
    def _tokenize(text: str, lang: str) -> list[str]:
        """Whitespace tokens for en; character-level tokens for zh."""
        if lang == "zh":
            return re.findall(r"[\u4e00-\u9fff]", text)
        return text.lower().split()

    # ------------------------------------------------------------------
    # D2: authorial stance density (first-person + hedging)
    # ------------------------------------------------------------------

    def _authorial_stance_density(
        self,
        text: str,
        char_count: int,
        lang: str,
        hedging_count: int,
    ) -> float:
        """Combined first-person + hedging density per 1000 chars."""
        if char_count == 0:
            return 0.0
        first_person_words = FIRST_PERSON_EN if lang == "en" else FIRST_PERSON_ZH
        if lang == "en":
            haystack = text.lower()
            # Match as whole words to avoid counting substrings inside other words.
            fp_count = sum(len(re.findall(rf"\b{re.escape(w)}\b", haystack)) for w in first_person_words)
        else:
            fp_count = sum(text.count(w) for w in first_person_words)

        return (fp_count + hedging_count) * 1000.0 / char_count

    # ------------------------------------------------------------------
    # D3: readability index
    # ------------------------------------------------------------------

    def _readability_index(self, sentences: list[str], text: str, lang: str) -> float:
        """Flesch Reading Ease (en) / negative-avg-sentence-length proxy (zh)."""
        if len(sentences) < 2:
            return float("nan")

        if lang == "zh":
            avg_len = float(np.mean([len(s) for s in sentences]))
            return -1.0 * avg_len

        words = text.split()
        n_words = len(words)
        n_sents = len(sentences)
        if n_words == 0 or n_sents == 0:
            return float("nan")

        syllables = sum(self._count_syllables(w) for w in words)
        words_per_sentence = n_words / n_sents
        syllables_per_word = syllables / n_words
        return 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word

    @staticmethod
    def _count_syllables(word: str) -> int:
        """Rough vowel-group syllable counter (lower bound of 1 per word)."""
        word = re.sub(r"[^a-zA-Z]", "", word.lower())
        if not word:
            return 0
        groups = re.findall(r"[aeiouy]+", word)
        count = len(groups)
        # Silent trailing 'e' correction.
        if word.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)


# ======================================================================
# LinguisticClassifier wrapper
# ======================================================================


class LinguisticClassifier:
    """XGBoost classifier on top of :class:`LinguisticFeatures`.

    Mirrors :class:`StatisticalClassifier` API. The pipeline is
    ``SimpleImputer(median) -> StandardScaler -> XGBClassifier`` so that
    NaN values in the feature matrix (common for short text) are imputed
    rather than dropped.

    Parameters
    ----------
    backend : str
        Kept for API symmetry with :class:`StatisticalClassifier`; only
        ``"xgboost"`` is supported.
    """

    FEATURE_NAMES: list[str] = FEATURE_NAMES

    def __init__(self, backend: str = "xgboost"):
        if backend != "xgboost":
            raise ValueError(f"LinguisticClassifier only supports backend='xgboost', got {backend!r}")
        self.backend = backend
        self._pipeline: Pipeline | None = None
        self.threshold: float = 0.5
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
        )
        self._pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", clf),
            ]
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        features: np.ndarray | list[LinguisticFeatures],
        labels: np.ndarray,
    ) -> dict:
        """Train the classifier.

        Parameters
        ----------
        features : array-like of shape (n_samples, 14) or list of LinguisticFeatures
        labels : array-like of shape (n_samples,) — 0 = human, 1 = AI

        Returns
        -------
        dict with ``backend``, ``train_accuracy``, ``n_samples``.
        """
        x_train = self._ensure_array(features)
        self._pipeline.fit(x_train, labels)
        train_acc = self._pipeline.score(x_train, labels)
        logger.info("LinguisticClassifier trained: backend=%s, accuracy=%.4f", self.backend, train_acc)
        return {"backend": self.backend, "train_accuracy": train_acc, "n_samples": len(labels)}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, features: np.ndarray | LinguisticFeatures) -> dict:
        """Predict on a single sample or batch.

        Returns
        -------
        dict with ``label``, ``p_ai``, ``confidence`` for single samples
        (or ``labels`` / ``p_ai`` / ``confidences`` lists for batches).
        """
        if self._pipeline is None:
            raise RuntimeError("Classifier not trained. Call .fit() or .load() first.")

        x_arr = self._ensure_array(features)
        proba = self._pipeline.predict_proba(x_arr)  # (n, 2): col 0 = human, col 1 = AI
        p_ai = float(proba[0, 1]) if x_arr.shape[0] == 1 else proba[:, 1].tolist()

        if x_arr.shape[0] == 1:
            label = "ai" if p_ai > self.threshold else "human"
            confidence = max(p_ai, 1.0 - p_ai)
            return {"label": label, "p_ai": p_ai, "confidence": confidence}

        labels = ["ai" if p > self.threshold else "human" for p in p_ai]
        confidences = [max(p, 1.0 - p) for p in p_ai]
        return {"labels": labels, "p_ai": p_ai, "confidences": confidences}

    def set_threshold(self, threshold: float) -> None:
        """Override the AI decision threshold for inference."""
        self.threshold = float(threshold)
        logger.info("LinguisticClassifier threshold updated: %.6f", self.threshold)

    def predict_proba(self, features: np.ndarray | list[LinguisticFeatures]) -> np.ndarray:
        """Return raw probability array of shape (n_samples, 2)."""
        x_arr = self._ensure_array(features)
        return self._pipeline.predict_proba(x_arr)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the trained pipeline to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self._pipeline, "backend": self.backend}, path, compress=3)
        logger.info("LinguisticClassifier saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load a trained pipeline from disk."""
        path = Path(path)
        data = joblib.load(path)
        self._pipeline = data["pipeline"]
        self.backend = data["backend"]
        logger.info("LinguisticClassifier loaded from %s (backend=%s)", path, self.backend)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_array(features) -> np.ndarray:
        """Convert various input types to a 2-D numpy array (n_samples, 14)."""
        if isinstance(features, LinguisticFeatures):
            return features.to_array().reshape(1, -1)
        if isinstance(features, list) and features and isinstance(features[0], LinguisticFeatures):
            return np.stack([f.to_array() for f in features])
        arr = np.asarray(features, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr


# ======================================================================
# LinguisticDiagnostics
# ======================================================================


@dataclass
class LinguisticDiagnostics:
    """Human-likeness diagnostics derived from :class:`LinguisticFeatures`.

    Scores are 0-10 for the three levels (micro/meso/macro); the
    composite :attr:`human_likeness_score` is 0-100.
    """

    micro_score: float       # 0-10 (sentence-level human-likeness)
    meso_score: float        # 0-10 (paragraph-level)
    macro_score: float       # 0-10 (document-level)
    top_signals: list[str]   # human-readable feature names that triggered
    human_likeness_score: float  # 0-100 composite

    @classmethod
    def from_features(cls, features: LinguisticFeatures, lang: str) -> LinguisticDiagnostics:
        """Compute diagnostics from extracted features.

        Heuristic binning — bin edges are documented inline and should be
        calibrated against labelled data before being used in production.
        """
        micro = cls._micro_score(features)
        meso = cls._meso_score(features)
        macro = cls._macro_score(features)
        signals = cls._top_signals(features)
        composite = (0.4 * micro + 0.3 * meso + 0.3 * macro) * 10.0
        return cls(
            micro_score=micro,
            meso_score=meso,
            macro_score=macro,
            top_signals=signals,
            human_likeness_score=composite,
        )

    # ------------------------------------------------------------------
    # Level scorers (0-10, higher = more human-like)
    # ------------------------------------------------------------------

    @staticmethod
    def _safe(x: float, default: float = 0.0) -> float:
        return default if math.isnan(x) else x

    @classmethod
    def _micro_score(cls, features: LinguisticFeatures) -> float:
        """Sentence-level signals: burstiness/CV (variety = human),
        hedging density, punctuation style.
        """
        score = 0.0
        cv = cls._safe(features.sentence_length_cv)
        burst = cls._safe(features.sentence_length_burstiness)
        hedge = cls._safe(features.hedging_density)
        punct = cls._safe(features.punctuation_style)
        # TODO: bin edges are heuristic — recalibrate on labelled data.
        if cv > 0.6:
            score += 2.5
        elif cv > 0.3:
            score += 1.25
        if burst > 0.0:
            score += 1.5
        if hedge > 5.0:
            score += 3.0
        elif hedge > 1.0:
            score += 1.5
        if punct > 2.0:
            score += 2.0
        elif punct > 0.5:
            score += 1.0
        return min(score, 10.0)

    @classmethod
    def _meso_score(cls, features: LinguisticFeatures) -> float:
        """Paragraph-level signals: high paragraph_length_variance = human,
        low template score = human.
        """
        score = 0.0
        var = cls._safe(features.paragraph_length_variance)
        # template_score is in [0, 1]; NaN must NOT be conflated with 0.0
        # (which would otherwise always look "very human"). Skip the
        # component entirely when the feature is NaN.
        if not math.isnan(features.paragraph_template_score):
            template = features.paragraph_template_score
            # TODO: bin edges are heuristic.
            if var > 4.0:
                score += 4.0
            elif var > 1.0:
                score += 2.0
            if template < 0.2:
                score += 6.0
            elif template < 0.4:
                score += 3.0
        else:
            # Only the variance component is usable.
            # TODO: bin edges are heuristic.
            if var > 4.0:
                score += 4.0
            elif var > 1.0:
                score += 2.0
        return min(score, 10.0)

    @classmethod
    def _macro_score(cls, features: LinguisticFeatures) -> float:
        """Document-level signals: high MTLD = human, high authorial stance = human,
        readability within human range.
        """
        score = 0.0
        mtld = cls._safe(features.lexical_diversity_mtld)
        stance = cls._safe(features.authorial_stance_score)
        readability = cls._safe(features.readability_index)
        # TODO: bin edges are heuristic.
        if mtld > 60.0:
            score += 3.0
        elif mtld > 30.0:
            score += 1.5
        if stance > 5.0:
            score += 4.0
        elif stance > 1.0:
            score += 2.0
        # Human academic prose typically scores 30-60 on Flesch Reading Ease.
        if 20.0 < readability < 70.0:
            score += 3.0
        elif 10.0 < readability < 80.0:
            score += 1.5
        return min(score, 10.0)

    @classmethod
    def _top_signals(cls, features: LinguisticFeatures) -> list[str]:
        """Up to 3 feature names that strongly indicate "human".

        Thresholds are intentionally conservative — only flag features
        whose value clearly exceeds the human-like baseline.
        """
        signals: list[str] = []
        if not math.isnan(features.hedging_density) and features.hedging_density > 5.0:
            signals.append("hedging_density")
        if not math.isnan(features.sentence_length_cv) and features.sentence_length_cv > 0.6:
            signals.append("sentence_length_cv")
        if not math.isnan(features.lexical_diversity_mtld) and features.lexical_diversity_mtld > 60.0:
            signals.append("lexical_diversity_mtld")
        if not math.isnan(features.authorial_stance_score) and features.authorial_stance_score > 10.0:
            signals.append("authorial_stance_score")
        if not math.isnan(features.paragraph_length_variance) and features.paragraph_length_variance > 4.0:
            signals.append("paragraph_length_variance")
        return signals[:3]


# ======================================================================
# Module-level sanity check
# ======================================================================

# Verify that FEATURE_NAMES stays in sync with the dataclass field order.
assert [f.name for f in fields(LinguisticFeatures)] == FEATURE_NAMES, (
    "FEATURE_NAMES is out of sync with LinguisticFields field order"
)
assert len(FEATURE_NAMES) == 14, f"Expected 14 feature names, got {len(FEATURE_NAMES)}"
