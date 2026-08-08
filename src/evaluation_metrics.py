"""Metrics for the final CFPB generative-model evaluation.

The CFPB complaint dataset does not include analyst-written reference summaries, so the
primary automatic metrics are factual-grounding proxies that can be computed from the source
complaint and the structured fields supplied to the generator. Human scores can be added through
the generated evaluation template.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")


def normalize_tokens(text: object) -> set[str]:
    """Return lowercase alphanumeric tokens for lightweight overlap checks."""
    return {token.lower() for token in _TOKEN_RE.findall(str(text))}


def non_empty(text: object) -> bool:
    """Whether a generated summary contains non-whitespace text."""
    return bool(str(text).strip())


def word_count(text: object) -> int:
    """Whitespace-delimited summary length."""
    return len(str(text).split())


def risk_level_mentioned(summary: object, risk_level: object) -> bool:
    """Check whether the supplied risk level is explicitly reflected in the output."""
    risk = str(risk_level).strip().lower()
    if not risk:
        return False
    return risk in str(summary).lower()


def red_flag_coverage(summary: object, red_flags: Iterable[object]) -> float:
    """Estimate how much of the structured red-flag evidence is reflected in the summary.

    Each red flag is represented by its informative content words. A flag is counted as covered
    when at least one non-trivial token from the flag appears in the generated summary. This is a
    transparent proxy, not a semantic-equivalence metric.
    """
    summary_tokens = normalize_tokens(summary)
    flags = [str(flag).strip() for flag in red_flags if str(flag).strip()]
    if not flags:
        return 1.0

    stop = {
        "a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or",
        "the", "to", "with", "consumer", "request", "language",
    }
    covered = 0
    for flag in flags:
        tokens = {t for t in normalize_tokens(flag) if len(t) >= 4 and t not in stop}
        if not tokens:
            continue
        if summary_tokens.intersection(tokens):
            covered += 1
    return covered / len(flags)


def archetype_keyword_coverage(summary: object, archetype: object) -> float:
    """Token-overlap proxy for whether the provisional archetype appears in the output."""
    archetype_tokens = {
        token for token in normalize_tokens(archetype)
        if len(token) >= 4 and token not in {"other", "suspected", "fraud", "transaction"}
    }
    if not archetype_tokens:
        return 1.0
    summary_tokens = normalize_tokens(summary)
    return len(archetype_tokens.intersection(summary_tokens)) / len(archetype_tokens)


def unsupported_number_rate(summary: object, source_text: object) -> float:
    """Fraction of numbers in the summary that do not appear in the source complaint.

    New amounts, dates, or counts are a useful hallucination warning sign. A rate of 0 is ideal.
    If the summary contains no numbers, the rate is 0.
    """
    summary_numbers = _NUMBER_RE.findall(str(summary))
    if not summary_numbers:
        return 0.0
    source_numbers = set(_NUMBER_RE.findall(str(source_text)))
    unsupported = sum(number not in source_numbers for number in summary_numbers)
    return unsupported / len(summary_numbers)


def source_token_support(summary: object, source_text: object) -> float:
    """Fraction of content tokens in the summary that also occur in the complaint.

    This is an intentionally simple lexical grounding proxy. It should be interpreted together
    with human factuality ratings because valid paraphrases can reduce this score.
    """
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
        "it", "of", "on", "or", "that", "the", "this", "to", "was", "were", "with",
    }
    summary_tokens = {t for t in normalize_tokens(summary) if len(t) >= 4 and t not in stop}
    if not summary_tokens:
        return 0.0
    source_tokens = normalize_tokens(source_text)
    return len(summary_tokens.intersection(source_tokens)) / len(summary_tokens)


def aggregate_metric(values: Iterable[float]) -> float:
    """Safe mean for metric lists."""
    values = list(values)
    if not values:
        return 0.0
    return float(np.mean(values))
