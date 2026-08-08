"""Validate generated summaries and provide a grounded fallback."""

from __future__ import annotations

import re


INSTRUCTION_ECHO_PATTERNS = [
    r"\bdo not invent facts\b",
    r"\bdescribe the suspected fraud pattern\b",
    r"\bdescribe the fraud pattern\b",
    r"\bsummarize the following consumer complaint\b",
    r"\bsummarize this consumer complaint\b",
    r"\bfor a fraud analyst\b",
    r"\ba fraud analyst\b",
]


def validate_generated_summary(
    summary: str,
) -> tuple[bool, list[str]]:
    """Check a generated summary for obvious generation failures.

    Returns:
        (is_valid, issues)
    """

    value = str(summary).strip()
    lower = value.lower()

    issues: list[str] = []

    # Empty output.
    if not value:
        issues.append("empty_output")

    # Very short outputs are rarely useful analyst summaries.
    if len(value.split()) < 3:
        issues.append("too_short")

    # Detect prompt/instruction copying.
    for pattern in INSTRUCTION_ECHO_PATTERNS:
        if re.search(pattern, lower, flags=re.IGNORECASE):
            issues.append("instruction_echo")
            break

    # Detect outputs dominated by CFPB redaction tokens.
    redacted_count = len(
        re.findall(
            r"\b(?:redacted|\[redacted\])\b",
            lower,
            flags=re.IGNORECASE,
        )
    )

    word_count = max(len(value.split()), 1)

    #if redacted_count >= 3:
        #issues.append("excessive_redaction_echo")

    if redacted_count / word_count >= 0.40:
        issues.append("redaction_dominated")

    # Detect pathological repetition such as:
    # "Do not invent facts. Do not invent facts."
    sentences = [
        sentence.strip().lower()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            value,
        )
        if sentence.strip()
    ]

    if (
        len(sentences) >= 2
        and len(set(sentences)) == 1
    ):
        issues.append("repeated_sentence")

    return len(issues) == 0, issues


def build_extractive_fallback(
    complaint: str,
    max_segments: int = 2,
    max_words: int = 45,
) -> str:
    """Create a concise, fraud-focused extractive fallback.

    The fallback uses only text from the original complaint.
    Long run-on sentences are divided into smaller segments before
    fraud relevance is scored.
    """

    value = str(complaint).strip()

    if not value:
        return ""

    fraud_keywords = {
        "fraud",
        "fraudulent",
        "scam",
        "scammed",
        "unauthorized",
        "transaction",
        "transactions",
        "charge",
        "charges",
        "charged",
        "transfer",
        "transferred",
        "wire",
        "payment",
        "money",
        "account",
        "card",
        "bank",
        "stolen",
        "lost",
        "refund",
        "refunded",
        "reimburse",
        "reimbursed",
        "dispute",
        "disputed",
        "withdrawal",
        "withdrawals",
        "identity",
        "check",
        "forged",
        "intercepted",
    }

    # ---------------------------------------------------------
    # Split complaint into smaller source segments
    # ---------------------------------------------------------

    # First split normally on sentence-ending punctuation.
    sentences = re.split(
        r"(?<=[.!?])\s+",
        value,
    )

    segments: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        words = sentence.split()

        # If the sentence is already reasonably short,
        # keep it as one candidate.
        if len(words) <= 40:
            segments.append(sentence)
            continue

        # CFPB narratives often contain very long run-on
        # sentences. Split them further at commas,
        # semicolons, colons, or common conjunctions.
        smaller_parts = re.split(
            r"\s*(?:[,;:])\s*|\s+\b(?:and then|then|but|however)\b\s+",
            sentence,
            flags=re.IGNORECASE,
        )

        for part in smaller_parts:
            part = part.strip()

            if len(part.split()) >= 5:
                segments.append(part)

    # ---------------------------------------------------------
    # Score candidate segments
    # ---------------------------------------------------------

    candidates = []

    for index, segment in enumerate(segments):
        words = segment.split()

        if len(words) < 5:
            continue

        redacted_count = len(
            re.findall(
                r"\b(?:redacted|\[redacted\])\b",
                segment,
                flags=re.IGNORECASE,
            )
        )

        redaction_ratio = (
            redacted_count
            / max(len(words), 1)
        )

        # Ignore mostly-redacted fragments.
        if redaction_ratio >= 0.40:
            continue

        segment_words = set(
            re.findall(
                r"\b\w+\b",
                segment.lower(),
            )
        )

        keyword_score = sum(
            1
            for keyword in fraud_keywords
            if keyword in segment_words
        )

        # Give a small bonus to earlier source segments
        # when fraud scores are otherwise equal.
        candidates.append(
            {
                "index": index,
                "segment": segment,
                "score": keyword_score,
            }
        )

    # ---------------------------------------------------------
    # Choose the most relevant source segments
    # ---------------------------------------------------------

    if candidates:

        ranked = sorted(
            candidates,
            key=lambda item: (
                -item["score"],
                item["index"],
            ),
        )

        selected = ranked[:max_segments]

        # Restore their original narrative order.
        selected = sorted(
            selected,
            key=lambda item: item["index"],
        )

        result = ". ".join(
            item["segment"].strip(" .")
            for item in selected
        )

        if result:
            result += "."

    else:
        result = value

    # ---------------------------------------------------------
    # Hard maximum length
    # ---------------------------------------------------------

    result_words = result.split()

    if len(result_words) > max_words:
        result = " ".join(
            result_words[:max_words]
        ).rstrip(" ,;:")

        if not result.endswith((".", "!", "?")):
            result += "."

    return result.strip()