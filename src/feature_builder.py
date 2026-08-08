"""Build transparent structured signals and prompts for FLAN-T5."""

from __future__ import annotations

import re
from typing import Any

from src.red_flags import extract_red_flags, score_risk
from utils.helpers import format_red_flags, truncate_words


ARCHETYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("account takeover or unauthorized transaction", re.compile(
        r"\b(?:account takeover|hacked|unauthori[sz]ed|did not authori[sz]e|not mine|without my permission)\b", re.I
    )),
    ("identity theft or synthetic identity fraud", re.compile(
        r"\b(?:identity theft|stolen identity|opened .* account|credit report|social security)\b", re.I
    )),
    ("impersonation or phishing scam", re.compile(
        r"\b(?:impersonat|phish|spoof|claimed to be|pretend(?:ed)? to be|verification code|otp)\b", re.I
    )),
    ("fake check or overpayment scam", re.compile(
        r"\b(?:fake check|counterfeit check|cashier'?s check|overpayment|check bounced)\b", re.I
    )),
    ("investment or cryptocurrency scam", re.compile(
        r"\b(?:investment|crypto(?:currency)?|bitcoin|ethereum|guaranteed return|trading platform|profit)\b", re.I
    )),
    ("romance or relationship manipulation scam", re.compile(
        r"\b(?:romance|dating|boyfriend|girlfriend|fianc[eé]|relationship)\b", re.I
    )),
    ("remote-access or technical-support scam", re.compile(
        r"\b(?:remote access|screen share|anydesk|teamviewer|technical support|tech support)\b", re.I
    )),
    ("rental, marketplace, or purchase scam", re.compile(
        r"\b(?:rental|landlord|apartment|marketplace|buyer|seller|never received|not delivered|gift card)\b", re.I
    )),
    ("debt-collection or government impostor scam", re.compile(
        r"\b(?:debt collector|collection agency|irs|government agent|police|arrest|social security administration)\b", re.I
    )),
    ("money-transfer or payment-app scam", re.compile(
        r"\b(?:payment_app|wire transfer|send money|moneygram|western union)\b", re.I
    )),
]


def infer_provisional_archetype(text: object) -> str:
    """Assign a transparent provisional archetype for preliminary generation.

    This is intentionally rule based. It keeps the end-to-end demo runnable before the final
    Transformer classifier has been accepted and exported.
    """
    value = str(text)
    for label, pattern in ARCHETYPE_PATTERNS:
        if pattern.search(value):
            return label
    return "other suspected fraud or disputed transaction"

def clean_for_generation(text: str) -> str:
    """Normalize noisy CFPB redaction markers before generation."""

    value = str(text)

    # Collapse repeated "redacted" tokens.
    value = re.sub(
        r"\b(?:redacted\s+){1,}redacted\b",
        "[REDACTED]",
        value,
        flags=re.IGNORECASE,
    )

    # Collapse repeated [REDACTED] markers.
    value = re.sub(
        r"(?:\[REDACTED\]\s*){2,}",
        "[REDACTED] ",
        value,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace.
    value = re.sub(r"\s+", " ", value).strip()

    return value

def build_structured_record(row: Any, text_column: str, max_input_words: int) -> dict[str, Any]:
    """Convert one dataframe row into the structured generator input."""
    clean_text = clean_for_generation(
    truncate_words(
            row[text_column],
            max_input_words
        )
    )
    red_flags = extract_red_flags(clean_text)
    archetype = infer_provisional_archetype(clean_text)
    risk_level, risk_score = score_risk(red_flags, archetype)

    return {
        "clean_text": clean_text,
        "archetype": archetype,
        "archetype_source": "transparent rule-based preliminary label",
        "archetype_confidence": None,
        "red_flags": red_flags,
        "risk_level": risk_level,
        "risk_score": int(risk_score),
    }


def build_generation_prompt(record: dict[str, Any]) -> str:
    """Create a concise, grounded complaint-summary prompt."""

    return (
        "Summarize the following consumer complaint in two concise sentences "
        "for a fraud analyst. Do not invent facts.\n\n"
        f"Complaint: {record['clean_text']}\n\n"
        "Summary:"
    )
