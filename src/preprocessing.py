"""Text normalization helpers for CFPB complaint narratives."""

from __future__ import annotations

import html
import re
import unicodedata

PAYMENT_APPS = [
    "cash app", "cashapp", "venmo", "paypal", "zelle", "apple pay",
    "google pay", "samsung pay", "xoom", "skrill", "revolut", "wise",
    "transferwise", "payoneer", "remitly", "moneygram", "western union",
]
BANKS = [
    "wells fargo", "bank of america", "chase", "citibank", "citi",
    "capital one", "us bank", "usaa", "pnc", "td bank", "truist",
    "american express", "amex", "discover", "chime", "sofi", "varo",
    "metabank", "greendot", "green dot", "netspend", "money network",
]
CRYPTO_EXCHANGES = [
    "coinbase", "binance", "kraken", "gemini", "crypto.com", "robinhood",
]
MERCHANTS = ["walmart", "amazon", "ebay"]

def _compile_terms(terms: list[str]) -> re.Pattern:
    alternatives = "|".join(re.escape(x) for x in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)

CATEGORY_PATTERNS = {
    "payment_app": _compile_terms(PAYMENT_APPS),
    "bank": _compile_terms(BANKS),
    "crypto_exchange": _compile_terms(CRYPTO_EXCHANGES),
    "merchant": _compile_terms(MERCHANTS),
}

REDACTION_PATTERN = re.compile(r"\b[xX]{2,}\b")
MONEY_PATTERN = re.compile(r"(?:\{\$[\d,.]+\}|\$\s?[\d,.]+)")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)")
LONG_NUMBER_PATTERN = re.compile(r"(?<!\d)\d{6,}(?!\d)")
DATE_PATTERN = re.compile(r"\b(?:\d{1,2}[/-]){1,2}\d{2,4}\b")
SPACE_PATTERN = re.compile(r"\s+")

# High-recall filter: used only to define the scam/fraud working corpus.
SCAM_PATTERN = re.compile(
    r"\b(?:scam|fraud|fraudulent|unauthori[sz]ed|not authori[sz]ed|stolen|"
    r"hacked|account takeover|identity theft|phish(?:ing)?|spoof(?:ed|ing)?|"
    r"impersonat(?:e|ed|ion)|fake|deceiv(?:e|ed)|trick(?:ed)?|coerc(?:e|ed|ion)|"
    r"extort(?:ion)?|romance|investment scheme|ponzi|gift card|remote access|"
    r"puppy|rental scam|job scam|overpayment|counterfeit check|social engineering|"
    r"verification code|one[- ]time code|otp|send money|wire transfer|"
    r"received a suspicious (?:call|text|message)|unexpected (?:call|text|message))\b",
    re.IGNORECASE,
)

def normalize_complaint(text: object) -> str:
    """Normalize a complaint while preserving scam behavior and channel information."""
    value = html.unescape(str(text))
    value = unicodedata.normalize("NFKC", value).lower()
    value = REDACTION_PATTERN.sub(" redacted ", value)
    value = MONEY_PATTERN.sub(" money_amount ", value)
    value = URL_PATTERN.sub(" url ", value)
    value = EMAIL_PATTERN.sub(" email_address ", value)
    value = PHONE_PATTERN.sub(" phone_number ", value)
    value = LONG_NUMBER_PATTERN.sub(" long_number ", value)
    value = DATE_PATTERN.sub(" date_value ", value)
    for token, pattern in CATEGORY_PATTERNS.items():
        value = pattern.sub(f" {token} ", value)
    # Keep words such as bitcoin/ethereum/cryptocurrency: they are scam channels,
    # not merely company identities.
    return SPACE_PATTERN.sub(" ", value).strip()

def is_scam_candidate(text: object) -> bool:
    return bool(SCAM_PATTERN.search(str(text)))
