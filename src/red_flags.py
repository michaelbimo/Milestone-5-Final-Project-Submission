"""Rule-based red-flag extraction and transparent risk scoring."""

from __future__ import annotations

import re

RED_FLAG_PATTERNS: dict[str, re.Pattern] = {
    "urgent or threatening language": re.compile(r"\b(?:urgent|immediately|right away|act now|threat|arrest|suspend|freeze)\b", re.I),
    "impersonation of a trusted organization": re.compile(r"\b(?:claimed to be|pretend(?:ed)? to be|impersonat|posing as|bank representative|bank employee|government agent|police|irs|social security)\b", re.I),
    "request for credentials or verification codes": re.compile(r"\b(?:password|passcode|pin|verification code|one[- ]time code|otp|security code|login)\b", re.I),
    "request to transfer money": re.compile(r"\b(?:wire|transfer|send money|payment_app|gift card|crypto(?:currency)?|bitcoin|ethereum)\b", re.I),
    "unexpected contact": re.compile(r"\b(?:unsolicited|unexpected|random (?:call|text|message)|contacted me|received a text|received a call)\b", re.I),
    "remote device access": re.compile(r"\b(?:remote access|screen share|anydesk|teamviewer|installed an app)\b", re.I),
    "investment return promise": re.compile(r"\b(?:guaranteed return|high return|double your money|investment opportunity|profit promised)\b", re.I),
    "romance or relationship manipulation": re.compile(r"\b(?:dating|romance|boyfriend|girlfriend|fianc[eé]|relationship)\b", re.I),
    "check overpayment or counterfeit check": re.compile(r"\b(?:overpayment|counterfeit check|fake check|cashier'?s check|check bounced)\b", re.I),
    "consumer denies authorizing the transaction": re.compile(r"\b(?:unauthori[sz]ed|did not authori[sz]e|not mine|without my permission)\b", re.I),
}

HIGH_SEVERITY_FLAGS = {
    "request for credentials or verification codes",
    "remote device access",
    "consumer denies authorizing the transaction",
    "request to transfer money",
}

def extract_red_flags(text: object) -> list[str]:
    value = str(text)
    return [name for name, pattern in RED_FLAG_PATTERNS.items() if pattern.search(value)]

def score_risk(red_flags: list[str], archetype: str = "") -> tuple[str, int]:
    score = len(red_flags)
    score += sum(flag in HIGH_SEVERITY_FLAGS for flag in red_flags)
    if re.search(r"account takeover|identity theft|investment|romance|impersonation", archetype, re.I):
        score += 1
    if score >= 5:
        return "High", score
    if score >= 2:
        return "Medium", score
    return "Low", score
