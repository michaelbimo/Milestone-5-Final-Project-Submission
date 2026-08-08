"""Build transparent structured signals and prompts for FLAN-T5."""

from __future__ import annotations

import re
from typing import Any

from src.red_flags import (
    extract_red_flags,
    score_risk,
)

from utils.helpers import (
    truncate_words,
)


# ============================================================
# Archetype evidence patterns
# ============================================================


UNAUTHORIZED_PATTERN = re.compile(
    r"\b(?:"
    r"unauthori[sz]ed|"
    r"did not authori[sz]e|"
    r"never authori[sz]ed|"
    r"without my permission|"
    r"without my consent|"
    r"not mine|"
    r"account drained|"
    r"fraudulent withdrawals?|"
    r"fraudulent transactions?|"
    r"unknown transactions?|"
    r"unrecognized transactions?|"
    r"did not recognize (?:the )?(?:transaction|transactions|charge|charges)|"
    r"transactions? (?:that )?(?:i am|i'm|im) unaware of|"
    r"charges? (?:that )?(?:i am|i'm|im) unaware of"
    r")\b",
    re.I,
)


IDENTITY_THEFT_PATTERN = re.compile(
    r"\b(?:"
    r"identity theft|"
    r"stolen identity|"
    r"someone used my identity|"
    r"account opened in my name|"
    r"accounts? opened without my knowledge|"
    r"credit account opened without my permission|"
    r"using my social security number without my permission"
    r")\b",
    re.I,
)


IMPERSONATION_ACTION_PATTERN = re.compile(
    r"\b(?:"
    r"claimed to be|"
    r"claiming to be|"
    r"pretend(?:ed|ing)? to be|"
    r"posed as|"
    r"posing as|"
    r"impersonat(?:ed|ing|ion)|"
    r"said (?:he|she|they) (?:was|were) from|"
    r"fake caller id"
    r")\b",
    re.I,
)


GOVERNMENT_ENTITY_PATTERN = re.compile(
    r"\b(?:"
    r"irs|"
    r"internal revenue service|"
    r"social security administration|"
    r"government|"
    r"police department|"
    r"law enforcement"
    r")\b",
    re.I,
)


TRUSTED_ORG_PATTERN = re.compile(
    r"\b(?:"
    r"bank|"
    r"credit union|"
    r"fraud department|"
    r"customer service|"
    r"utility|"
    r"power company|"
    r"electric company"
    r")\b",
    re.I,
)


PHISHING_PATTERN = re.compile(
    r"\b(?:"
    r"phishing|"
    r"spoof(?:ed|ing)?|"
    r"verification code|"
    r"one[- ]time code|"
    r"otp"
    r")\b",
    re.I,
)


FAKE_CHECK_PATTERN = re.compile(
    r"\b(?:"
    r"fake check|"
    r"counterfeit check|"
    r"forged check|"
    r"check bounced|"
    r"check was forged|"
    r"check was counterfeit|"
    r"check (?:was|is) not good|"
    r"unauthorized endorsement"
    r")\b",
    re.I,
)


CASHIERS_CHECK_PATTERN = re.compile(
    r"\bcashier'?s check\b",
    re.I,
)


CHECK_PROBLEM_PATTERN = re.compile(
    r"\b(?:"
    r"bounced|"
    r"forged|"
    r"counterfeit|"
    r"fraud|"
    r"scam|"
    r"not good|"
    r"invalid"
    r")\b",
    re.I,
)


INVESTMENT_PATTERN = re.compile(
    r"\b(?:"
    r"investment|"
    r"invest money|"
    r"trading platform|"
    r"crypto(?:currency)?|"
    r"bitcoin|"
    r"ethereum|"
    r"guaranteed (?:a )?(?:profit|return)|"
    r"promised (?:a )?profit|"
    r"profit in return|"
    r"upgrade.*profit"
    r")\b",
    re.I,
)


ROMANCE_PATTERN = re.compile(
    r"\b(?:"
    r"romance scam|"
    r"dating scam|"
    r"boyfriend|"
    r"girlfriend|"
    r"fianc[eé]|"
    r"romantic partner|"
    r"online romance"
    r")\b",
    re.I,
)


REMOTE_ACCESS_PATTERN = re.compile(
    r"\b(?:"
    r"remote access|"
    r"screen share|"
    r"anydesk|"
    r"teamviewer|"
    r"remote desktop"
    r")\b",
    re.I,
)


PURCHASE_CONTEXT_PATTERN = re.compile(
    r"\b(?:"
    r"marketplace|"
    r"seller|"
    r"buyer|"
    r"purchase|"
    r"tickets?|"
    r"rental|"
    r"landlord|"
    r"apartment|"
    r"membership"
    r")\b",
    re.I,
)


PURCHASE_FRAUD_PATTERN = re.compile(
    r"\b(?:"
    r"scam|"
    r"scammed|"
    r"fraud|"
    r"fraudulent|"
    r"fake|"
    r"never received|"
    r"not delivered|"
    r"did not receive|"
    r"seller disappeared|"
    r"account is fake"
    r")\b",
    re.I,
)


PAYMENT_APP_PATTERN = re.compile(
    r"\b(?:"
    r"payment_app|"
    r"moneygram|"
    r"western union|"
    r"wire transfer|"
    r"wire payment"
    r")\b",
    re.I,
)


PAYMENT_FRAUD_PATTERN = re.compile(
    r"\b(?:"
    r"fraud|"
    r"fraudulent|"
    r"scam|"
    r"scammed|"
    r"unauthori[sz]ed|"
    r"without my consent|"
    r"without my permission|"
    r"account drained|"
    r"stolen"
    r")\b",
    re.I,
)


UTILITY_IMPERSONATION_PATTERN = re.compile(
    r"\b(?:"
    r"power|"
    r"electric|"
    r"utility|"
    r"gas service"
    r")\b",
    re.I,
)


PAYMENT_DEMAND_PATTERN = re.compile(
    r"\b(?:"
    r"needed to submit payment|"
    r"submit a second payment|"
    r"make another payment|"
    r"pay.*avoid suspension|"
    r"payment.*within the next hour"
    r")\b",
    re.I,
)


# ============================================================
# Helper
# ============================================================


def _patterns_near_each_other(
    text: str,
    first: re.Pattern,
    second: re.Pattern,
    max_distance: int = 150,
) -> bool:
    """Return True when two concepts occur close together."""

    first_matches = list(
        first.finditer(text)
    )

    second_matches = list(
        second.finditer(text)
    )

    for first_match in first_matches:
        for second_match in second_matches:

            distance = min(
                abs(
                    first_match.end()
                    - second_match.start()
                ),
                abs(
                    second_match.end()
                    - first_match.start()
                ),
            )

            if distance <= max_distance:
                return True

    return False


# ============================================================
# Archetype inference
# ============================================================


def infer_provisional_archetype(
    text: object,
) -> str:
    """Assign a conservative rule-based fraud archetype.

    The classifier requires evidence of the fraud behavior rather than
    assigning categories from generic words such as "police",
    "social security", "gift card", or "payment_app".
    """

    value = str(text)

    # --------------------------------------------------------
    # 1. Explicit identity theft
    # --------------------------------------------------------

    if IDENTITY_THEFT_PATTERN.search(
        value
    ):
        return (
            "identity theft or synthetic identity fraud"
        )

    # --------------------------------------------------------
    # 2. Government impersonation
    # --------------------------------------------------------

    if re.search(
        r"\b(?:"
        r"irs scam|"
        r"government impostor scam"
        r")\b",
        value,
        re.I,
    ):
        return (
            "debt-collection or government impostor scam"
        )

    if _patterns_near_each_other(
        value,
        IMPERSONATION_ACTION_PATTERN,
        GOVERNMENT_ENTITY_PATTERN,
        max_distance=150,
    ):
        return (
            "debt-collection or government impostor scam"
        )

    # Utility impersonation
    if (
        UTILITY_IMPERSONATION_PATTERN.search(
            value
        )
        and PAYMENT_DEMAND_PATTERN.search(
            value
        )
        and re.search(
            r"\b(?:"
            r"scammer|"
            r"scammers|"
            r"fake caller id|"
            r"claimed|"
            r"affirmed"
            r")\b",
            value,
            re.I,
        )
    ):
        return (
            "impersonation or phishing scam"
        )

    # --------------------------------------------------------
    # 3. Other impersonation / phishing
    # --------------------------------------------------------

    if PHISHING_PATTERN.search(
        value
    ):
        return (
            "impersonation or phishing scam"
        )

    if _patterns_near_each_other(
        value,
        IMPERSONATION_ACTION_PATTERN,
        TRUSTED_ORG_PATTERN,
        max_distance=150,
    ):
        return (
            "impersonation or phishing scam"
        )

    # --------------------------------------------------------
    # 4. Investment scam
    # --------------------------------------------------------

    if INVESTMENT_PATTERN.search(
        value
    ):
        return (
            "investment or cryptocurrency scam"
        )

    # --------------------------------------------------------
    # 5. Romance scam
    # --------------------------------------------------------

    if ROMANCE_PATTERN.search(
        value
    ):
        return (
            "romance or relationship manipulation scam"
        )

    # --------------------------------------------------------
    # 6. Remote-access / tech-support scam
    # --------------------------------------------------------

    if REMOTE_ACCESS_PATTERN.search(
        value
    ):
        return (
            "remote-access or technical-support scam"
        )

    # --------------------------------------------------------
    # 7. Fake / forged check
    # --------------------------------------------------------

    if FAKE_CHECK_PATTERN.search(
        value
    ):
        return (
            "fake check or overpayment scam"
        )

    if (
        CASHIERS_CHECK_PATTERN.search(
            value
        )
        and CHECK_PROBLEM_PATTERN.search(
            value
        )
    ):
        return (
            "fake check or overpayment scam"
        )

    # --------------------------------------------------------
    # 8. Unauthorized account/card transaction
    # --------------------------------------------------------

    if UNAUTHORIZED_PATTERN.search(
        value
    ):
        return (
            "account takeover or unauthorized transaction"
        )

    # --------------------------------------------------------
    # 9. Marketplace / purchase scam
    #
    # Require purchase context AND fraud evidence.
    # --------------------------------------------------------

    if _patterns_near_each_other(
        value,
        PURCHASE_CONTEXT_PATTERN,
        PURCHASE_FRAUD_PATTERN,
        max_distance=200,
    ):
        return (
            "rental, marketplace, or purchase scam"
        )

    # --------------------------------------------------------
    # 10. Payment-app / money-transfer fraud
    #
    # Merely mentioning payment_app is not sufficient.
    # --------------------------------------------------------

    if _patterns_near_each_other(
        value,
        PAYMENT_APP_PATTERN,
        PAYMENT_FRAUD_PATTERN,
        max_distance=200,
    ):
        return (
            "money-transfer or payment-app scam"
        )

    # --------------------------------------------------------
    # 11. Conservative fallback
    # --------------------------------------------------------

    return (
        "other suspected fraud or disputed transaction"
    )


# ============================================================
# Text cleaning
# ============================================================


def clean_for_generation(
    text: str,
) -> str:
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
    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


# ============================================================
# Structured record
# ============================================================


def build_structured_record(
    row: Any,
    text_column: str,
    max_input_words: int,
) -> dict[str, Any]:
    """Convert one dataframe row into structured generator input."""

    clean_text = clean_for_generation(
        truncate_words(
            row[text_column],
            max_input_words,
        )
    )

    # Extract deterministic red flags.
    red_flags = extract_red_flags(
        clean_text
    )

    # Infer the provisional fraud archetype.
    archetype = infer_provisional_archetype(
        clean_text
    )

    # --------------------------------------------------------
    # Consistency rule
    #
    # If the red-flag detector has strong evidence that the
    # consumer denies authorizing a transaction, do not leave
    # the archetype as the generic fallback.
    # --------------------------------------------------------

    if (
        archetype
        == "other suspected fraud or disputed transaction"
        and
        "consumer denies authorizing the transaction"
        in red_flags
    ):
        archetype = (
            "account takeover or unauthorized transaction"
        )

    # Calculate deterministic risk.
    risk_level, risk_score = score_risk(
        red_flags,
        archetype,
    )

    return {
        "clean_text":
            clean_text,

        "archetype":
            archetype,

        "archetype_source":
            "conservative rule-based preliminary label",

        "archetype_confidence":
            None,

        "red_flags":
            red_flags,

        "risk_level":
            risk_level,

        "risk_score":
            int(
                risk_score
            ),
    }


# ============================================================
# FLAN-T5 generation prompt
# ============================================================


def build_generation_prompt(
    record: dict[str, Any],
) -> str:
    """Create a concise, grounded complaint-summary prompt."""

    return (
        "Summarize the following consumer complaint in two concise sentences "
        "for a fraud analyst. Do not invent facts.\n\n"
        f"Complaint: {record['clean_text']}\n\n"
        "Summary:"
    )