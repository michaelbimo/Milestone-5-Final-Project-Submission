"""Rule-based red-flag extraction and transparent risk scoring.

The rules in this module are intentionally conservative.

A red flag should only be emitted when the complaint contains evidence of
the behavior itself, not merely a keyword such as "payment_app", "bank",
"relationship", or "police".
"""

from __future__ import annotations

import re


# ============================================================
# Core evidence patterns
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
    r"charges? (?:that )?(?:i am|i'm|im) unaware of|"
    r"did not make (?:this|these|the) "
    r"(?:purchase|purchases|transaction|transactions)|"
    r"someone (?:used|took|withdrew|transferred)"
    r")\b",
    re.I,
)


THREAT_PATTERN = re.compile(
    r"\b(?:"
    r"threat(?:en|ened|ening)?|"
    r"will be arrested|"
    r"would be arrested|"
    r"police will|"
    r"account will be (?:closed|frozen|suspended)|"
    r"service will be (?:cut off|disconnected|suspended|terminated)|"
    r"avoid (?:arrest|suspension|termination|disconnection)|"
    r"within (?:the )?next (?:hour|hours|minutes)|"
    r"pay immediately|"
    r"act now"
    r")\b",
    re.I,
)


IMPERSONATION_PATTERN = re.compile(
    r"\b(?:"
    r"claimed to be|"
    r"claiming to be|"
    r"pretend(?:ed|ing)? to be|"
    r"posed as|"
    r"posing as|"
    r"impersonat(?:ed|ing|ion)|"
    r"said (?:he|she|they) (?:was|were) from|"
    r"called (?:me )?(?:claiming|saying) "
    r"(?:he|she|they) (?:was|were) from"
    r")\b",
    re.I,
)


TRUSTED_ORGANIZATION_PATTERN = re.compile(
    r"\b(?:"
    r"bank|"
    r"credit union|"
    r"irs|"
    r"internal revenue service|"
    r"government|"
    r"social security|"
    r"social security administration|"
    r"police|"
    r"utility|"
    r"power company|"
    r"electric company|"
    r"customer service|"
    r"fraud department|"
    r"fraud specialist"
    r")\b",
    re.I,
)


CREDENTIAL_PATTERN = re.compile(
    r"\b(?:"
    r"password|"
    r"passcode|"
    r"pin|"
    r"verification code|"
    r"one[- ]time code|"
    r"otp|"
    r"security code|"
    r"login credentials?|"
    r"bank card information|"
    r"debit card information|"
    r"credit card information"
    r")\b",
    re.I,
)


CREDENTIAL_REQUEST_PATTERN = re.compile(
    r"\b(?:"
    r"ask(?:ed)?|"
    r"request(?:ed)?|"
    r"wanted|"
    r"told me to|"
    r"instruct(?:ed)? me to|"
    r"required me to|"
    r"made me|"
    r"tried to get"
    r")\b",
    re.I,
)


TRANSFER_ACTION_PATTERN = re.compile(
    r"\b(?:"
    r"send money|"
    r"send payment|"
    r"transfer money|"
    r"transfer funds|"
    r"wire money|"
    r"wire funds|"
    r"make (?:a|another|the) payment|"
    r"submit (?:a|another|second) payment|"
    r"purchase (?:a |the )?(?:gift card|prepaid card)|"
    r"buy (?:a |the )?(?:gift card|prepaid card)|"
    r"pay (?:by|via|with)"
    r")\b",
    re.I,
)


TRANSFER_REQUEST_PATTERN = re.compile(
    r"\b(?:"
    r"ask(?:ed)? me to|"
    r"told me to|"
    r"instruct(?:ed)? me to|"
    r"required me to|"
    r"made me|"
    r"forced me to|"
    r"wanted me to|"
    r"said i (?:had|needed) to|"
    r"i had to"
    r")\b",
    re.I,
)


UNEXPECTED_CONTACT_PATTERN = re.compile(
    r"\b(?:"
    r"unsolicited (?:call|text|message)|"
    r"unexpected (?:call|text|message)|"
    r"random (?:call|text|message)|"
    r"received (?:an? )?(?:unexpected|unsolicited) "
    r"(?:call|text|message)|"
    r"fake caller id"
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


INVESTMENT_PROMISE_PATTERN = re.compile(
    r"\b(?:"
    r"guaranteed (?:a )?(?:return|profit)|"
    r"guaranteed profit|"
    r"high return|"
    r"double your money|"
    r"profit promised|"
    r"promised (?:a )?profit|"
    r"guaranteed.*profit"
    r")\b",
    re.I,
)


ROMANCE_PATTERN = re.compile(
    r"\b(?:"
    r"romance scam|"
    r"dating|"
    r"boyfriend|"
    r"girlfriend|"
    r"fianc[eé]|"
    r"romantic partner|"
    r"online romance"
    r")\b",
    re.I,
)


FAKE_CHECK_PATTERN = re.compile(
    r"\b(?:"
    r"fake check|"
    r"counterfeit check|"
    r"check bounced|"
    r"check was forged|"
    r"forged check|"
    r"check (?:is|was) not good|"
    r"bad check|"
    r"unauthorized endorsement"
    r")\b",
    re.I,
)


CASHIERS_CHECK_PATTERN = re.compile(
    r"\bcashier'?s check\b",
    re.I,
)


CHECK_FAILURE_PATTERN = re.compile(
    r"\b(?:"
    r"bounced|"
    r"forged|"
    r"counterfeit|"
    r"not good|"
    r"fraud|"
    r"scam|"
    r"returned|"
    r"invalid"
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
    max_distance: int = 120,
) -> bool:
    """Return True when two concepts occur near one another."""

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
# Red-flag extraction
# ============================================================


def extract_red_flags(
    text: object,
) -> list[str]:
    """Extract conservative, evidence-backed fraud red flags."""

    value = str(text)

    flags: list[str] = []

    # --------------------------------------------------------
    # Urgency / threatening language
    # --------------------------------------------------------

    if THREAT_PATTERN.search(
        value
    ):
        flags.append(
            "urgent or threatening language"
        )

    # --------------------------------------------------------
    # Impersonation
    #
    # Require both impersonation language and an organization.
    # Mentioning "bank", "police", or "IRS" alone is not enough.
    # --------------------------------------------------------

    if _patterns_near_each_other(
        value,
        IMPERSONATION_PATTERN,
        TRUSTED_ORGANIZATION_PATTERN,
        max_distance=120,
    ):
        flags.append(
            "impersonation of a trusted organization"
        )

    elif re.search(
        r"\b(?:"
        r"irs scam|"
        r"government impostor|"
        r"government impersonation"
        r")\b",
        value,
        re.I,
    ):
        flags.append(
            "impersonation of a trusted organization"
        )

    # --------------------------------------------------------
    # Credentials / verification codes
    # --------------------------------------------------------

    if _patterns_near_each_other(
        value,
        CREDENTIAL_REQUEST_PATTERN,
        CREDENTIAL_PATTERN,
        max_distance=100,
    ):
        flags.append(
            "request for credentials or verification codes"
        )

    # --------------------------------------------------------
    # Requested money transfer/payment
    #
    # A simple mention of payment_app, wire, transfer, or
    # gift card is intentionally NOT enough.
    # --------------------------------------------------------

    if _patterns_near_each_other(
        value,
        TRANSFER_REQUEST_PATTERN,
        TRANSFER_ACTION_PATTERN,
        max_distance=120,
    ):
        flags.append(
            "request to transfer money"
        )

    # --------------------------------------------------------
    # Unexpected contact
    # --------------------------------------------------------

    if UNEXPECTED_CONTACT_PATTERN.search(
        value
    ):
        flags.append(
            "unexpected contact"
        )

    # --------------------------------------------------------
    # Remote access
    # --------------------------------------------------------

    if REMOTE_ACCESS_PATTERN.search(
        value
    ):
        flags.append(
            "remote device access"
        )

    # --------------------------------------------------------
    # Investment-return promise
    # --------------------------------------------------------

    if INVESTMENT_PROMISE_PATTERN.search(
        value
    ):
        flags.append(
            "investment return promise"
        )

    # --------------------------------------------------------
    # Romance manipulation
    #
    # Generic word "relationship" is intentionally excluded.
    # --------------------------------------------------------

    if ROMANCE_PATTERN.search(
        value
    ):
        flags.append(
            "romance or relationship manipulation"
        )

    # --------------------------------------------------------
    # Fake / counterfeit check
    # --------------------------------------------------------

    fake_check = bool(
        FAKE_CHECK_PATTERN.search(
            value
        )
    )

    suspicious_cashiers_check = bool(
        CASHIERS_CHECK_PATTERN.search(
            value
        )
        and CHECK_FAILURE_PATTERN.search(
            value
        )
    )

    if (
        fake_check
        or suspicious_cashiers_check
    ):
        flags.append(
            "check overpayment or counterfeit check"
        )

    # --------------------------------------------------------
    # Consumer explicitly denies authorization
    # --------------------------------------------------------

    if UNAUTHORIZED_PATTERN.search(
        value
    ):
        flags.append(
            "consumer denies authorizing the transaction"
        )

    return flags


# ============================================================
# Risk scoring
# ============================================================


HIGH_SEVERITY_FLAGS = {
    "request for credentials or verification codes",
    "remote device access",
    "consumer denies authorizing the transaction",
}


MEDIUM_SEVERITY_FLAGS = {
    "impersonation of a trusted organization",
    "request to transfer money",
    "investment return promise",
    "romance or relationship manipulation",
    "check overpayment or counterfeit check",
}


def score_risk(
    red_flags: list[str],
    archetype: str = "",
) -> tuple[str, int]:
    """Calculate a transparent rule-based risk score."""

    score = len(
        red_flags
    )

    # Strong signals receive one additional point.
    score += sum(
        flag in HIGH_SEVERITY_FLAGS
        for flag in red_flags
    )

    # Multiple medium-severity indicators add one more point.
    medium_count = sum(
        flag in MEDIUM_SEVERITY_FLAGS
        for flag in red_flags
    )

    if medium_count >= 2:
        score += 1

    # Strong archetypes contribute one point.
    if re.search(
        r"\b(?:"
        r"account takeover|"
        r"identity theft|"
        r"investment|"
        r"impersonation"
        r")\b",
        str(archetype),
        re.I,
    ):
        score += 1

    if score >= 5:
        return "High", score

    if score >= 2:
        return "Medium", score

    return "Low", score