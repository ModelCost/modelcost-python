"""Local PII scanner using compiled regular expressions.

Provides fast, offline detection for PII, PHI, secrets, and financial data.
In metadata-only mode the full_scan() method runs classification locally
so that raw content never leaves the customer's environment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PiiResult:
    """Result of a PII scan."""

    detected: bool
    entities: list[dict[str, Any]] = field(default_factory=list)
    redacted_text: str = ""


@dataclass
class GovernanceViolation:
    """A single governance violation detected by the local scanner."""

    category: str
    type: str
    severity: str
    start: int
    end: int


@dataclass
class FullScanResult:
    """Result of a full governance scan across all categories."""

    detected: bool
    violations: list[GovernanceViolation] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PII patterns (compiled, thread-safe)
# ---------------------------------------------------------------------------

_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
_CREDIT_CARD_GENERIC = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")
_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
_PHONE_PATTERN = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ssn", _SSN_PATTERN),
    ("credit_card", _CREDIT_CARD_PATTERN),
    ("email", _EMAIL_PATTERN),
    ("phone", _PHONE_PATTERN),
]

# ---------------------------------------------------------------------------
# Secrets patterns
# ---------------------------------------------------------------------------

_OPENAI_KEY_PATTERN = re.compile(r"sk-[a-zA-Z0-9]{20,}")
_AWS_KEY_PATTERN = re.compile(r"AKIA[0-9A-Z]{16}")
_PRIVATE_KEY_PATTERN = re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----")
_JWT_PATTERN = re.compile(
    r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
)
_GENERIC_SECRET_PATTERN = re.compile(
    r'(?i)(?:password|api_key|apikey|secret|token|bearer)\s*[:=]\s*["\']?([A-Za-z0-9_\-/.]{8,})["\']?'
)

_SECRETS_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("api_key_openai", _OPENAI_KEY_PATTERN, "critical"),
    ("api_key_aws", _AWS_KEY_PATTERN, "critical"),
    ("private_key", _PRIVATE_KEY_PATTERN, "critical"),
    ("jwt_token", _JWT_PATTERN, "high"),
    ("generic_secret", _GENERIC_SECRET_PATTERN, "critical"),
]

# ---------------------------------------------------------------------------
# Financial patterns
# ---------------------------------------------------------------------------

_IBAN_PATTERN = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b")

_FINANCIAL_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("iban", _IBAN_PATTERN, "high"),
]

# ---------------------------------------------------------------------------
# PHI medical terms
# ---------------------------------------------------------------------------

_MEDICAL_TERMS: frozenset[str] = frozenset(
    [
        "diabetes",
        "hiv",
        "aids",
        "cancer",
        "tumor",
        "disease",
        "medication",
        "diagnosis",
        "treatment",
        "surgery",
        "prescription",
        "patient",
        "doctor",
        "hospital",
        "clinic",
        "medical record",
        "insulin",
        "prozac",
        "chemotherapy",
        "depression",
        "anxiety",
        "bipolar",
        "schizophrenia",
        "hepatitis",
        "tuberculosis",
        "epilepsy",
        "asthma",
        "arthritis",
        "alzheimer",
    ]
)


def _is_valid_luhn(number: str) -> bool:
    """Luhn algorithm for credit card validation."""
    if len(number) < 13 or len(number) > 19:
        return False
    total = 0
    alternate = False
    for ch in reversed(number):
        if not ch.isdigit():
            return False
        n = int(ch)
        if alternate:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alternate = not alternate
    return total % 10 == 0


class PiiScanner:
    """Detects and redacts PII from text using regex patterns.

    The compiled patterns are created at module level and are inherently
    thread-safe.

    For metadata-only mode, use :meth:`full_scan` which runs PII, PHI,
    secrets, and financial detection entirely in-process.
    """

    def scan(self, text: str) -> PiiResult:
        """Scan *text* for PII and return a :class:`PiiResult`."""
        entities: list[dict[str, Any]] = []

        for pii_type, pattern in _PATTERNS:
            for match in pattern.finditer(text):
                entities.append(
                    {
                        "type": pii_type,
                        "value": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                    }
                )

        # Sort by start position for deterministic ordering
        entities.sort(key=lambda e: e["start"])

        redacted = self.redact(text) if entities else text

        return PiiResult(
            detected=len(entities) > 0,
            entities=entities,
            redacted_text=redacted,
        )

    def full_scan(
        self,
        text: str,
        categories: list[str] | None = None,
    ) -> FullScanResult:
        """Full governance scan across multiple categories.

        Used in metadata-only mode where content never leaves the
        customer's environment.
        """
        if categories is None:
            categories = ["pii", "phi", "secrets", "financial"]

        violations: list[GovernanceViolation] = []
        detected_categories: set[str] = set()

        if "pii" in categories:
            for v in self._scan_pii_violations(text):
                violations.append(v)
                detected_categories.add("pii")

        if "phi" in categories:
            for v in self._scan_phi(text):
                violations.append(v)
                detected_categories.add("phi")

        if "secrets" in categories:
            for v in self._scan_secrets(text):
                violations.append(v)
                detected_categories.add("secrets")

        if "financial" in categories:
            for v in self._scan_financial(text):
                violations.append(v)
                detected_categories.add("financial")

        return FullScanResult(
            detected=len(violations) > 0,
            violations=violations,
            categories=sorted(detected_categories),
        )

    def redact(self, text: str) -> str:
        """Replace all detected PII in *text* with asterisks."""
        result = text
        for _pii_type, pattern in _PATTERNS:
            result = pattern.sub(lambda m: "*" * len(m.group()), result)
        return result

    # ─── Category Scanners ────────────────────────────────────────────

    def _scan_pii_violations(self, text: str) -> list[GovernanceViolation]:
        violations: list[GovernanceViolation] = []

        # SSN
        for match in _SSN_PATTERN.finditer(text):
            violations.append(
                GovernanceViolation("pii", "ssn", "critical", match.start(), match.end())
            )

        # Email
        for match in _EMAIL_PATTERN.finditer(text):
            violations.append(
                GovernanceViolation("pii", "email", "high", match.start(), match.end())
            )

        # Credit card (Luhn-validated)
        for match in _CREDIT_CARD_GENERIC.finditer(text):
            digits = re.sub(r"[\s-]", "", match.group())
            if _is_valid_luhn(digits):
                violations.append(
                    GovernanceViolation(
                        "pii", "credit_card", "critical", match.start(), match.end()
                    )
                )

        # Phone
        for match in _PHONE_PATTERN.finditer(text):
            violations.append(
                GovernanceViolation("pii", "phone", "medium", match.start(), match.end())
            )

        return violations

    def _scan_phi(self, text: str) -> list[GovernanceViolation]:
        text_lower = text.lower()
        has_medical_context = any(term in text_lower for term in _MEDICAL_TERMS)

        if not has_medical_context:
            return []

        # Medical context + PII = PHI violation
        pii_violations = self._scan_pii_violations(text)
        return [
            GovernanceViolation(
                "phi", f"phi_{v.type}", "critical", v.start, v.end
            )
            for v in pii_violations
        ]

    def _scan_secrets(self, text: str) -> list[GovernanceViolation]:
        violations: list[GovernanceViolation] = []
        for secret_type, pattern, severity in _SECRETS_PATTERNS:
            for match in pattern.finditer(text):
                violations.append(
                    GovernanceViolation(
                        "secrets", secret_type, severity, match.start(), match.end()
                    )
                )
        return violations

    def _scan_financial(self, text: str) -> list[GovernanceViolation]:
        violations: list[GovernanceViolation] = []

        # Credit card (Luhn-validated)
        for match in _CREDIT_CARD_GENERIC.finditer(text):
            digits = re.sub(r"[\s-]", "", match.group())
            if _is_valid_luhn(digits):
                violations.append(
                    GovernanceViolation(
                        "financial", "credit_card", "critical", match.start(), match.end()
                    )
                )

        # IBAN
        for fin_type, pattern, severity in _FINANCIAL_PATTERNS:
            for match in pattern.finditer(text):
                violations.append(
                    GovernanceViolation(
                        "financial", fin_type, severity, match.start(), match.end()
                    )
                )

        return violations
