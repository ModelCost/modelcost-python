"""Tests for modelcost.pii (PiiScanner)."""

from __future__ import annotations

import pytest

from modelcost.pii import PiiScanner


@pytest.fixture()
def scanner() -> PiiScanner:
    return PiiScanner()


class TestSsnDetection:
    def test_detects_ssn(self, scanner: PiiScanner) -> None:
        result = scanner.scan("My SSN is 123-45-6789")
        assert result.detected is True
        ssn_entities = [e for e in result.entities if e["type"] == "ssn"]
        assert len(ssn_entities) == 1
        assert ssn_entities[0]["value"] == "123-45-6789"

    def test_ssn_positions(self, scanner: PiiScanner) -> None:
        text = "SSN: 123-45-6789 end"
        result = scanner.scan(text)
        ssn = [e for e in result.entities if e["type"] == "ssn"][0]
        assert text[ssn["start"] : ssn["end"]] == "123-45-6789"


class TestCreditCardDetection:
    def test_detects_credit_card(self, scanner: PiiScanner) -> None:
        result = scanner.scan("Card: 4111111111111111")
        assert result.detected is True
        cc_entities = [e for e in result.entities if e["type"] == "credit_card"]
        assert len(cc_entities) >= 1

    def test_detects_card_with_spaces(self, scanner: PiiScanner) -> None:
        result = scanner.scan("Card: 4111 1111 1111 1111")
        assert result.detected is True


class TestEmailDetection:
    def test_detects_email(self, scanner: PiiScanner) -> None:
        result = scanner.scan("Email me at test@example.com please")
        assert result.detected is True
        email_entities = [e for e in result.entities if e["type"] == "email"]
        assert len(email_entities) == 1
        assert email_entities[0]["value"] == "test@example.com"


class TestPhoneDetection:
    def test_detects_phone(self, scanner: PiiScanner) -> None:
        result = scanner.scan("Call me at 555-123-4567")
        assert result.detected is True
        phone_entities = [e for e in result.entities if e["type"] == "phone"]
        assert len(phone_entities) == 1
        assert "555-123-4567" in phone_entities[0]["value"]


class TestCleanText:
    def test_no_false_positives_on_clean_text(self, scanner: PiiScanner) -> None:
        result = scanner.scan("The quick brown fox jumps over the lazy dog.")
        assert result.detected is False
        assert len(result.entities) == 0

    def test_no_false_positives_on_numbers(self, scanner: PiiScanner) -> None:
        result = scanner.scan("Order #12345 was placed on 2026-02-16.")
        assert result.detected is False


class TestRedaction:
    def test_redaction_replaces_with_asterisks(self, scanner: PiiScanner) -> None:
        text = "SSN is 123-45-6789 and email is user@test.com"
        result = scanner.scan(text)
        assert result.detected is True
        assert "123-45-6789" not in result.redacted_text
        assert "user@test.com" not in result.redacted_text
        # Asterisks should be present
        assert "***" in result.redacted_text

    def test_redact_method_directly(self, scanner: PiiScanner) -> None:
        text = "My SSN: 123-45-6789"
        redacted = scanner.redact(text)
        assert "123-45-6789" not in redacted
        assert "***" in redacted


# ─── fullScan tests ────────────────────────────────────────────


class TestSecretsDetection:
    def test_detects_openai_api_key(self, scanner: PiiScanner) -> None:
        result = scanner.full_scan("My key is sk-abc123def456ghi789jkl012mno")
        assert result.detected is True
        assert "secrets" in result.categories
        secret = next((v for v in result.violations if v.type == "api_key_openai"), None)
        assert secret is not None
        assert secret.severity == "critical"

    def test_detects_aws_access_key(self, scanner: PiiScanner) -> None:
        result = scanner.full_scan("AWS key: AKIAIOSFODNN7EXAMPLE")
        assert result.detected is True
        assert "secrets" in result.categories
        aws = next((v for v in result.violations if v.type == "api_key_aws"), None)
        assert aws is not None

    def test_detects_private_key(self, scanner: PiiScanner) -> None:
        result = scanner.full_scan("-----BEGIN RSA PRIVATE KEY-----\ncontent\n-----END RSA PRIVATE KEY-----")
        assert result.detected is True
        pk = next((v for v in result.violations if v.type == "private_key"), None)
        assert pk is not None
        assert pk.severity == "critical"

    def test_detects_jwt_token(self, scanner: PiiScanner) -> None:
        result = scanner.full_scan(
            "Token: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        )
        assert result.detected is True
        jwt = next((v for v in result.violations if v.type == "jwt_token"), None)
        assert jwt is not None

    def test_detects_generic_secret(self, scanner: PiiScanner) -> None:
        result = scanner.full_scan("config: password=SuperSecretValue123")
        assert result.detected is True
        generic = next((v for v in result.violations if v.type == "generic_secret"), None)
        assert generic is not None


class TestPhiDetection:
    def test_detects_phi_with_medical_context_and_pii(self, scanner: PiiScanner) -> None:
        result = scanner.full_scan("Patient with SSN 123-45-6789 has diabetes and needs insulin")
        assert result.detected is True
        assert "phi" in result.categories
        phi = [v for v in result.violations if v.category == "phi"]
        assert len(phi) > 0

    def test_no_phi_without_pii_cooccurrence(self, scanner: PiiScanner) -> None:
        result = scanner.full_scan("The patient is being treated for diabetes", ["phi"])
        phi = [v for v in result.violations if v.category == "phi"]
        assert len(phi) == 0


class TestFinancialDetection:
    def test_detects_credit_card_with_luhn(self, scanner: PiiScanner) -> None:
        result = scanner.full_scan("Card: 4111111111111111", ["financial"])
        assert result.detected is True
        assert "financial" in result.categories

    def test_detects_iban(self, scanner: PiiScanner) -> None:
        result = scanner.full_scan("Transfer to DE89370400440532013000", ["financial"])
        assert result.detected is True
        iban = next((v for v in result.violations if v.type == "iban"), None)
        assert iban is not None


class TestLuhnValidation:
    def test_valid_luhn_numbers(self, scanner: PiiScanner) -> None:
        from modelcost.pii import _is_valid_luhn

        assert _is_valid_luhn("4111111111111111") is True
        assert _is_valid_luhn("5500000000000004") is True

    def test_invalid_luhn_numbers(self, scanner: PiiScanner) -> None:
        from modelcost.pii import _is_valid_luhn

        assert _is_valid_luhn("1234567890123456") is False

    def test_too_short_numbers(self, scanner: PiiScanner) -> None:
        from modelcost.pii import _is_valid_luhn

        assert _is_valid_luhn("123") is False
        assert _is_valid_luhn("") is False


class TestFullScanCleanText:
    def test_no_violations_for_clean_text(self, scanner: PiiScanner) -> None:
        result = scanner.full_scan("Just a normal business message.")
        assert result.detected is False
        assert len(result.violations) == 0
        assert len(result.categories) == 0

    def test_no_violations_for_empty_string(self, scanner: PiiScanner) -> None:
        result = scanner.full_scan("")
        assert result.detected is False


class TestFullScanCategoryFiltering:
    def test_only_scans_requested_categories(self, scanner: PiiScanner) -> None:
        text = "SSN: 123-45-6789 and key: sk-abc123def456ghi789jkl012mno"

        pii_only = scanner.full_scan(text, ["pii"])
        assert "pii" in pii_only.categories
        assert "secrets" not in pii_only.categories

        secrets_only = scanner.full_scan(text, ["secrets"])
        assert "secrets" in secrets_only.categories
        assert "pii" not in secrets_only.categories
