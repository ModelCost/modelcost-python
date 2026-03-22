"""Pydantic models for governance / PII scanning."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GovernanceScanRequest(BaseModel):
    """Payload sent to the governance scan endpoint."""

    model_config = {
        "populate_by_name": True,
    }

    org_id: str = Field(..., alias="orgId")
    text: str
    feature: str | None = None
    environment: str | None = None


class DetectedViolation(BaseModel):
    """A single violation detected by the governance scanner."""

    model_config = {
        "populate_by_name": True,
    }

    type: str
    subtype: str
    severity: str
    start: int
    end: int


class GovernanceSignalRequest(BaseModel):
    """Payload for reporting classification signals without raw text (metadata-only mode)."""

    model_config = {
        "populate_by_name": True,
    }

    organization_id: str = Field(..., alias="organizationId")
    violation_type: str = Field(..., alias="violationType")
    violation_subtype: str = Field(..., alias="violationSubtype")
    severity: str
    environment: str | None = None
    action_taken: str = Field(..., alias="actionTaken")
    was_allowed: bool = Field(..., alias="wasAllowed")
    detected_at: str | None = Field(default=None, alias="detectedAt")
    source: str = "metadata_only"
    violation_count: int = Field(default=1, alias="violationCount")


class GovernanceScanResponse(BaseModel):
    """Response from the governance scan endpoint."""

    model_config = {
        "populate_by_name": True,
    }

    is_allowed: bool = Field(..., alias="isAllowed")
    action: str | None = None
    violations: list[DetectedViolation] = Field(default_factory=list)
    redacted_text: str | None = Field(default=None, alias="redactedText")
