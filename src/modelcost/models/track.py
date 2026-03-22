"""Pydantic models for cost-tracking requests and responses."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Any

from pydantic import BaseModel, Field


class TrackRequest(BaseModel):
    """Payload sent to the ``/track`` endpoint."""

    model_config = {
        "populate_by_name": True,
    }

    api_key: str = Field(..., alias="apiKey")
    timestamp: datetime
    provider: str
    model: str
    feature: str | None = None
    customer_id: str | None = Field(default=None, alias="customerId")
    input_tokens: int = Field(..., ge=0, alias="inputTokens")
    output_tokens: int = Field(..., ge=0, alias="outputTokens")
    latency_ms: int | None = Field(default=None, alias="latencyMs")
    metadata: dict[str, Any] | None = None


class TrackResponse(BaseModel):
    """Response from the ``/track`` endpoint."""

    model_config = {
        "populate_by_name": True,
    }

    status: str
