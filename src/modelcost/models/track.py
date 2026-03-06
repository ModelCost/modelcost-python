"""Pydantic models for cost-tracking requests and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

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
    feature: Optional[str] = None
    customer_id: Optional[str] = Field(default=None, alias="customerId")
    input_tokens: int = Field(..., ge=0, alias="inputTokens")
    output_tokens: int = Field(..., ge=0, alias="outputTokens")
    latency_ms: Optional[int] = Field(default=None, alias="latencyMs")
    metadata: Optional[Dict[str, Any]] = None


class TrackResponse(BaseModel):
    """Response from the ``/track`` endpoint."""

    model_config = {
        "populate_by_name": True,
    }

    status: str
