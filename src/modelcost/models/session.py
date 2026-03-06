"""Pydantic models for agent session governance."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """Payload sent to POST /v1/sessions."""

    model_config = {"populate_by_name": True}

    api_key: str = Field(..., alias="apiKey")
    session_id: str = Field(..., alias="sessionId")
    feature: Optional[str] = None
    user_id: Optional[str] = Field(default=None, alias="userId")
    max_spend_usd: Optional[float] = Field(default=None, alias="maxSpendUsd")
    max_iterations: Optional[int] = Field(default=None, alias="maxIterations")


class CreateSessionResponse(BaseModel):
    """Response from POST /v1/sessions."""

    model_config = {"populate_by_name": True}

    id: str
    session_id: str = Field(..., alias="sessionId")
    status: str
    max_spend_usd: Optional[float] = Field(default=None, alias="maxSpendUsd")
    max_iterations: Optional[int] = Field(default=None, alias="maxIterations")


class RecordSessionCallRequest(BaseModel):
    """Payload sent to POST /v1/sessions/{id}/calls."""

    model_config = {"populate_by_name": True}

    api_key: str = Field(..., alias="apiKey")
    call_sequence: int = Field(..., alias="callSequence")
    call_type: str = Field(..., alias="callType")
    tool_name: Optional[str] = Field(default=None, alias="toolName")
    input_tokens: int = Field(default=0, alias="inputTokens")
    output_tokens: int = Field(default=0, alias="outputTokens")
    cumulative_input_tokens: int = Field(default=0, alias="cumulativeInputTokens")
    cost_usd: float = Field(default=0.0, alias="costUsd")
    cumulative_cost_usd: float = Field(default=0.0, alias="cumulativeCostUsd")
    pii_detected: bool = Field(default=False, alias="piiDetected")


class CloseSessionRequest(BaseModel):
    """Payload sent to POST /v1/sessions/{id}/close."""

    model_config = {"populate_by_name": True}

    api_key: str = Field(..., alias="apiKey")
    status: str
    termination_reason: Optional[str] = Field(default=None, alias="terminationReason")
    final_spend_usd: float = Field(..., alias="finalSpendUsd")
    final_iteration_count: int = Field(..., alias="finalIterationCount")


class SessionCallSummary(BaseModel):
    """Individual call within a session summary."""

    model_config = {"populate_by_name": True}

    call_sequence: int = Field(..., alias="callSequence")
    call_type: str = Field(..., alias="callType")
    tool_name: Optional[str] = Field(default=None, alias="toolName")
    input_tokens: int = Field(..., alias="inputTokens")
    output_tokens: int = Field(..., alias="outputTokens")
    cost_usd: float = Field(..., alias="costUsd")
    cumulative_cost_usd: float = Field(..., alias="cumulativeCostUsd")
    pii_detected: bool = Field(..., alias="piiDetected")
    created_at: datetime = Field(..., alias="createdAt")


class SessionSummaryResponse(BaseModel):
    """Response from GET /v1/sessions/{id}/summary."""

    model_config = {"populate_by_name": True}

    id: str
    session_id: str = Field(..., alias="sessionId")
    feature: Optional[str] = None
    user_id: Optional[str] = Field(default=None, alias="userId")
    max_spend_usd: Optional[float] = Field(default=None, alias="maxSpendUsd")
    max_iterations: Optional[int] = Field(default=None, alias="maxIterations")
    current_spend_usd: float = Field(..., alias="currentSpendUsd")
    iteration_count: int = Field(..., alias="iterationCount")
    status: str
    termination_reason: Optional[str] = Field(default=None, alias="terminationReason")
    started_at: datetime = Field(..., alias="startedAt")
    ended_at: Optional[datetime] = Field(default=None, alias="endedAt")
    calls: List[SessionCallSummary] = Field(default_factory=list)
