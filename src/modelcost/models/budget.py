"""Pydantic models for budget checking and status."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class BudgetCheckResponse(BaseModel):
    """Response from the budget-check endpoint."""

    model_config = {
        "populate_by_name": True,
    }

    allowed: bool
    action: Optional[str] = None
    throttle_percentage: Optional[int] = Field(default=None, alias="throttlePercentage")
    reason: Optional[str] = None


class BudgetPolicy(BaseModel):
    """A single budget policy with current spend information."""

    model_config = {
        "populate_by_name": True,
    }

    id: str
    name: str
    scope: str
    scope_identifier: Optional[str] = Field(default=None, alias="scopeIdentifier")
    budget_amount_usd: float = Field(..., alias="budgetAmountUsd")
    period: str
    custom_period_days: Optional[int] = Field(default=None, alias="customPeriodDays")
    action: str
    throttle_percentage: Optional[int] = Field(default=None, alias="throttlePercentage")
    alert_thresholds: List[int] = Field(default_factory=list, alias="alertThresholds")
    current_spend_usd: float = Field(..., alias="currentSpendUsd")
    spend_percentage: float = Field(..., alias="spendPercentage")
    period_start: datetime = Field(..., alias="periodStart")
    is_active: bool = Field(..., alias="isActive")
    created_at: datetime = Field(..., alias="createdAt")
    updated_at: datetime = Field(..., alias="updatedAt")


class BudgetStatusResponse(BaseModel):
    """Full budget status including all active policies."""

    model_config = {
        "populate_by_name": True,
    }

    policies: List[BudgetPolicy]
    total_budget_usd: float = Field(..., alias="totalBudgetUsd")
    total_spend_usd: float = Field(..., alias="totalSpendUsd")
    policies_at_risk: int = Field(..., alias="policiesAtRisk")
