"""Pydantic models for budget checking and status."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003

from pydantic import BaseModel, Field


class BudgetCheckResponse(BaseModel):
    """Response from the budget-check endpoint."""

    model_config = {
        "populate_by_name": True,
    }

    allowed: bool
    action: str | None = None
    throttle_percentage: int | None = Field(default=None, alias="throttlePercentage")
    reason: str | None = None


class BudgetPolicy(BaseModel):
    """A single budget policy with current spend information."""

    model_config = {
        "populate_by_name": True,
    }

    id: str
    name: str
    scope: str
    scope_identifier: str | None = Field(default=None, alias="scopeIdentifier")
    budget_amount_usd: float = Field(..., alias="budgetAmountUsd")
    period: str
    custom_period_days: int | None = Field(default=None, alias="customPeriodDays")
    action: str
    throttle_percentage: int | None = Field(default=None, alias="throttlePercentage")
    alert_thresholds: list[int] = Field(default_factory=list, alias="alertThresholds")
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

    policies: list[BudgetPolicy]
    total_budget_usd: float = Field(..., alias="totalBudgetUsd")
    total_spend_usd: float = Field(..., alias="totalSpendUsd")
    policies_at_risk: int = Field(..., alias="policiesAtRisk")
