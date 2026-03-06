"""Pydantic models for model pricing information."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelPricing(BaseModel):
    """Pricing for a specific model from a provider."""

    model_config = {
        "populate_by_name": True,
    }

    provider: str
    input_cost_per_1k: float = Field(..., alias="inputCostPer1k")
    output_cost_per_1k: float = Field(..., alias="outputCostPer1k")
