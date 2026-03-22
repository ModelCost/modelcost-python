"""SDK configuration using pydantic for validation."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class ModelCostConfig(BaseModel):
    """Configuration for the ModelCost SDK.

    All settings can be provided explicitly or read from environment variables
    via the ``from_env()`` class method.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    api_key: str
    org_id: str
    environment: str = "production"
    base_url: str = "https://api.modelcost.ai"
    monthly_budget: float | None = None
    budget_action: Literal["alert", "throttle", "block"] = "alert"
    fail_open: bool = True
    flush_interval_seconds: float = 5.0
    flush_batch_size: int = 100
    sync_interval_seconds: float = 10.0
    content_privacy: bool = False

    @field_validator("api_key")
    @classmethod
    def _api_key_must_start_with_mc(cls, v: str) -> str:
        if not v.startswith("mc_"):
            raise ValueError("api_key must start with 'mc_'")
        return v

    @classmethod
    def from_env(cls, **overrides: object) -> ModelCostConfig:
        """Build a config from environment variables.

        Recognised variables:
        - ``MODELCOST_API_KEY``
        - ``MODELCOST_ORG_ID``
        - ``MODELCOST_ENV``
        - ``MODELCOST_BASE_URL``

        Any keyword arguments override the environment values.
        """
        env_values: dict[str, object] = {}

        api_key = os.environ.get("MODELCOST_API_KEY")
        if api_key is not None:
            env_values["api_key"] = api_key

        org_id = os.environ.get("MODELCOST_ORG_ID")
        if org_id is not None:
            env_values["org_id"] = org_id

        env = os.environ.get("MODELCOST_ENV")
        if env is not None:
            env_values["environment"] = env

        base_url = os.environ.get("MODELCOST_BASE_URL")
        if base_url is not None:
            env_values["base_url"] = base_url

        content_privacy = os.environ.get("MODELCOST_CONTENT_PRIVACY")
        if content_privacy is not None:
            env_values["content_privacy"] = content_privacy.lower() == "true"

        env_values.update(overrides)
        return cls(**env_values)  # type: ignore[arg-type]
