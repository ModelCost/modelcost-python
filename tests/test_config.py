"""Tests for modelcost.config."""

from __future__ import annotations

import os
from unittest import mock

import pytest
from pydantic import ValidationError

from modelcost.config import ModelCostConfig


class TestModelCostConfig:
    """Tests for explicit construction and validation."""

    def test_valid_config_from_args(self) -> None:
        cfg = ModelCostConfig(
            api_key="mc_test_key_12345",
            org_id="org_abc",
            environment="staging",
        )
        assert cfg.api_key == "mc_test_key_12345"
        assert cfg.org_id == "org_abc"
        assert cfg.environment == "staging"
        assert cfg.base_url == "https://api.modelcost.ai"
        assert cfg.fail_open is True
        assert cfg.flush_interval_seconds == 5.0
        assert cfg.flush_batch_size == 100
        assert cfg.sync_interval_seconds == 10.0
        assert cfg.monthly_budget is None
        assert cfg.budget_action == "alert"

    def test_invalid_api_key_prefix_raises(self) -> None:
        with pytest.raises(ValidationError, match="api_key must start with 'mc_'"):
            ModelCostConfig(api_key="sk_bad_prefix", org_id="org_1")

    def test_missing_api_key_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelCostConfig(org_id="org_1")  # type: ignore[call-arg]

    def test_missing_org_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            ModelCostConfig(api_key="mc_valid_key")  # type: ignore[call-arg]


class TestModelCostConfigFromEnv:
    """Tests for the ``from_env()`` class method."""

    def test_from_env_reads_variables(self) -> None:
        env = {
            "MODELCOST_API_KEY": "mc_env_key_xyz",
            "MODELCOST_ORG_ID": "org_env_456",
            "MODELCOST_ENV": "staging",
            "MODELCOST_BASE_URL": "https://custom.api.example.com",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = ModelCostConfig.from_env()
        assert cfg.api_key == "mc_env_key_xyz"
        assert cfg.org_id == "org_env_456"
        assert cfg.environment == "staging"
        assert cfg.base_url == "https://custom.api.example.com"

    def test_from_env_override_takes_precedence(self) -> None:
        env = {
            "MODELCOST_API_KEY": "mc_env_key",
            "MODELCOST_ORG_ID": "org_env",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            cfg = ModelCostConfig.from_env(api_key="mc_override_key")
        assert cfg.api_key == "mc_override_key"
        assert cfg.org_id == "org_env"

    def test_from_env_missing_required_raises(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError):
                ModelCostConfig.from_env()
