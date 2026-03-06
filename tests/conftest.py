"""Shared fixtures for the ModelCost SDK test suite."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import httpx
import pytest
import respx

from modelcost.config import ModelCostConfig

# Path to JSON test fixtures
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture()
def config() -> ModelCostConfig:
    """Return a valid test configuration."""
    return ModelCostConfig(
        api_key="mc_test_abc123def456ghi789jkl012mno345pqr678stu901vwx",
        org_id="org_test_123",
        environment="test",
        base_url="https://api.modelcost.ai",
        fail_open=True,
        flush_batch_size=10,
        sync_interval_seconds=0.1,
    )


@pytest.fixture()
def mock_client(config: ModelCostConfig) -> respx.MockRouter:
    """Return a ``respx`` mock router pre-configured for the test base URL.

    Usage in tests::

        def test_something(mock_client):
            mock_client.post("/v1/track").respond(200, json={"status": "ok"})
            ...
    """
    with respx.mock(base_url=config.base_url) as router:
        yield router


def load_fixture(name: str) -> Dict[str, Any]:
    """Load a JSON fixture by file name (without extension)."""
    path = FIXTURES_DIR / f"{name}.json"
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


@pytest.fixture()
def track_request_fixture() -> Dict[str, Any]:
    return load_fixture("track_request")


@pytest.fixture()
def track_response_fixture() -> Dict[str, Any]:
    return load_fixture("track_response")


@pytest.fixture()
def budget_check_allowed_fixture() -> Dict[str, Any]:
    return load_fixture("budget_check_allowed")


@pytest.fixture()
def budget_check_blocked_fixture() -> Dict[str, Any]:
    return load_fixture("budget_check_blocked")


@pytest.fixture()
def budget_status_fixture() -> Dict[str, Any]:
    return load_fixture("budget_status")


@pytest.fixture()
def governance_scan_clean_fixture() -> Dict[str, Any]:
    return load_fixture("governance_scan_clean")


@pytest.fixture()
def governance_scan_pii_fixture() -> Dict[str, Any]:
    return load_fixture("governance_scan_pii")


@pytest.fixture()
def error_response_fixture() -> Dict[str, Any]:
    return load_fixture("error_response")
