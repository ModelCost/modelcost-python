"""Exception hierarchy for the ModelCost SDK."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ModelCostError(Exception):
    """Base exception for all ModelCost SDK errors."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ConfigurationError(ModelCostError):
    """Raised when the SDK configuration is invalid. Always fatal."""

    pass


class BudgetExceededError(ModelCostError):
    """Raised when a budget limit has been exceeded."""

    def __init__(
        self,
        message: str,
        remaining_budget: float,
        scope: str,
        override_url: Optional[str] = None,
    ) -> None:
        self.remaining_budget = remaining_budget
        self.scope = scope
        self.override_url = override_url
        super().__init__(message)


class RateLimitedError(ModelCostError):
    """Raised when the client is rate-limited."""

    def __init__(
        self,
        message: str,
        retry_after_seconds: float,
        limit_dimension: str,
    ) -> None:
        self.retry_after_seconds = retry_after_seconds
        self.limit_dimension = limit_dimension
        super().__init__(message)


class PiiDetectedError(ModelCostError):
    """Raised when PII is detected in text."""

    def __init__(
        self,
        message: str,
        detected_entities: List[Dict[str, Any]],
        redacted_text: str,
    ) -> None:
        self.detected_entities = detected_entities
        self.redacted_text = redacted_text
        super().__init__(message)


class ModelCostApiError(ModelCostError):
    """Raised when the ModelCost API returns an error response."""

    def __init__(
        self,
        message: str,
        status_code: int,
        error: str,
    ) -> None:
        self.status_code = status_code
        self.error = error
        super().__init__(message)


class SessionBudgetExceeded(ModelCostError):
    """Raised when a session's spend limit would be exceeded."""

    def __init__(
        self,
        message: str,
        session_id: str,
        current_spend: float,
        max_spend: Optional[float],
    ) -> None:
        self.session_id = session_id
        self.current_spend = current_spend
        self.max_spend = max_spend
        super().__init__(message)


class SessionIterationLimitExceeded(ModelCostError):
    """Raised when a session's iteration limit is reached."""

    def __init__(
        self,
        message: str,
        session_id: str,
        current_iterations: int,
        max_iterations: int,
    ) -> None:
        self.session_id = session_id
        self.current_iterations = current_iterations
        self.max_iterations = max_iterations
        super().__init__(message)
