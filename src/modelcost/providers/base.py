"""Abstract base class for provider instrumentation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple


class BaseProvider(ABC):
    """Base class that every provider adapter must implement."""

    @abstractmethod
    def wrap(self, client: Any) -> Any:
        """Return an instrumented version of *client*.

        The returned object should behave identically to the original
        client but transparently track costs, enforce budgets, and scan
        for PII on every request.
        """
        ...

    @abstractmethod
    def extract_usage(self, response: Any) -> Tuple[int, int]:
        """Extract (input_tokens, output_tokens) from a provider response."""
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the canonical provider name (e.g. ``'openai'``)."""
        ...

    @abstractmethod
    def get_model_name(self, kwargs: dict[str, Any]) -> str:
        """Extract the model identifier from the call kwargs."""
        ...
