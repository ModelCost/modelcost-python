"""Common enumerations shared across the SDK."""

from __future__ import annotations

from enum import Enum


class BudgetAction(str, Enum):
    """Action to take when a budget threshold is reached."""

    ALERT = "alert"
    THROTTLE = "throttle"
    BLOCK = "block"


class BudgetScope(str, Enum):
    """Scope at which a budget policy applies."""

    ORGANIZATION = "organization"
    FEATURE = "feature"
    ENVIRONMENT = "environment"
    CUSTOM = "custom"


class BudgetPeriod(str, Enum):
    """Time period for budget evaluation."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class Provider(str, Enum):
    """Supported AI model providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AWS_BEDROCK = "aws_bedrock"
    COHERE = "cohere"
    MISTRAL = "mistral"
