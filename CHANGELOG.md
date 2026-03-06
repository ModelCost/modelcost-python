# Changelog

All notable changes to `modelcost` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-01

### Added
- Initial alpha release
- Cost tracking with automatic provider wrapping (OpenAI, Anthropic, Google)
- Budget enforcement with configurable actions (alert, throttle, block)
- PII detection scanning for SSN, email, phone, API keys, and credit cards
- Token bucket rate limiting
- Automatic background telemetry flushing
- Sync and async client support
- Full type annotations with PEP 561 marker
