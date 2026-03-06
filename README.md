# ModelCost Python SDK

Python SDK for the ModelCost API. Track, govern, and optimize your AI model spending in real time.

## Installation

```bash
pip install modelcost
```

For development:

```bash
pip install modelcost[dev]
```

## Quick Start

```python
import modelcost

# Initialize the SDK
modelcost.init(api_key="mc_your_api_key", org_id="org_123")

# Wrap your OpenAI client for automatic tracking
import openai
client = openai.OpenAI()
wrapped = modelcost.wrap(client)

# All calls are now tracked automatically
response = wrapped.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
)

# Or track costs manually
modelcost.track_cost(
    provider="openai",
    model="gpt-4o",
    input_tokens=150,
    output_tokens=50,
    feature="chatbot",
)

# Check budget before expensive operations
budget = modelcost.check_budget(feature="chatbot", estimated_cost=0.50)
if not budget.allowed:
    print(f"Budget exceeded: {budget.reason}")

# Scan text for PII before sending to models
result = modelcost.scan_pii("Contact me at test@example.com")
if result.detected:
    print(f"PII found: {result.entities}")

# Get current usage summary
usage = modelcost.get_usage()

# Flush any buffered events and shut down
modelcost.shutdown()
```

## Configuration

You can configure the SDK via environment variables:

| Variable | Description |
|---|---|
| `MODELCOST_API_KEY` | Your API key (must start with `mc_`) |
| `MODELCOST_ORG_ID` | Your organization ID |
| `MODELCOST_ENV` | Environment name (default: `production`) |
| `MODELCOST_BASE_URL` | API base URL (default: `https://api.modelcost.ai`) |

## License

MIT
