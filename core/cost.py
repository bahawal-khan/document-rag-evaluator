"""Token and provider/model-specific cost estimates for live chat.

Token counting is a character-based estimate (~4 characters per token), so
these figures are estimates, not billing records. Prices are USD per million
tokens, checked 2026-10-16. Update this table when providers change pricing.
"""

from datetime import date


# Standard text-token prices; no cache discounts or provider-specific surcharges.
# Sources: OpenAI https://developers.openai.com/api/docs/pricing/
#          Google https://ai.google.dev/gemini-api/docs/pricing
#          Anthropic https://platform.claude.com/docs/en/about-claude/pricing
#          Groq https://console.groq.com/docs/models
MODEL_PRICING_USD_PER_MILLION = {
    "openai": {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4o": (2.50, 10.00),
    },
    "gemini": {
        "gemini-3.8-flash": (0.75, 3.75),  # introductory rate through 2026-12-31
        "gemini-2.5-flash": (0.30, 2.50),
        "gemini-2.5-flash-lite": (0.10, 0.40),
    },
    "anthropic": {
        "claude-haiku-4-5": (1.00, 5.00),
        "claude-sonnet-4-6": (3.00, 15.00),
    },
    "groq": {
        "openai/gpt-oss-20b": (0.075, 0.30),
        "openai/gpt-oss-120b": (0.15, 0.60),
    },
}


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    *,
    provider: str | None = None,
    model_name: str | None = None,
) -> float | None:
    """Return estimated USD cost, or None when the selected model is unpriced.

    OpenRouter pricing depends on the routed provider and custom model IDs can
    change without notice, so unknown combinations are intentionally omitted.
    Local Ollama has no API token charge; local compute cost is not estimated.
    """
    provider = (provider or "").strip().lower()
    model_name = (model_name or "").strip().lower()
    if provider == "ollama":
        return 0.0
    rates = MODEL_PRICING_USD_PER_MILLION.get(provider, {}).get(model_name)
    if provider == "gemini" and model_name == "gemini-3.8-flash":
        rates = (0.75, 3.75) if date.today() < date(2027, 1, 1) else (1.50, 7.50)
    if rates is None:
        return None
    input_rate, output_rate = rates
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
