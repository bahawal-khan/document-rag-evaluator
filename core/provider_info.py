"""Per-provider guidance shown in the UI and in rate-limit errors.

Kept free of Streamlit imports so core modules can use it. Nothing here
hard-codes numeric limits: they differ by account tier and change often, so the
text points to the provider's own usage page instead.
"""

PROVIDER_GUIDE = {
    "groq": {
        "label": "Groq",
        "key_url": "https://console.groq.com/keys",
        "usage_url": "https://console.groq.com/settings/limits",
        "summary": "Very fast open-source models. The free tier has per-minute and per-day token limits.",
        "tip": "On the free tier use Fast mode and a small dataset; a daily token limit resets on a rolling window.",
    },
    "openai": {
        "label": "OpenAI",
        "key_url": "https://platform.openai.com/api-keys",
        "usage_url": "https://platform.openai.com/usage",
        "summary": "GPT models. Needs prepaid credit; limits grow with your usage tier.",
        "tip": "gpt-4o-mini is the cheapest good judge. A 'quota' error here normally means no credit left.",
    },
    "gemini": {
        "label": "Gemini",
        "key_url": "https://aistudio.google.com/apikey",
        "usage_url": "https://aistudio.google.com/usage",
        "summary": "Google Gemini models. A free tier exists with tight per-minute and per-day request limits.",
        "tip": "Use a Flash or Flash-Lite model for judging; the free tier works best with the 'Free tier' pacing option.",
    },
    "anthropic": {
        "label": "Anthropic",
        "key_url": "https://console.anthropic.com/settings/keys",
        "usage_url": "https://console.anthropic.com/settings/limits",
        "summary": "Claude models. Needs prepaid credit; rate limits depend on your usage tier.",
        "tip": "Anthropic has no embeddings API, so pick Local, OpenAI or Gemini under Search and embeddings.",
    },
    "openrouter": {
        "label": "OpenRouter",
        "key_url": "https://openrouter.ai/keys",
        "usage_url": "https://openrouter.ai/activity",
        "summary": "One key for many providers' models. ':free' models are heavily rate limited.",
        "tip": "Enter any model ID from openrouter.ai/models as a custom model. Paid models avoid the free-model caps.",
    },
    "ollama": {
        "label": "Ollama (local)",
        "key_url": "",
        "usage_url": "",
        "summary": "Models run on your own machine, so there is no key, quota or cost.",
        "tip": "Needs a running Ollama server that this app can reach; it will not work on Streamlit Cloud.",
    },
}


def provider_guide(provider: str | None) -> dict:
    return PROVIDER_GUIDE.get((provider or "").lower(), {})


def provider_usage_hint(provider: str | None) -> str:
    """One sentence telling the user where to check their own limits for this provider."""
    guide = provider_guide(provider)
    if guide.get("usage_url"):
        return f" Check your {guide['label']} limits and usage at {guide['usage_url']}."
    return ""
