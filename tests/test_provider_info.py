import pytest

from core.model_settings import MODELS, PROVIDERS
from core.provider_info import PROVIDER_GUIDE, provider_usage_hint
from core.rate_limit import QuotaExhaustedError, plan_rate_limit_retry


def test_every_provider_has_guide_and_models():
    for label, (provider, default_model) in PROVIDERS.items():
        assert provider in PROVIDER_GUIDE
        assert default_model in MODELS[provider]


def test_quota_error_names_the_users_own_provider():
    err = "You exceeded your current quota, please check your plan and billing details."
    with pytest.raises(QuotaExhaustedError) as info:
        plan_rate_limit_retry(err, 0, 3, provider="openai")
    assert "platform.openai.com/usage" in str(info.value)
    assert "groq" not in str(info.value).lower()


def test_ollama_has_no_usage_hint():
    assert provider_usage_hint("ollama") == ""
