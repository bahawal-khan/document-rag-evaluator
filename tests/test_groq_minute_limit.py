from core.rate_limit import classify_rate_limit

GROQ_TPM = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` "
    "in organization `org_x` service tier `on_demand` on tokens per minute (TPM): Limit 8000, "
    "Used 7900, Requested 1200. Please try again in 54s. Need more tokens? Upgrade to Dev Tier "
    "today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
)


def test_groq_per_minute_limit_is_not_treated_as_billing():
    kind, wait = classify_rate_limit(GROQ_TPM)
    assert kind == "minute"
    assert wait is not None and wait < 120


def test_groq_daily_limit_still_detected():
    kind, _ = classify_rate_limit(GROQ_TPM.replace("tokens per minute (TPM)", "tokens per day (TPD)"))
    assert kind == "daily"


def test_real_billing_error_still_detected():
    kind, _ = classify_rate_limit("You exceeded your current quota, please check your plan and billing details.")
    assert kind == "billing"
