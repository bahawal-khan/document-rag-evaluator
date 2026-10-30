"""
LIVE-PATH safety checks — deliberately NOT LLM-based (to stay fast; the
latency budget is 2-2.5s, and an extra LLM-judge call can't fit in that).

Deep/thorough Toxicity and PII checks (DeepEval's LLM-based metrics) run in
the offline Evaluation Suite instead — this is just a fast, regex-based
"tripwire" for the live path.
"""

import re

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")
CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
SSN_LIKE_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# Match complete words and phrases. Broad substring checks blocked ordinary
# research language such as "cell killing" and "killing rate".
TOXICITY_RE = re.compile(
    r"\b(?:idiot|stupid)\b|\bhate\s+you\b|\bshut\s+up\b|\bnonsense\s+fool\b",
    flags=re.IGNORECASE,
)


def check_pii(text: str):
    """Returns a list of PII types found (empty list = clean)."""
    found = []
    if EMAIL_RE.search(text):
        found.append("email")
    if CREDIT_CARD_RE.search(text.replace(" ", "")):
        found.append("credit_card_like")
    if SSN_LIKE_RE.search(text):
        found.append("ssn_like")
    if PHONE_RE.search(text):
        found.append("phone_number")
    return found


def check_toxicity(text: str):
    """Return True for direct abuse phrases without substring false positives."""
    return bool(TOXICITY_RE.search(text or ""))
