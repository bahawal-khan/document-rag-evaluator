from types import SimpleNamespace

from core.followup import (
    MAX_CHARS_PER_MESSAGE,
    MAX_HISTORY_MESSAGES,
    clean_rewrite,
    format_history,
    history_for_rewrite,
    rewrite_followup,
)

HISTORY = [
    {"role": "user", "content": "What is the main finding of the paper?"},
    {"role": "assistant", "content": "The model improves recall by 12%."},
]


def test_without_history_the_question_is_unchanged():
    called = []
    assert rewrite_followup("and its limits?", [], lambda prompt: called.append(prompt)) == "and its limits?"
    assert called == []


def test_followup_uses_the_model_rewrite():
    seen = {}

    def invoke(prompt):
        seen["prompt"] = prompt
        return SimpleNamespace(content="What are the limitations of the model that improves recall by 12%?")

    result = rewrite_followup("and its limits?", HISTORY, invoke)
    assert result.startswith("What are the limitations")
    assert "main finding" in seen["prompt"] and "and its limits?" in seen["prompt"]


def test_model_failure_falls_back_to_the_original_question():
    def broken(prompt):
        raise RuntimeError("rate limited")

    assert rewrite_followup("and its limits?", HISTORY, broken) == "and its limits?"


def test_unusable_model_output_is_ignored():
    assert clean_rewrite("", "q") == "q"
    assert clean_rewrite("x" * 2000, "q") == "q"
    assert clean_rewrite(SimpleNamespace(content=None), "q") == "q"


def test_clean_rewrite_strips_labels_quotes_and_extra_lines():
    assert clean_rewrite('Standalone question: "What is X?"\nExplanation...', "q") == "What is X?"
    assert clean_rewrite([{"type": "text", "text": "What is Y?"}], "q") == "What is Y?"


def test_history_is_trimmed_and_ignores_empty_messages():
    many = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(20)]
    many.append({"role": "assistant", "content": "   "})
    kept = history_for_rewrite(many)
    assert len(kept) == MAX_HISTORY_MESSAGES
    assert kept[-1]["content"] == "m19"


def test_long_messages_are_shortened_in_the_prompt():
    text = format_history([{"role": "assistant", "content": "word " * 1000}])
    assert text.startswith("Assistant: ")
    assert len(text) <= MAX_CHARS_PER_MESSAGE + 20
