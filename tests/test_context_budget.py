from core.generator import auto_input_budget, build_generator, build_prompt_within_budget


def _chunks(count, size):
    return [
        {"text": ("word " * size)[:size], "source": "a.txt", "location": f"Chunk {i}", "rerank_score": 10 - i}
        for i in range(count)
    ]


def test_auto_budget_grows_with_k_and_chunk_size():
    assert auto_input_budget(30, 1000) > auto_input_budget(4, 1000) > auto_input_budget(4, 300)


def test_default_budget_drops_chunks_but_auto_budget_keeps_all_k():
    chunks = _chunks(8, 1000)
    _, default_used, _ = build_prompt_within_budget("what?", chunks)
    budget = auto_input_budget(8, 1000)
    _, auto_used, tokens = build_prompt_within_budget("what?", chunks, budget)
    assert len(default_used) < 8          # the old fixed budget silently ignored most of K
    assert len(auto_used) == 8            # the sized budget really delivers K chunks
    assert tokens <= budget


def test_answer_length_setting_reaches_the_model():
    llm = build_generator(provider="ollama", model_name="qwen2.5:7b", max_tokens=321, streaming=False)
    params = " ".join(str(getattr(llm, name, "")) for name in ("num_predict", "max_tokens", "max_completion_tokens"))
    assert "321" in params
