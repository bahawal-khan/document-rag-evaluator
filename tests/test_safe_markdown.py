from core.safe_markdown import neutralize_markdown, safe_stream


def test_inline_and_reference_images_become_text():
    assert "![" not in neutralize_markdown("a ![x](https://evil.example/?q=1) b")
    assert "![" not in neutralize_markdown("![x][r]\n\n[r]: https://evil.example/")


def test_links_and_formatting_survive():
    text = "See [docs](https://example.com) and **bold** `code`"
    assert neutralize_markdown(text) == text


def test_marker_split_across_stream_chunks_is_still_neutralised():
    out = "".join(safe_stream(["look ", "!", "[x](https://evil.example/?q=", "1)", "!"]))
    assert "![" not in out
    assert out.endswith("!")
    assert "https://evil.example" in out  # shown as text, never fetched as an image
