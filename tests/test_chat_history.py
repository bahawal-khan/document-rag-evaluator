import os

import pytest

import core.chat_history as ch


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    monkeypatch.delenv("CHAT_HISTORY_DATABASE_URL", raising=False)
    ch._failed_until.clear()
    ch._schema_ready_for.clear()
    yield
    ch._failed_until.clear()
    ch._schema_ready_for.clear()


def test_history_is_off_without_a_database_url():
    assert ch.is_enabled() is False
    key = ch.user_key("a@b.c")
    assert ch.list_chats(key) == []
    assert ch.create_chat(key, "hello") is None
    assert ch.add_message(key, "x", "user", "hi") is False
    assert ch.load_messages(key, "x") is None


def test_user_key_is_stable_and_does_not_contain_the_email():
    assert ch.user_key("A@B.c ") == ch.user_key("a@b.c")
    assert "a@b.c" not in ch.user_key("a@b.c")
    assert ch.user_key("a@b.c") != ch.user_key("other@b.c")
    assert ch.user_key("") is None and ch.user_key(None) is None


def test_titles_are_short_single_line_text():
    assert ch.make_title("  hello \n world ") == "hello world"
    assert len(ch.make_title("x" * 500)) <= ch.TITLE_CHARS
    assert ch.make_title("   ") == "New chat"


def test_an_unreachable_database_never_raises(monkeypatch):
    monkeypatch.setenv("CHAT_HISTORY_DATABASE_URL", "postgresql://nobody@127.0.0.1:1/postgres")
    key = ch.user_key("a@b.c")
    assert ch.list_chats(key) == []
    assert ch.create_chat(key, "hello") is None
    assert ch.delete_all_chats(key) == 0
    assert ch.last_error is not None


TEST_DB = os.getenv("TEST_DATABASE_URL")


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run the database tests")
def test_chats_are_saved_loaded_and_private(monkeypatch):
    monkeypatch.setenv("CHAT_HISTORY_DATABASE_URL", TEST_DB)
    mine, theirs = ch.user_key("mine@test.dev"), ch.user_key("theirs@test.dev")
    ch.delete_all_chats(mine)
    ch.delete_all_chats(theirs)
    chat_id = ch.create_chat(mine, "A question")
    assert chat_id
    assert ch.add_message(mine, chat_id, "user", "hello")
    assert ch.add_message(mine, chat_id, "assistant", "hi", "meta")
    assert ch.load_messages(mine, chat_id) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi", "meta": "meta"},
    ]
    # another user can neither read, write nor delete it
    assert ch.load_messages(theirs, chat_id) is None
    assert ch.add_message(theirs, chat_id, "user", "intruder") is False
    assert ch.delete_chat(theirs, chat_id) is False
    assert [chat["id"] for chat in ch.list_chats(mine)] == [chat_id]
    assert ch.list_chats(theirs) == []
    assert ch.delete_chat(mine, chat_id) is True
    assert ch.load_messages(mine, chat_id) is None


@pytest.mark.skipif(not TEST_DB, reason="set TEST_DATABASE_URL to run the database tests")
def test_old_chats_are_dropped_beyond_the_limit(monkeypatch):
    monkeypatch.setenv("CHAT_HISTORY_DATABASE_URL", TEST_DB)
    monkeypatch.setattr(ch, "MAX_CHATS_PER_USER", 3)
    key = ch.user_key("limit@test.dev")
    ch.delete_all_chats(key)
    for index in range(5):
        ch.create_chat(key, f"chat {index}")
    assert len(ch.list_chats(key)) == 3
    ch.delete_all_chats(key)
