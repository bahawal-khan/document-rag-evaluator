"""Streamlit pieces that connect the chat page to saved chat history."""

from __future__ import annotations

import streamlit as st

from core import chat_history


def history_active() -> bool:
    """True when this visitor is signed in and the history database is configured."""
    return bool(st.session_state.get("chat_history_key")) and chat_history.is_enabled()


def setup_chat_history(email: str | None) -> None:
    """Remember who is signed in and drop the previous person's chat if someone else signed in."""
    key = chat_history.user_key(email)
    previous = st.session_state.get("chat_history_key")
    if previous is not None and previous != key:
        st.session_state["messages"] = []
        for name in ("active_chat_id", "saved_chats"):
            st.session_state.pop(name, None)
    st.session_state["chat_history_key"] = key


_DELETE_BUTTON_STYLE = """
<style>
[class*="st-key-delete_chat_"] button,
[class*="st-key-delete_all_chats_button"] button {
    color: #ff6b6b !important;
    background: rgba(255, 107, 107, 0.14) !important;
    border: 1px solid rgba(255, 107, 107, 0.65) !important;
}
[class*="st-key-delete_chat_"] button {
    padding-left: 0 !important;
    padding-right: 0 !important;
    min-width: 0 !important;
}
[class*="st-key-delete_chat_"] button p { font-size: 1.4rem; line-height: 1; }
[class*="st-key-delete_chat_"] button:hover,
[class*="st-key-delete_all_chats_button"] button:hover:not(:disabled) {
    color: #ffffff !important;
    background: #e5484d !important;
    border-color: #e5484d !important;
}
[class*="st-key-delete_all_chats_button"] button:disabled {
    opacity: 0.55;
}
</style>
"""


def _start_new_chat() -> None:
    st.session_state["messages"] = []
    st.session_state.pop("active_chat_id", None)


def remember_message(role: str, content: str, meta: str | None = None) -> None:
    """Add a message to the visible chat and, when history is on, save it."""
    message = {"role": role, "content": content}
    if meta:
        message["meta"] = meta
    st.session_state.messages.append(message)
    if not history_active():
        return
    key = st.session_state["chat_history_key"]
    chat_id = st.session_state.get("active_chat_id")
    if not chat_id:
        if role != "user":
            return
        chat_id = chat_history.create_chat(key, content)
        if not chat_id:
            return
        st.session_state["active_chat_id"] = chat_id
        st.session_state.pop("saved_chats", None)  # the list is re-read on the next run
    chat_history.add_message(key, chat_id, role, content, meta)


def render_chat_sidebar() -> None:
    """Memory switch plus the list of saved chats."""
    st.toggle(
        "Remember earlier messages in this chat",
        value=True,
        key="chat_memory_on",
        help=(
            "Follow-up questions such as \"what about its limitations?\" are rewritten using the "
            "last few messages so the right passages are found. This uses one extra short model call "
            "per follow-up."
        ),
    )
    if not history_active():
        return

    key = st.session_state["chat_history_key"]
    with st.expander("💬 Saved chats"):
        st.markdown(_DELETE_BUTTON_STYLE, unsafe_allow_html=True)
        if st.button("➕ New chat", key="new_chat_button", width="stretch"):
            _start_new_chat()
            st.rerun()

        if "saved_chats" in st.session_state:
            chats = st.session_state["saved_chats"]
        else:
            chats = chat_history.list_chats(key)
            if chat_history.last_error is None:  # do not cache a failed read as "no chats"
                st.session_state["saved_chats"] = chats
        if chat_history.last_error:
            st.warning(
                f"Chats cannot be saved right now: the database connection failed "
                f"({chat_history.last_error}). "
                + (chat_history.last_error_hint + " " if chat_history.last_error_hint else "")
                + "Check `database_url` in the app Secrets, then reboot the app."
            )
        active_id = st.session_state.get("active_chat_id")
        if not chats:
            st.caption("No saved chats yet. Your questions and answers are saved here automatically.")

        for chat in chats:
            open_column, delete_column = st.columns([4, 1])
            label = ("● " if chat["id"] == active_id else "") + chat["title"]
            if open_column.button(label, key=f"open_chat_{chat['id']}", width="stretch"):
                messages = chat_history.load_messages(key, chat["id"])
                if messages is None:
                    st.warning("This chat could not be opened. Try again in a moment.")
                else:
                    st.session_state["messages"] = messages
                    st.session_state["active_chat_id"] = chat["id"]
                    st.rerun()
            if delete_column.button("×", key=f"delete_chat_{chat['id']}", help="Delete this chat"):
                chat_history.delete_chat(key, chat["id"])
                if chat["id"] == active_id:
                    _start_new_chat()
                st.session_state.pop("saved_chats", None)
                st.rerun()

        if chats:
            confirm = st.checkbox("Delete all my saved chats", key="confirm_delete_all_chats")
            if st.button("Delete all", key="delete_all_chats_button", disabled=not confirm):
                chat_history.delete_all_chats(key)
                _start_new_chat()
                st.session_state.pop("saved_chats", None)
                st.rerun()
        st.caption("Questions, answers and sources are saved to your account. API keys are never saved.")
