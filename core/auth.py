"""Optional Google (OIDC) sign-in built on Streamlit's own ``st.login``.

Sign-in is switched on by an ``[auth]`` table in the Streamlit secrets. Without
it (for example when running locally) the app works exactly as before, so a
missing or half-finished configuration never locks the owner out of testing.
"""

import streamlit as st


def auth_configured() -> bool:
    """True when the secrets contain a usable ``[auth]`` section and this Streamlit supports it."""
    if not hasattr(st, "login") or not hasattr(st, "user"):
        return False
    try:
        config = st.secrets.get("auth")
    except Exception:
        return False
    if not config:
        return False
    try:
        return bool(config.get("client_id")) and bool(config.get("client_secret")) and bool(config.get("redirect_uri"))
    except Exception:
        return False


def _signed_in_email() -> str | None:
    user = st.user
    if not getattr(user, "is_logged_in", False):
        return None
    if getattr(user, "email_verified", True) is False:
        return None  # an unverified address must not count as an identity
    return str(getattr(user, "email", "") or "") or "signed-in user"


def require_login() -> str | None:
    """Stop the page with a sign-in screen unless the visitor is signed in.

    Returns the signed-in email, or None when sign-in is not configured.
    """
    if not auth_configured():
        return None
    email = _signed_in_email()
    if email:
        with st.sidebar:
            st.caption(f"Signed in as {email}")
            st.button("Sign out", key="sign_out_button", on_click=st.logout)
        return email

    st.title("Document RAG Evaluator")
    st.write("Sign in to upload documents, chat with them and run evaluations.")
    try:
        st.button("Sign in with Google", type="primary", key="sign_in_button", on_click=st.login)
    except Exception:
        st.error("Sign-in is not configured correctly. Please contact the app owner.")
    st.caption("We only use your name and email to identify you. See the Privacy page in the sidebar.")
    st.stop()
