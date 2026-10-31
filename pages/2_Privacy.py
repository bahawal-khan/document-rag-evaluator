"""Public privacy policy page (also used as the privacy-policy link for Google sign-in)."""

import os

import streamlit as st

st.set_page_config(page_title="Privacy Policy", page_icon="🔒", layout="centered")

from core.ui import apply_responsive_styles
from core.workspace import WORKSPACE_TTL_DAYS

apply_responsive_styles()

st.title("Privacy Policy")
st.caption("Document RAG Evaluator")

retention_days = int(WORKSPACE_TTL_DAYS)

st.markdown(
    f"""
### What this app does
You upload documents, ask questions about them, and evaluate the answers. Everything you upload and create
lives in a private workspace tied to your browser session.

### What we collect
- **Sign-in:** if you sign in with Google, we receive your name and email address from Google. They are used only
  to identify you inside the app. We do not sell or share them.
- **Your documents and results:** uploaded files, the search index built from them, your question set, answers
  and evaluation results are stored in your workspace so the app can work. Inactive workspaces are deleted
  automatically after about **{retention_days} days**, and you can clear your workspace yourself from the sidebar.
- **Usage traces:** the app records question timing and cost estimates for the operational report in your own
  workspace. These are not shared with other users.

### API keys
Keys you type for OpenAI, Gemini, Anthropic, Groq, OpenRouter or any other provider are kept only in your current
session. The app does not write them to files or databases.

### Data sent to model providers
To answer questions and score results, the app sends your questions and excerpts of your documents to the model
provider **you** choose (for example Groq, OpenAI, Gemini or Anthropic). Their own privacy terms apply to that data.
If you use local embeddings, document text is embedded on the server and is not sent to an embedding provider.

### Sharing
We do not sell your data and do not share it with other users. Your workspace is isolated from other visitors.

### Your choices
You can clear your workspace at any time from the sidebar, or stop using the app. To ask about your data, use the
contact below.
"""
)

contact = os.getenv("CONTACT_EMAIL", "").strip()
try:
    contact = contact or str(st.secrets.get("contact_email", "")).strip()
except Exception:
    pass
if contact:
    st.markdown(f"### Contact\n{contact}")
