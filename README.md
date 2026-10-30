<div align="center">

# 📄 Document RAG Evaluator

### Chat with your documents — then *measure* how good the answers really are.

[![Live App](https://img.shields.io/badge/🚀_Live_App-Open-5b5bf0?style=for-the-badge)](https://document-rag-evaluator.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-130_passing-2ea44f?style=for-the-badge)](#-run-the-tests)

**[👉 Try it live: document-rag-evaluator.streamlit.app](https://document-rag-evaluator.streamlit.app)**

</div>

---

## ✨ What is this?

Upload a **PDF, DOCX, TXT or Markdown** file, ask questions, and get answers **grounded only in your document** with citations.
Then open the **Evaluation Suite** to score the system with LLM-judge metrics and download a PDF report.

Bring your own API key — it lives only in your browser session and is never saved.

## 🎯 Highlights

| | Feature |
|---|---|
| 💬 | **Cited answers** from your own documents, off-topic questions are refused |
| 🔎 | **Hybrid search**: dense vectors + BM25 keyword search + optional cross-encoder reranker |
| 🤖 | **Any provider**: Groq, OpenAI, Gemini, Anthropic, OpenRouter, Ollama (or type any model ID) |
| 📊 | **Evaluation Suite** built on DeepEval: retrieval, answer quality, safety, scope |
| 🧪 | **Golden dataset generator**: creates test questions from your own documents |
| ⚡ | **Free or paid keys**: pacing profiles, *Fast mode*, honest progress bar and cost estimate |
| 👥 | **Many users at once**: each visitor gets an isolated workspace, queue and rate limits |
| 🔒 | **Hardened**: upload limits, XML-bomb protection, no remote images in answers |

## 🧭 How it works

```mermaid
flowchart TD
    A["📤 Upload<br/>PDF · DOCX · TXT · MD"] --> B["✂️ Chunk + embed"]
    B --> C[("🗄️ Chroma vectors<br/>+ SQLite keyword index")]
    Q["❓ Your question"] --> R["🔎 Hybrid search<br/>dense + BM25"]
    C --> R
    R --> K["🎯 Rerank<br/>cross-encoder"]
    K --> G["🤖 LLM answer<br/>with citations"]
    G --> U["✅ You"]

    classDef input fill:#e0e7ff,stroke:#5b5bf0,color:#1e1b4b
    classDef store fill:#fef3c7,stroke:#f59e0b,color:#451a03
    classDef step fill:#fce7f3,stroke:#db2777,color:#500724
    classDef out fill:#dcfce7,stroke:#16a34a,color:#052e16
    class A,Q input
    class C store
    class B,R,K,G step
    class U out
```

### 📊 The evaluation loop

```mermaid
flowchart TD
    D["📚 Your documents"] --> GD["🧪 Golden questions<br/>made by an LLM"]
    GD --> RUN["▶️ Run your RAG<br/>on every question"]
    RUN --> J["⚖️ LLM judge<br/>scores each answer"]
    J --> M["📈 Metrics"]
    M --> P["📄 PDF report"]

    classDef a fill:#e0e7ff,stroke:#5b5bf0,color:#1e1b4b
    classDef b fill:#fce7f3,stroke:#db2777,color:#500724
    classDef c fill:#dcfce7,stroke:#16a34a,color:#052e16
    class D,GD a
    class RUN,J b
    class M,P c
```

<details>
<summary><b>📏 Which metrics are measured?</b></summary>

| Group | Metrics |
|---|---|
| 🔍 **Retriever** | Contextual Precision · Recall · Relevancy |
| ✍️ **Generator** | Answer Relevancy · Faithfulness |
| 🎓 **Quality** | Correctness · Completeness |
| 🛡️ **Safety** | Toxicity · PII Leakage · Off-topic scope checks |

</details>

## 🚀 Use it online

1. Open **[the live app](https://document-rag-evaluator.streamlit.app)**
2. In the sidebar pick a **provider** and paste your **API key**
3. Upload a document and press **Ingest Papers**
4. Ask questions 💬
5. Open **Evaluation Suite** (left menu) → generate a golden dataset → run evaluation

> 💡 **Free-tier key?** Choose *Provider plan → Free* and turn on *Fast mode* to avoid rate limits.

## 💻 Run locally

```bash
git clone https://github.com/bahawal-khan/document-rag-evaluator.git
cd document-rag-evaluator

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
streamlit run app.py
```

Optional settings live in [`.env.example`](.env.example) — copy it to `.env`.

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repo to GitHub
2. On [share.streamlit.io](https://share.streamlit.io) choose the repo, branch `main`, file `app.py`
3. **Advanced settings** → Python **3.12 or 3.13**
4. **Secrets** → `EVAL_MAX_WORKERS = "1"` (Cloud has ~2.7 GB RAM)

## ⚙️ Useful settings

| Variable | Default | What it does |
|---|---|---|
| `EVAL_MAX_WORKERS` | 2 | Evaluations running at once |
| `MAX_WORKSPACES` | 300 | Maximum active visitor sessions |
| `MAX_CHAT_PER_MINUTE` | 20 | Questions per visitor per minute |
| `MAX_WORKSPACE_INDEX_MB` | 400 | Largest index one visitor can build |
| `INGEST_MAX_CONCURRENT` | 2 | Visitors indexing at the same time |

## 🧪 Run the tests

```bash
pip install -r requirements-dev.txt
pytest tests -q
```

## 🗂️ Project layout

```text
app.py                  chat page
pages/                  Evaluation Suite page
core/                   retrieval, ingestion, providers, rate limits, jobs
eval/                   DeepEval metrics and PDF report
tests/                  automated tests
```

## ⚠️ Good to know

- Your API key is kept **only in your browser session**.
- Uploaded files are stored per visitor and **deleted after inactivity**.
- Scores from a small question set are a *diagnostic*, not proof — use 20+ questions before trusting them.

## 📜 License

MIT License. See [LICENSE](LICENSE).

<div align="center">

Built by **[Bahawal Khan](https://github.com/bahawal-khan)** · ⭐ Star the repo if it helped!

</div>