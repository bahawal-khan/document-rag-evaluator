# 📄 Document RAG Assistant — with Full Evaluation Suite

A Streamlit RAG app for uploaded document sets, with a complete evaluation
suite built in: **component-level**, **pipeline-level**, **application-level
(quality + safety)**, and **operational (latency/cost)** — plus a downloadable
PDF report.

## What's inside

```
├── app.py                      # Main chat page (streaming RAG, live guardrails)
├── pages/
│   └── 1_Evaluation_Suite.py   # Offline eval suite + PDF report
├── core/
│   ├── config.py                # All thresholds/targets in one place
│   ├── ingest.py                 # PDF/DOCX/TXT/Markdown → chunks → ChromaDB
│   ├── retriever.py
│   ├── generator.py              # Streaming, token-budget, scope-check guardrails
│   ├── judge.py                  # Provider-neutral judge wrapper (retry + JSON repair)
│   ├── cache.py                  # Exact-match answer cache
│   ├── cost.py                   # Token counting + cost estimate
│   ├── safety_fast.py            # Regex-based PII/toxicity (live path, fast)
│   ├── tracer.py                 # Lightweight local tracing + latency percentiles
│   └── golden.py                 # Auto-generates a golden QA set from your papers
├── eval/
│   ├── component_eval.py         # Retriever + Generator metrics
│   ├── pipeline_eval.py          # RAG Triad (reuses component-eval's raw result)
│   ├── application_eval.py       # Quality (GEval) + Safety (Toxicity/PII/Scope)
│   └── report_pdf.py             # Builds the downloadable PDF report
├── papers/                       # Drop your PDFs here (or upload via UI)
└── data/workspaces/              # Per-session indexes, datasets, cache (gitignored)
```

## Responsive layout

The chat and evaluation pages adapt to mobile, tablet, and desktop viewport
widths. On narrow screens, multi-column sections stack vertically, page
spacing and headings scale down, and evaluation tabs and tables can scroll
horizontally without forcing the whole page wider than the device.

## Setup

```bash
python -m venv venv
venv\Scripts\activate                              # Windows PowerShell
pip install -r requirements.txt
copy .env.example .env                            # optional LangSmith tracing only
streamlit run app.py
```

On macOS/Linux, activate with `source venv/bin/activate` instead.

## Run locally

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

## Troubleshooting

- **Groq 404 / "Generated 0 of N questions"**: the Groq SDK appends
  `/openai/v1` itself, so the base URL must be `https://api.groq.com`. The app
  now normalizes the `/openai/v1` form, and the error message shows the URL that
  was actually called. A 404 for a model means the model ID is unavailable to
  your key; pick another.
- **`No module named 'torchvision'` in the terminal**: harmless log noise from
  Streamlit's file watcher touching optional Transformers image modules. The app
  filters it; torchvision is not needed. Fully restart Streamlit after updating.
- **`ConnectionResetError: [WinError 10054]`**: Windows asyncio noise when a
  connection closes. Evaluations now run synchronously and the known message is
  filtered; it does not affect results.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository; include `requirements.txt`,
   `.streamlit/config.toml`, and the `pages/`, `core/`, and `eval/` folders.
   Do not commit `.env`, API keys, uploaded documents, or workspace data.
2. In Streamlit Community Cloud, create an app from that repository and set
   the main file path to `app.py`. Keep the app private and invite only trusted
   viewers until an authenticated hosting setup has been chosen. Community
   Cloud supports private apps with email-based viewer access; see its
   [sharing guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app).
3. Deploy, then enter each user's provider credentials in the app sidebar.
   Do not put provider keys in the repository or Community Cloud Secrets.
   Configure LangSmith secrets only if external tracing is explicitly desired.
4. In **Advanced settings** pick Python 3.12 or 3.13 (the version is chosen
   there, not in the repo). Keep a single dependency file (`requirements.txt`);
   Community Cloud uses only the first one it finds. The app imports
   `sentence-transformers`/`torch` (a large install and several hundred MB of
   RAM), so a free-tier container can run out of memory under heavy use; reboot
   the app if that happens.
5. Operational knobs (optional) can go in the app's Secrets, e.g.
   `WORKSPACE_TTL_DAYS = "7"`; see `.streamlit/secrets.toml.example`.
6. Community Cloud local disk is not guaranteed persistent. Download generated
   datasets and evaluation reports; re-index documents after a restart if the
   workspace was removed.

### Configure models and API keys

Enter provider keys in the sidebar for each role. Keys are kept in Streamlit
session state only and are not written to `.env`, datasets, caches, or logs.
You can choose separate providers and model IDs for **answer generation**,
**golden dataset generation**, and the **LLM judge**. Ollama runs locally and
does not need a key.

Built-in adapters: OpenAI, Gemini, Anthropic, Groq, OpenRouter, and Ollama.
Use a lightweight model ID accepted by your provider to reduce cost. Provider
accounts, pricing, model availability, and rate limits are managed by the
provider; the app does not supply a key or proxy requests.

The shared Retrieval configuration lets each session select Dense Vector
Search, BM25 Keyword Search, or Hybrid Search; choose a local Sentence
Transformers, OpenAI, or Gemini embedding model; and enable or disable the
local cross-encoder reranker. Dense and hybrid indexing use the selected
embedding profile. Each profile gets its own collection so incompatible vector
dimensions cannot be mixed. API embedding keys are passed directly to provider
clients and are not written to Chroma metadata, cache files, job records, or
logs. BM25 queries do not call an embedding provider. Changing an embedding
profile requires re-indexing the documents under that profile.

### Supported uploads and isolation

Upload PDF, DOCX, TXT, or Markdown documents from the main page. Files are
parsed page-by-page (PDF), paragraph-by-paragraph (DOCX), or in bounded text
blocks, then upserted into a persistent Chroma index in batches. Re-uploading
identical content does not duplicate its index entries. The golden dataset,
answer cache, index, and operational traces use a separate random workspace
directory for each Streamlit session. Local workspace data is ignored by Git.

Hybrid retrieval gathers dense-vector and SQLite FTS5 BM25 candidates, merges their ranks, and optionally uses cross-encoder/ms-marco-MiniLM-L-6-v2 to rerank the final passages. If the cross-encoder is unavailable, the selected retriever's base ranking remains available. The first reranker use downloads model weights. The Evaluation Suite includes a retriever comparison that reports Precision@K, Recall@K, MRR, and nDCG@K on the same dataset, K, embedding profile, candidate depth, and reranker settings. Retrieval labels come from source/page references or explicit relevant-document fields; missing labels are reported as not applicable.

Configure RERANKER_MODEL, RERANKER_SCOPE_THRESHOLD, and SCOPE_DISTANCE_THRESHOLD in the server environment. BM25 has no calibrated distance threshold, so returned keyword matches are sent through the grounded prompt and the model can decline when the context does not support an answer. Calibrate scope thresholds on labeled in-scope and out-of-scope questions before deployment.

The uploader accepts at most three files per batch, capped at 25MB each and
75MB total; DOCX main XML is capped at 100MB expanded size. Streamlit buffers
uploads before indexing, so the per-file, total-batch, and message-size limits
are applied at both the server configuration and app layers. Local embeddings
need CPU and memory, so this is suitable for ordinary research corpora,
not an unlimited ingestion service. A one-million-page workload is **not**
feasible to promise on Streamlit Community Cloud: request limits, available
RAM/disk, CPU, timeouts, embedding throughput, and provider quotas vary. Very
large corpora need a dedicated persistent machine and an asynchronous worker
architecture, which is intentionally outside this Streamlit-only project.
Temporary local copies of uploads are removed after each indexing attempt.
Community Cloud's local filesystem is not durable across every restart or
redeployment; download or back up datasets you need to keep.

Session workspaces isolate application data between active Streamlit sessions.
Inactive workspace files and completed evaluation status records are removed
after seven days by default (`WORKSPACE_TTL_DAYS` can change the retention
period). This app does not implement its own account system. Do not make a
Community Cloud app public unless it is behind a separate access-control layer;
Community Cloud's private sharing is suitable for a small trusted group. A
public portfolio demo should use a separate deployment with strict resource
limits and no shared credentials.

## How the live chat stays fast

Every live answer goes through this order — each step can short-circuit
before an expensive LLM call happens:

1. **Cache check** — identical question asked before? Instant answer, $0 cost.
2. **Toxic input filter** — regex-based, blocks abusive prompts before any call.
3. **Scope check** — if the retriever's best match is too far (low similarity),
   the question is treated as off-topic and refused *without calling the LLM*.
4. **Input-budget trimming** — context is trimmed to a **1500-token estimate**
   for system + context + question; oversized questions are rejected, and
   output is capped at **650 tokens**. Input counting uses a character-based
   estimate, so actual provider tokenization can vary.
5. **Streaming** — the answer streams token-by-token (`st.write_stream`),
   and both time-to-first-token and total latency are recorded.
6. **PII output scan** — after generation, a fast regex scan flags emails,
   phone numbers, and card-like numbers in the response.

Heavier, LLM-judge-based checks (real Toxicity/PII/Faithfulness scoring)
are deliberately **not** run on the live path — they run in the offline
Evaluation Suite instead, where latency doesn't matter.

## Evaluation levels

| Level | Metrics | Where |
|---|---|---|
| **Component** | Contextual Precision, Recall, Relevancy (retriever) · Answer Relevancy, Faithfulness (generator) | `eval/component_eval.py` |
| **Pipeline** | Same 5 metrics, but scored as "did this question pass *all* of them together?" (RAG Triad, extended) | `eval/pipeline_eval.py` |
| **Application — Quality** | Correctness, Completeness (custom `GEval`) | `eval/application_eval.py` |
| **Application — Safety** | Toxicity, PII Leakage (DeepEval built-in) · Scope Adherence (custom `GEval` on off-topic probes) | `eval/application_eval.py` |
| **Operational** | P50/P95/P99 latency, cache hit rate, cost/query, reliability | `core/tracer.py`, computed from live chat traces |

All LLM-judge metrics use the same judge wrapper (retry-safe, handles provider rate limits and repairs malformed JSON automatically). Each model role uses the provider, model ID, and session-only key selected in the sidebar.

### Evaluation speed and rate limits

The selected component, quality, and content-safety metrics share one set of
generated answers and are submitted in a single DeepEval run. Scope adherence
uses its separate five off-topic probes. DeepEval concurrency stays at one,
and its scheduling throttle is zero. Provider request pacing defaults to zero;
set a provider-specific interval such as `GROQ_MIN_INTERVAL_SECONDS=2.5` when
the account tier needs local pacing. A 10-question run across all groups may
still take several minutes because metrics can make multiple model calls and
each request has provider latency and possible quota waits. Metrics may make
more than one model call per case, so the page's request/time estimate is a
lower bound rather than a completion-time guarantee.
Golden generation checkpoints its partial dataset after each batch. The evaluation page warns when a run contains fewer than 20 questions, and the full PDF is available only when every evaluation section has actual metric results. An evaluation run executes in a single background worker. Its phase and final
summaries are written atomically to a short-lived status file, so the page can
poll without holding the Streamlit script thread and a browser refresh can
reattach to the run. The worker queue allows one active and one waiting run;
process restarts still cancel active work.

When a provider returns a 429, generation and judge calls honor its
retry/reset hints when available, otherwise use bounded exponential backoff.
Provider requests have a 90-second timeout, and DeepEval evaluations have a
15-minute bounded task timeout by default; neither can hang forever. Adjust
the sample size or run fewer metric groups to reduce quota use.

### Many users at once

Each visitor gets an isolated workspace, and everything that can interfere is scoped per user:

- Rate-limit pacing and the adaptive back-off after a 429 are tracked per workspace, so one user hitting a quota never slows anyone else.
- Each evaluation run has its own judge and its own API key; keys are never written to disk or shared.
- Evaluations run in a background worker pool (`EVAL_MAX_WORKERS`, default 2; `EVAL_QUEUE_SLOTS`, default 4 per worker). A user can have one run active or queued at a time; extra runs wait in the queue and the page shows "Waiting for a free evaluation worker".
- DeepEval keeps one process-wide result store, so only its scoring call is serialized; retrieval and answer generation for different users still overlap.
- Streamlit Community Cloud has about 2.7 GB of memory, so keep `EVAL_MAX_WORKERS` at 1-2 there. Raise it on a larger host.
- A job left "running" by an app restart is shown as failed with a clear message instead of spinning forever.

Safety limits for a public deployment (all optional environment variables):

| Variable | Default | Purpose |
|---|---|---|
| `MAX_WORKSPACES` | 300 | Stops bots from filling the disk with empty sessions |
| `MAX_WORKSPACE_INDEX_MB` | 400 | Largest index one session may build |
| `MAX_CHUNKS_PER_DOCUMENT` | 20000 | Rejects files that would take hours to embed |
| `INGEST_MAX_CONCURRENT` | 2 | Users indexing documents at the same time |
| `LOCAL_EMBEDDING_ALLOWLIST` | empty | Extra local embedding models users may pick (comma separated) |
| `ALLOW_CUSTOM_LOCAL_MODELS` | off | Allow any local model name; use only on a private machine |

Model answers cannot load remote images (Markdown image markers are neutralised), so a malicious document cannot leak chat content through an image URL. If you set LangSmith keys on the server, every visitor's questions and documents are traced to YOUR LangSmith project; leave it off for public apps.

Free vs paid keys: pick **Provider plan** on the Evaluation page. *Free* paces requests conservatively, *Paid* runs at full speed, *Auto* starts fast and slows down only for the provider that rate-limits you. *Fast mode* skips DeepEval's written explanations, removing about a quarter of the judge calls.

## Latency & cost targets

| Target | Value |
|---|---|
| P50 latency | ≤ 2.0s (target), 2.5s hard limit |
| P99 latency | ≤ 3.5s (hard cut-off at 4.0s — see below) |
| Estimated input tokens/query | ≤ 1500 (system + context + question) |
| Output tokens/query | ≤ 650 |

How these are enforced:
- **Retriever is cached per session** (`st.session_state`) — it used to be
  rebuilt on every message, which reloaded the embedding model each time
  (1-3s of pure overhead per query). This was the single biggest latency fix.
- **Response deadline** (`HARD_REQUEST_DEADLINE_SECONDS`, default 4.0s) —
  the UI stops waiting for streamed output at the deadline, including when
  the connection stalls before the first token. The provider request runs in
  a background thread and cannot be forcibly canceled by Python, so provider
  work or billing may continue until its own request timeout expires.
- `TOP_K` (4) and `MAX_OUTPUT_TOKENS` (650) keep prompts and
  responses short, which helps latency and cost together.
- Percentiles use generated model-call latency only. Cache response latency
  and cache hit rate are reported separately, so cache-heavy traffic cannot
  make model latency appear faster. The PDF reports **NO DATA** when there are
  no generated latency samples.

### Cost
The UI estimates cost from a provider/model price table and approximate token
counts. Unknown custom and OpenRouter model routes are shown as unpriced rather
than using the wrong model's rate; Ollama reports no API token charge and does
not estimate local compute. Estimates exclude discounts and provider-specific
extras, and the price table should be checked against the selected provider's
current pricing before budgeting. See `core/cost.py` for the sources and
checked date.

## Repository checks

Runtime dependencies are pinned in `requirements.txt`; `pandas` is declared
directly for the evaluation page. GitHub Actions runs `pytest tests -q` on
pushes and pull requests using `requirements-dev.txt`.

## Online tracing (LangSmith)

LangSmith tracing is optional. Set these in `.env` only if you explicitly
want prompts and model outputs sent to LangSmith:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=rag-research-assistant
```

Every LangChain model call explicitly attaches a `LangChainTracer`
callback when these are set — this is more reliable than relying purely on
ambient env-var detection. The sidebar shows a **"Test LangSmith Connection"**
button that pings the LangSmith API directly and reports the real error if
something's wrong (bad key, wrong project, `langsmith` package missing, etc.)
— use it first if traces aren't showing up in the dashboard.

## Golden dataset and evaluation reports

Since your documents aren't known in advance, the app can **auto-generate**
a golden Q&A set by sampling chunks and asking the LLM to write a question +
answer per chunk (self-QA generation) — up to **500 questions**, generated
in **batches of 5 per model call**. The index is scanned in bounded pages
(only a fixed-size reservoir sample is kept in memory). A progress bar shows batch-by-batch progress,
and the UI reports exactly how many batches succeeded vs. failed (failed
batches — e.g. from a rate limit — are retried with backoff and JSON
auto-repair, same pattern as the judge) instead of silently returning fewer
questions than requested. Duplicate and empty questions are removed. The
dataset can be reviewed and edited in the table, saved, or uploaded as
`golden_dataset.json`.

Each question is tagged **easy / medium / hard** by the LLM itself, so the
Results tab can show a **pass-rate-by-difficulty** breakdown — useful for
seeing whether the RAG system is specifically weaker on harder questions.
Evaluation results include correctness/completeness, faithfulness, answer
relevancy, contextual relevance, and retrieval precision/recall, with failed
examples and judge explanations. Download JSON/JSONL reports, CSV metric
aggregates, or the existing PDF report.

**Important:** the evaluation suite has a *separate* "how many questions to
actually evaluate on" slider, independent of dataset size. The estimate
before running includes generated answers and scope probes, but is a lower
bound: DeepEval metrics can require multiple model calls, and provider latency
or 429 reset waits add time. Selected groups reuse the same generated answers
to avoid spending additional generation requests for each group.

## Understanding why a metric scored low

Every metric's summary keeps a few example judge explanations for failed
cases. The Results tab shows an expandable "Why did this score low?" section
per metric, and the PDF report has a dedicated appendix listing the worst-
scoring metrics with real example questions and the judge's reasoning —
so a low score always comes with an explanation, not just a number.

## Report

The Evaluation Suite page produces JSON and JSONL metric/failure reports,
a CSV of metric aggregates, and a PDF with an executive summary, full metric
tables, and live operational stats.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
