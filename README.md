# AI Insurance Claims Processing Agent

An AI agent that processes insurance claims — reading documents, extracting key fields, validating consistency, and deciding what to do next.

See 
* [ARCHITECTURE.md](ARCHITECTURE.md) for system design documentation.
* [Slides](https://github.com/RUI030/Jaseci-Lab-AI-Engineer-take-home/blob/main/demo/Slide_TechDeepDive.pdf) for interview.
* [Download for visual Demo](https://github.com/RUI030/Jaseci-Lab-AI-Engineer-take-home/blob/main/demo/claim_report.html)
---

## Setup

Requirements are split by use case — install only what you need:

| File | When to install |
|---|---|
| `requirements/basic.txt` | Always — core deps for CLI and automated tests |
| `requirements/localmodel.txt` | Only for `model_id: "qwen_local"` (local GPU inference) |
| `requirements/demo.txt` | Only for running Jupyter notebooks in `demo/` |

**API adapters (Gemini or Qwen API) — no GPU required:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/basic.txt
cp .env.example .env
# edit .env — add GEMINI_API_KEY (or QWEN_API_KEY + QWEN_BASE_URL)
```

**Local Qwen inference (`model_id: "qwen_local"`) — requires CUDA/MPS GPU:**

```bash
pip install -r requirements/basic.txt -r requirements/localmodel.txt
# 8 GB+ VRAM for 3B model, 16 GB+ for 7B
```

**Interactive demo notebooks:**

```bash
pip install -r requirements/basic.txt -r requirements/demo.txt
```

---

## Choosing a Model

Edit `config/settings.yaml`:

```yaml
model_id: "gemini"       # Gemini 2.5 Flash — needs GEMINI_API_KEY
# model_id: "qwen"       # Qwen API via DashScope — needs QWEN_API_KEY + QWEN_BASE_URL
# model_id: "qwen_local" # Qwen VLM on local GPU — needs requirements/localmodel.txt
```

To change the local model size (e.g. 7B instead of 3B), update `qwen_local.model`:

```yaml
qwen_local:
  model: "Qwen/Qwen2.5-VL-7B-Instruct"  # any Qwen VL Hub ID works
```

API retry behaviour (on rate-limit or transient errors) is configured under `retry:` in the same file:

```yaml
retry:
  max_attempts: 3
  base_delay_seconds: 2.0
  max_delay_seconds: 30.0
```

---

## Usage

```bash
# Process a single claim (interactive — agent will prompt for customer reply if needed)
python main.py --claim claims/CLM-001

# Process all claims and display priority ranking
python main.py --all

# Force re-processing, ignoring any cached results
python main.py --all --no-cache
python main.py --claim claims/CLM-001 --no-cache
```

Results are cached to `<claim_folder>/.cache/claim_state.json` after each run. On subsequent runs:
- `complete` and `needs_review` claims are loaded from cache and skipped — they are terminal states until staff intervenes (`--no-cache` to reset).
- `incomplete` claims are re-processed — any new `.txt` reply files in the folder are auto-loaded as customer replies.
- `--no-cache` forces a full re-run for every claim.

---

## Testing

**Unit tests** (no API key required):

```bash
pytest tests/ -v -m "not integration"
```

**Integration tests** (requires `GEMINI_API_KEY` in `.env`):

```bash
pytest tests/integration/ -v -m integration
```

**HTML report** (no API key, no Jupyter required):

```bash
python demo/gen_visual.py                        # writes demo/claim_report.html
python demo/gen_visual.py --output /tmp/out.html # custom output path
xdg-open demo/claim_report.html                 # Linux
open demo/claim_report.html                     # macOS
```

Reads from `.cache/claim_state.json` in each claim folder — run `python main.py --all` first if caches are missing. Generates a self-contained Bootstrap page with priority ranking, per-claim document/field/issue tables, decision summary, and outbound message log.

**Interactive demo notebooks** (human-verified output):

```bash
jupyter notebook demo/doc_parsing.ipynb
```

`demo/doc_parsing.ipynb` — parse individual claim documents and inspect extracted fields with inline document previews. Useful for verifying VLM output quality across different models.

`demo/claim_results.ipynb` — same data as the HTML report but rendered inline in Jupyter.

---

## Design Notes

**Status enum matches the spec:** `complete | incomplete | needs_review`. `incomplete` is reserved for customer-fixable problems (missing doc, same-filename duplicate). `needs_review` covers everything that requires human judgment. Unknown doc type on a non-required document (e.g. an adjuster note the VLM can't classify) does not block completion — supplementary docs that don't affect required fields are ignored.

**All statuses send a customer message.** `complete` sends a congratulatory confirmation; `needs_review` explains which specific item is uncertain, sets a 3–5 business day expectation, and offers resubmission to speed up review; `incomplete` requests missing docs or asks the customer to confirm our best-guess field values. Only `incomplete` enters the reply loop — `complete` and `needs_review` end after the message is sent.

**LLM-composed customer messages.** The primary path (`build_customer_message_llm`) passes structured claim context — status, document status, best-guess field values, and a running `conversation_summary` — to the LLM with guidelines from `config/messages.yaml`. The LLM composes a natural, context-aware email that acknowledges prior replies, presents our best-guess values for confirmation rather than asking open "which is correct?" questions, and distinguishes blocking issues (customer must act) from warning-level ones (staff will handle). A template-based fallback fires if the LLM call fails.

**Conversation summary tracks both sides.** After each customer reply the LLM updates `Claim.conversation_summary`, tracking customer answers (CONFIRMED/CORRECTED/PENDING) and what we have already told the customer (WE_SAID). Subsequent messages never contradict prior outbound messages or re-ask for information already provided. Fields the customer explicitly confirms are extracted from the summary at `confidence="medium"` and merged into `extracted_fields` — they can fill in gaps left by documents and override low-confidence (blurry OCR) extractions, but never override high-confidence document values.

**Priority scheduling is separate from claim processing.** `core/scheduler.py` contains stateless functions (`prioritize_claims`, `_is_express_eligible`, `_priority_key`) with no dependency on the LLM client or LangGraph graph. Sorting a list of claims does not require instantiating a full `ClaimAgent`.

**LangGraph usage:** LangGraph was chosen to learn the framework and explore where graph-based state machines add value. For this scope a plain loop would have been equally functional — the primary benefit was gaining familiarity with conditional edge routing and where it is preferable to nested `if/else`. Note: `ClaimState` holds non-serialisable objects (`llm_client`, `chatbot`, `parser`), so LangGraph checkpointing is not usable without refactoring those out of the state dict.

---

## Problems Encountered & Future Plans

**Problems encountered:**

- Gemini's `response_schema` does not accept all standard JSON Schema keywords (`$ref`, `additionalProperties`, `title`, etc.) — required a custom schema-flattening pass before every API call.
- Scanned PDFs with no text layer are silently handled by falling back from `PDFReader` to `ImageReader` when extracted text falls below a configurable threshold (`pdf_text_threshold` in `config/settings.yaml`).
- LangGraph state merging behaviour required explicit `reply_skipped` assignment on every branch to avoid stale state leaking across graph iterations.
- Gemini and Qwen APIs occasionally fail under high load; exponential-backoff retry (configurable in `settings.yaml`) was added to handle transient failures without surfacing them to the claim workflow.

**Future plans:**

- Async document processing (`DocReader` is stateless — straightforward to parallelise with `asyncio` or a thread pool)
- Confidence-based model fallback (retry low-confidence extractions with a larger model before escalating to human review)
- Persistent state store (`claim_state.json` → PostgreSQL or Redis)
- Web UI (replace `Chatbot` with FastAPI + Gradio; `Claim` already serialises to JSON via Pydantic)
- Jac/byLLM integration (Jaseci ecosystem)
- Content-based document type classification: the current approach uses the filename as a hint and asks the VLM to infer doc type from content. A more robust approach would be a dedicated lightweight classifier (fine-tuned BERT or a small VLM) trained on labelled insurance documents, decoupling type detection from field extraction and making it resilient to arbitrary filenames.
- Fraud risk scoring: customer-supplied data currently receives `source_trust="user_input"` — it informs the claim state but cannot override document-extracted values or substitute for required documents. The natural extension is an explicit fraud-risk layer: cross-reference VINs against a DMV or vehicle history API, flag claims where customer-stated values diverge significantly from all submitted documents, and integrate with external watchlists. The current trust model is intentionally conservative as a first line of defence; a dedicated fraud scorer would make that defence explicit and configurable.
