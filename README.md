# AI Insurance Claims Processing Agent

An AI agent that processes insurance claims — reading documents, extracting key fields, validating consistency, and deciding what to do next.

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design documentation.

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
- `complete` claims are loaded from cache and skipped.
- `incomplete` and `pending` claims are always re-processed (new documents may have been added).
- `--no-cache` forces a full re-run for every claim, including complete ones.

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

**Interactive demo notebooks** (human-verified output):

```bash
jupyter notebook demo/demo_doc_parsing.ipynb
```

`demo/demo_doc_parsing.ipynb` — parse individual claim documents and inspect extracted fields with inline document previews. Useful for verifying VLM output quality across different models.

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
