# AI Insurance Claims Processing Agent

An AI agent that processes insurance claims — reading documents, extracting key fields, validating consistency, and deciding what to do next.

## Setup

**API adapters (Gemini or Qwen API) — no GPU required:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env — add GEMINI_API_KEY (or QWEN_API_KEY + QWEN_BASE_URL)
```

**Local Qwen inference (`model_id: "qwen_local"`) — requires CUDA/MPS GPU:**

```bash
pip install -r requirements.txt -r requirements_localmodel.txt
# 8 GB+ VRAM for 3B model, 16 GB+ for 7B
```

## Choosing a Model

Edit `config/settings.yaml`:

```yaml
model_id: "gemini"      # Gemini 2.5 Flash (default) — needs GEMINI_API_KEY
# model_id: "qwen"      # Qwen API via DashScope  — needs QWEN_API_KEY + QWEN_BASE_URL
# model_id: "qwen_local"# Qwen VLM on local GPU   — needs requirements_localmodel.txt
```

To change the local model size (e.g. 7B instead of 3B), update `qwen_local.model` in the same file:

```yaml
qwen_local:
  model: "Qwen/Qwen2.5-VL-7B-Instruct"  # any Qwen VL Hub ID works
```

## Usage

```bash
# Process a single claim (interactive mode — agent will prompt for reply if needed)
python main.py --claim claims/CLM-001

# Process all 5 claims and show priority order
python main.py --all
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full design documentation.

Four-layer system:

```
Chatbot  (IO only)
  └── ClaimAgent  (LangGraph controller)
        └── ClaimParser  (validation + reply handling)
              └── DocReader  (PDF / Image / Text)
                    └── LLMAdapters  (Gemini / Qwen API / Qwen Local)
```

All source modules live in `core/`. Configuration is externalised to `config/`.

## Tool Design

| Tool | Trigger | Rationale |
|------|---------|-----------|
| `parse_documents` | on_start | Reads every file in the claim folder |
| `cross_validate` | after all docs parsed | Finds field inconsistencies across documents |
| `generate_customer_message` | status is incomplete or pending | Notifies customer of issues |
| `accept_reply` | on inbound message | Re-evaluates claim after customer responds |

Routing is **conditional** (LangGraph decides based on claim state), not a hardcoded pipeline. This is appropriate for a compliance-sensitive domain where unpredictable LLM-driven tool selection would be a liability.

## Key Decisions

- **`pending` vs `needs_review`**: `pending` describes ownership (company staff must act), not just state.
- **Confidence is immutable**: assigned at extraction time, never modified. Every value is traceable to its source.
- **Customer replies cannot resolve inconsistencies**: a trust model decision — even a matching reply doesn't eliminate a conflict between original documents.
- **Duplicate routing is type-aware**: same filename → `incomplete` (user can fix); same content, different filename → `pending` (data integrity concern requiring human review).

## Testing

There are two types of tests:

**Automated tests** (pytest, no human review needed):
```bash
pytest tests/ -v                          # all unit tests
pytest tests/ -v -k "not integration"     # skip integration tests
pytest tests/integration/ -v -m integration  # integration tests (requires API key)
```

Test files follow the `auto_test_*.py` naming convention so they are easy to distinguish from demo notebooks.

**Interactive demos** (Jupyter, human-verified output):
```bash
pip install jupyter
jupyter notebook demo/demo_doc_parsing.ipynb
```

`demo/demo_doc_parsing.ipynb` — parse individual claim documents and inspect extracted fields with inline document previews. Useful for verifying VLM output quality across different models.

## What I'd Do With More Time

- Async document processing (DocReader is stateless — easy to parallelise)
- Confidence-based model fallback (retry low-confidence extractions with a larger model)
- Persistent state (claim_state.json → PostgreSQL)
- Web UI (replace Chatbot with FastAPI + Gradio)
- Jac/byLLM integration (Jaseci ecosystem bonus)
