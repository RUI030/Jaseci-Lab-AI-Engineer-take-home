# AI Insurance Claims Processing Agent

An AI agent that processes insurance claims — reading documents, extracting key fields, validating consistency, and deciding what to do next.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your GEMINI_API_KEY
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
                    └── LLMAdapters  (Gemini / Qwen)
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

## Running Tests

```bash
pytest tests/ -v                          # all unit tests
pytest tests/ -v -k "not integration"     # skip integration tests
pytest tests/integration/ -v -m integration  # integration tests (requires API key)
```

## What I'd Do With More Time

- Async document processing (DocReader is stateless — easy to parallelise)
- Confidence-based model fallback (retry low-confidence extractions with a larger model)
- Persistent state (claim_state.json → PostgreSQL)
- Web UI (replace Chatbot with FastAPI + Gradio)
- Jac/byLLM integration (Jaseci ecosystem bonus)
