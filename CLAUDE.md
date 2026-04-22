# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/basic.txt
cp .env.example .env
# Add GEMINI_API_KEY to .env
```

## Commands

```bash
# Run a single claim
python main.py --claim claims/CLM-001

# Process all claims with priority ranking
python main.py --all

# All unit tests
pytest tests/ -v

# Unit tests only (skip integration)
pytest tests/ -v -m "not integration"

# Integration tests (requires GEMINI_API_KEY in .env)
pytest tests/integration/ -v -m integration

# Single test file
pytest tests/auto_test_agent.py -v

# Single test by name
pytest tests/auto_test_parser.py -v -k "test_cross_validate"

# Interactive demo (human-verified, requires jupyter)
jupyter notebook demo/demo_doc_parsing.ipynb
```

Test files use the `auto_test_*.py` prefix (not `test_*.py`). Interactive demos that require human review live in `demo/` as Jupyter notebooks.

## Architecture

Four-layer stack — each layer depends only on the one below it:

```
Chatbot        (core/chatbot.py)     — IO only, no business logic
ClaimAgent     (core/agent.py)       — LangGraph state machine controller
ClaimParser    (core/parser.py)      — validation, status determination, reply handling
DocReader      (core/doc_reader.py)  — PDF / Image / Text extraction via VLM
LLMAdapters    (core/llm_adapters.py)— Gemini and Qwen clients behind BaseLLMClient
```

**LangGraph graph** (`ClaimAgent._build_graph`):
- Entry → `parse_documents` → `cross_validate`
- Conditional: if status is `incomplete`/`pending` → `generate_message` → `accept_reply`
- Conditional after reply: if not skipped and under max rounds → `cross_validate` again; else END
- Routing is deterministic/conditional, not LLM-driven (compliance requirement)

**DocReader dispatch** (`get_doc_reader`): selects `PDFReader`, `ImageReader`, or `TextReader` by file extension. `PDFReader` falls back to `ImageReader` when extracted text is below `pdf_text_threshold` (set in `config/settings.yaml`).

**Field extraction flow**: Each `DocReader` calls the LLM with a structured prompt and expects a JSON response matching `_ExtractionResponse`. Fields are merged into `Claim.extracted_fields` — highest confidence wins per field name across all documents.

**Status determination** (`ClaimParser.determine_status`) priority order:
1. Missing required doc → `incomplete`
2. Same-filename duplicate → `incomplete`
3. Same-content duplicate → `pending`
4. Unresolved inconsistency before first reply → `incomplete`; after reply → `pending`
5. Parse failed or unknown doc type → `pending`
6. Low-confidence required field → `pending`
7. All required docs complete + all required fields valid → `complete`

**Customer replies**: always assigned `source_trust="user_input"` and `confidence="low"`. Replies cannot resolve inconsistencies — they can only add new comparison data to the conversation log.

**Claim state** is persisted to `<claim_folder>/.cache/claim_state.json` after each run.

## Configuration

All config is in `config/` — no magic constants in code:

| File | Purpose |
|------|---------|
| `settings.yaml` | Active `model_id` (`gemini`\|`qwen`), model params, `pdf_text_threshold` |
| `field_schema.json` | Required/optional fields, data types, validation rules, unify instructions |
| `messages.yaml` | Customer message templates and priority reason strings |
| `workflow.yaml` | `max_reply_rounds` and other routing parameters |

To switch models, change `model_id` in `config/settings.yaml`. `QWEN_API_KEY` and `QWEN_BASE_URL` env vars override the yaml values for Qwen.

## Key Design Decisions

- **Confidence is immutable**: set at extraction time, never changed. Every value is traceable to its source document.
- **Duplicate routing is type-aware**: same filename → `incomplete` (user can fix); same content, different filename → `pending` (data integrity, requires human review).
- **`pending` means ownership**: company staff must act, not just "something is wrong."
- **`messages` attribute on `ClaimState`** uses LangGraph's `add_messages` reducer but is not actively used for routing — it exists for future streaming support.
