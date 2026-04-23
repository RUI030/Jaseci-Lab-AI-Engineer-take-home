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
- Conditional after `cross_validate`:
  - `incomplete` + `reply_queue` non-empty → `accept_reply` (pre-loaded `.txt` reply, skip message)
  - all other cases → `generate_message` (customer always receives a decision notification)
- After `generate_message`: `incomplete` → `accept_reply`; `complete` or `needs_review` → END
- Conditional after reply: if not skipped and under max rounds → `cross_validate` again; else END
- Routing is deterministic/conditional, not LLM-driven (compliance requirement)

**`.txt` files as auto-replies**: during `parse_documents`, any `.txt` file is read as a customer reply, added to `ClaimState.reply_queue`, and recorded as `doc_type="customer_reply"` in `doc_table`. `node_accept_reply` drains the queue before falling back to stdin.

**DocReader dispatch** (`get_doc_reader`): selects `PDFReader`, `ImageReader`, or `TextReader` by file extension. `PDFReader` checks `client.supports_native_pdf()`: Gemini uploads the PDF directly via Files API; QwenLocal renders pages to images internally; Qwen API renders each page to PNG in PDFReader and passes them all.

**Field extraction flow**: Each `DocReader` calls the LLM with a structured prompt and expects a JSON response matching `_ExtractionResponse`. Fields are merged into `Claim.extracted_fields` — highest confidence wins per field name across all documents.

**Customer messages**: `ClaimParser.build_customer_message_llm` composes the outbound email via LLM using guidelines + claim context from `config/messages.yaml`. Context includes `conversation_summary` (rolling LLM-maintained summary), best-guess field values, and document status. Falls back to template-based `build_customer_message` on failure.

**Conversation summary**: `Claim.conversation_summary` is updated by `ClaimParser.update_conversation_summary` after each accepted reply. Tracks confirmed/corrected/pending items so the LLM doesn't re-ask for info already provided.

**Status determination** (`ClaimParser.determine_status`) priority order:
1. Missing required doc → `incomplete`
2. Same-filename duplicate → `incomplete`
3. Same-content duplicate → `needs_review`
4. Unresolved blocking inconsistency → `needs_review`
5. Parse failed or unknown doc type → `needs_review`
6. Low-confidence required field → `needs_review`
7. All required docs complete + all required fields valid → `complete`

**Customer replies**: always assigned `source_trust="user_input"` and `confidence="low"`. Replies cannot resolve inconsistencies — they can only add new comparison data to the conversation log.

**Cache behavior**: `complete` and `needs_review` are loaded from cache without re-processing. `incomplete` claims re-process on every run (picks up new `.txt` reply files). Use `--no-cache` to force full re-run.

**Claim state** is persisted to `<claim_folder>/.cache/claim_state.json` after each run.

## Configuration

All config is in `config/` — no magic constants in code:

| File | Purpose |
|------|---------|
| `settings.yaml` | Active `model_id` (`gemini`\|`qwen`\|`qwen_local`), model params, retry config |
| `field_schema.json` | Required/optional fields, data types, validation rules, unify instructions |
| `messages.yaml` | LLM guidelines for customer messages and conversation summaries; template fallback; priority reason strings |
| `workflow.yaml` | `max_reply_rounds` and other routing parameters |

To switch models, change `model_id` in `config/settings.yaml`. `QWEN_API_KEY` and `QWEN_BASE_URL` env vars override the yaml values for Qwen.

## Key Design Decisions

- **Confidence is immutable**: set at extraction time, never changed. Every value is traceable to its source document.
- **Duplicate routing is type-aware**: same filename → `incomplete` (user can fix); same content, different filename → `needs_review` (data integrity, requires human review).
- **Customer always gets a decision notification**: every claim status — `complete`, `incomplete`, `needs_review` — triggers a `generate_message` call so the customer is never left without a response. For `needs_review`, the message informs the customer that their claim is under manual review; staff then owns follow-up.
- **Confirmation-style messages**: for field inconsistencies, the LLM is given our best-guess value (`extracted_fields.unified_value`) and asks the customer to confirm or correct it — not an open "which is right?" question.
