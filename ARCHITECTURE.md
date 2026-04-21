# Architecture

## Overview

This system is a four-layer AI agent for processing insurance claims. Each layer has a single, clearly defined responsibility. Layers communicate only with the layer directly adjacent to them.

```
Chatbot
  └── ClaimAgent
        └── Claim / ClaimParser
              └── DocReader
```

---

## Layer Definitions

### 1. Chatbot

**File:** `chatbot.py`

**Responsibility:** Manages the user-facing interface and session loop. Passes user input to `ClaimAgent` and displays responses. Contains no business logic.

**Input:** Raw user text (stdin or UI event)  
**Output:** Formatted agent response (stdout or UI component)

**Design note:** Implemented as CLI for this assignment. The interface is intentionally thin so it can be swapped for Gradio or a REST API without touching any other layer.

---

### 2. ClaimAgent

**File:** `claim_agent.py`

**Responsibility:** Controls the workflow. Decides which tool to call next based on the current `ClaimState`. Does not read documents or validate fields directly — it delegates to lower layers and acts on their results.

**Input:** User message or trigger event, current `ClaimState`  
**Output:** Next action decision, updated `ClaimState`, response message

**Framework:** LangGraph

**Tools dispatched:**
| Tool | When called |
|------|-------------|
| `parse_documents` | On initial claim intake |
| `cross_validate` | After all documents are parsed |
| `generate_customer_message` | When status is `incomplete` |
| `update_from_reply` | When customer reply is received |

**Design note:** `ClaimAgent` decides *when* to call tools at runtime based on `ClaimState`. The sequence is not hardcoded — it is driven by the state of the claim at each step.

---

### 3. Claim

**File:** `claim.py`

**Responsibility:** Pure data container for a single claim. Holds all state, document records, conversation history, and extracted field values. Contains no processing logic.

**Key attributes:**
```python
claim_id: str
doc_table: list[DocRecord]
extracted_fields: dict[str, ExtractedField]
validation_issues: list[ValidationIssue]
status: Literal["complete", "incomplete", "needs_review"]
conversation_history: list[dict]
reply_count: int
```

**Design note:** `Claim` can be imported independently by any future system (reporting, APIs, dashboards) without pulling in parsing or validation logic.

---

### 4. ClaimParser

**File:** `claim_parser.py`

**Responsibility:** All processing logic that operates on a `Claim`. Determines required documents, runs cross-document validation, resolves field conflicts, and determines claim status.

**Input:** A `Claim` instance  
**Output:** Mutates `claim.validation_issues`, `claim.extracted_fields`, and `claim.status`

**Key methods:**
```python
check_required_docs(claim: Claim) -> list[str]      # returns missing doc types
cross_validate(claim: Claim) -> list[ValidationIssue]
determine_status(claim: Claim) -> str
merge_reply(claim: Claim, reply_text: str) -> Claim  # multi-turn update
```

**Confidence score rules:**

Confidence is determined by rule, not by LLM self-report.

| Condition | Confidence |
|-----------|-----------|
| Field present, format valid, consistent across all docs | `high` |
| Field present, format valid, appears in only one doc | `medium` |
| Field present, format valid, source is customer reply | `medium` |
| Field present but format validation fails | `low` |
| Field extracted from image (OCR path) | capped at `medium` |
| Field absent | `low` (flagged as missing) |

LLM self-reported confidence is accepted as initial value only. Rule-based overrides are applied after extraction.

**Design note:** Separating `ClaimParser` from `Claim` means parsing strategy can be swapped (e.g. replacing Gemini with a rule engine) without changing the data structure.

---

### 5. DocReader

**File:** `doc_reader.py`

**Responsibility:** Reads a single file and returns extracted field values. Accepts `target_fields` from the caller so the VLM prompt is focused rather than open-ended. Handles all file-type-specific preprocessing internally.

**Input:** `file_path: str`, `target_fields: list[str]`  
**Output:** `dict[str, ExtractedField]` with value and confidence per field

**Subclasses:**
```
BaseDocReader
  ├── PDFReader     — sends directly to Gemini File API
  ├── ImageReader   — deskew → Gemini vision
  └── TextReader    — plain read, send as text prompt
```

**Factory:**
```python
DocReaderFactory.get_reader(file_path) -> BaseDocReader
```
Dispatch is based on file extension.

**Design note:** Passing `target_fields` into `DocReader` improves VLM accuracy on tables and structured layouts. The decision of *which* fields to request is made by `Claim`/`ClaimParser`, not by `DocReader` itself.

---

## Data Flow

```
User input
    │
    ▼
Chatbot ──────────────────────────────────────────────┐
    │                                                  │
    ▼                                                  │
ClaimAgent                                             │
    │  reads/writes ClaimState                         │
    ▼                                                  │
Claim ◄──── ClaimParser                               │
    │           (cross_validate,                       │
    │            determine_status)                     │
    │                                                  │
    ▼                                                  │
DocReader                                              │
    │  returns ExtractedField per field                │
    ▼                                                  │
Gemini VLM                                             │
                                                       │
Agent response ────────────────────────────────────────┘
    │
    ▼
Chatbot (display)
```

---

## Cache and File Structure

Each claim folder contains original input files and a `.cache/` directory for system-generated outputs.

```
claims/
└── CLM-001/
    ├── police_report.pdf
    ├── finance_agreement.png
    ├── customer_reply.txt
    └── .cache/
        ├── parsed_results.json     # DocReader output per file
        ├── claim_state.json        # Full serialised Claim object
        └── conversation.json       # Chatbot session history
```

Original input files are never modified. All system outputs go to `.cache/`. Cached `parsed_results.json` is used on subsequent runs to avoid re-calling the VLM API.

---

## Key Design Decisions

**Single LLM backend.** All layers use the same Gemini 2.5 Flash model. Multiple agents are not used because the claim volume and task complexity do not justify the orchestration overhead. Multi-agent architecture is noted in the README as a future scaling option.

**Target-field-aware DocReader.** Passing `target_fields` into the reader improves VLM accuracy on tables by focusing attention rather than relying on post-hoc parsing of a full document dump.

**Rule-based confidence scoring.** LLM self-reported confidence is unreliable due to training bias. Confidence is assigned deterministically based on format validation results and cross-document consistency.

**Claim and ClaimParser are separated.** `Claim` is a pure data container that can be used independently by future systems. `ClaimParser` holds the processing logic and can be replaced without changing the data structure.

**DocReader is stateless.** Each call is independent. State is managed entirely by `Claim`. This makes `DocReader` easy to test in isolation and safe to parallelise in a future implementation.

---

## Scale-up Considerations

The following are not implemented in this assignment but are designed for in the architecture:

- **Parallel document processing:** `DocReader` is stateless, so multiple files within a claim can be processed concurrently with `asyncio` or a thread pool.
- **Swappable LLM backend:** All VLM calls are isolated in `DocReader`. Replacing Gemini requires changes to one class only.
- **Persistent state:** `claim_state.json` is designed to be loaded into a database (PostgreSQL, Redis) without structural changes.
- **Swappable parsing strategy:** `ClaimParser` can be replaced with a rule-based engine or a fine-tuned model without touching `Claim`.
- **API layer:** `Chatbot` can be replaced with a FastAPI endpoint. `Claim` serialises cleanly to JSON via Pydantic.

