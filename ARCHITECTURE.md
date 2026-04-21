# Architecture

## Overview

This system is a four-layer AI agent for processing insurance claims. Each layer has a single, clearly defined responsibility. Layers communicate only with the layer directly adjacent to them.

```
Chatbot  (IO only)
  └── ClaimAgent  (controls everything)
        └── Claim / ClaimParser
              └── DocReader
```

**Entry point assumption:** Input files are assumed to have been uploaded to a local folder before processing begins.

```python
chatbot = Chatbot()
agent = ClaimAgent(chatbot=chatbot)
agent.process_claim("claims/CLM-001/")
```

---

## Layer Definitions

### 1. Chatbot

**File:** `chatbot.py`

**Responsibility:** Pure IO layer. Renders output and captures user input on demand. Contains no business logic and no decision-making. `ClaimAgent` owns the conversation flow and calls `Chatbot` when it needs to interact with the user.

**Design note:** Implemented as CLI for this assignment. Swapping to Gradio or a REST API requires only replacing this class.

#### ADT

```python
class Chatbot:
    def ask(self, prompt: str) -> str          # called by ClaimAgent to get user input
    def display(self, message: str) -> None    # called by ClaimAgent to show output
```

---

### 2. ClaimAgent

**File:** `claim_agent.py`

**Responsibility:** Controls the entire workflow. Decides which tool to call, when to ask the user for input, and what to display. Reads workflow rules from `config/workflow.yaml` and message templates from `config/messages.yaml`. Does not read documents or validate fields directly.

**Design note:** `ClaimAgent` holds a reference to `Chatbot` and calls it when user interaction is needed. This keeps `Chatbot` as a passive IO tool and `ClaimAgent` as the single decision-maker.

**Framework:** LangGraph

**Tool dispatch:** Tools are not called in a hardcoded sequence. `ClaimAgent` uses LangGraph conditional edges to decide at runtime which tool to invoke next based on the current `ClaimState`. The trigger table below describes the conditions, not a fixed pipeline.

**Why deterministic routing is justified here:** Insurance claim processing has well-defined states and transition rules. Fully dynamic LLM-driven tool selection would introduce unpredictability in a compliance-sensitive domain. The right tradeoff is conditional dispatch (LangGraph decides based on state) rather than either a hardcoded sequence or unconstrained LLM autonomy.

**Async reply support via LangGraph checkpointing:** `ClaimAgent.accept_reply()`
is designed as a resumable entry point. `ClaimState` is fully serialisable
(all fields are Pydantic models), allowing LangGraph's checkpointer to
suspend execution after `generate_customer_message` and resume when a
reply arrives — without blocking a thread. The CLI implementation uses
synchronous `input()` for testing, but the state machine itself imposes
no blocking requirement. Replacing the CLI with a webhook or message
queue requires no changes to `ClaimAgent` logic.

#### ADT

```python
class ClaimAgent:
    chatbot: Chatbot
    llm_client: BaseLLMClient
    # single adapter instance, instantiated from model_id in config/settings.yaml
    workflow_config: dict       # loaded from config/workflow.yaml
    message_config: dict        # loaded from config/messages.yaml

    def __init__(self, chatbot: Chatbot) -> None
    def process_claim(self, folder_path: str, uploaded_at: str | None = None) -> Claim
    # uploaded_at: ISO 8601 string; falls back to folder mtime if None
    def accept_reply(self, claim: Claim) -> Claim
    # calls chatbot.ask() to get reply_text
    # then calls ClaimParser.handle_reply(claim, reply_text, self.llm_client)
    def prioritize_claims(self, claims: list[Claim]) -> list[PriorityRecord]
```

**Tools dispatched:**

| Tool | Trigger condition |
|------|------------------|
| `parse_documents` | on_start |
| `cross_validate` | after all documents parsed |
| `generate_customer_message` | status is `incomplete` or `pending` |
| `accept_reply` | on inbound customer message |

**Claim prioritization logic:**

```
Primary sort: status
  complete    → highest priority (ready to finalise immediately)
  incomplete  → second (notify user as early as possible)
  pending     → last (enters human review queue)

Secondary sort: uploaded_at (oldest first within same status group)

Express flag: applies to incomplete claims only
  condition:  unresolved validation_issues < 2
  effect:     flagged claim moves to front of the incomplete group
  rationale:  complete claims need no special routing;
              pending claims cannot bypass human review regardless
```

**Priority reason templates** are defined in `config/messages.yaml` under `priority_reason`. They are not generated ad hoc.

---

### 3. Claim

**File:** `claim.py`

**Responsibility:** Pure data container for a single claim. Holds all state, document records, conversation history, and extracted field values. Contains no processing logic.

**Design note:** `Claim` can be imported independently by any future system without pulling in processing logic.

#### ADT

```python
class Claim(BaseModel):
    claim_id: str
    status: Literal["complete", "incomplete", "pending"]
    reply_count: int
    uploaded_at: str                # ISO 8601; from argument or folder mtime
    doc_table: list[DocRecord]
    extracted_fields: dict[str, ExtractedField]
    validation_issues: list[ValidationIssue]
    conversation_log: list[ConversationRound]
```

**Status semantics:**

| Status | Meaning | Who acts |
|--------|---------|---------|
| `complete` | All required docs present, all fields valid, no unresolved issues | Nobody — ready to finalise |
| `incomplete` | User-side problem, system can tell user exactly what to do | User |
| `pending` | Requires human intervention, user cannot resolve alone | Company staff |

**Why `pending` over `needs_review`:** `needs_review` describes a state. `pending` describes ownership — it makes explicit that the ball is in the company's court. This maps more cleanly to real workflow routing.

**Status aggregation logic:**

```
any doc_status == "missing"                          → incomplete
any doc_status == "duplicate" (same_filename)        → incomplete
any doc_status == "duplicate" (same_content)         → pending
any field inconsistency (unresolved)                 → incomplete
user confirms docs correct, inconsistency remains    → pending
any parse_status == "parse_failed"                   → pending
any doc_type == "unknown"                            → pending
any low_confidence field (unresolved)                → pending
all required docs present + all fields valid         → complete
```

---

### 4. ClaimParser

**File:** `claim_parser.py`

**Responsibility:** All processing logic that operates on a `Claim`. Each method has a single responsibility. `handle_reply` is the orchestrator for reply processing — it calls the smaller methods rather than implementing everything inline.

**Design note:** Separating `ClaimParser` from `Claim` means processing strategy can be swapped without changing the data structure.

#### ADT

```python
class ClaimParser:

    def check_required_docs(
        self, claim: Claim
    ) -> list[str]
    # returns list of missing required doc types

    def cross_validate(
        self, claim: Claim
    ) -> list[ValidationIssue]
    # compares unified_value of same field across documents
    # only processes source_trust == "document"
    # never modifies confidence values

    def determine_status(
        self, claim: Claim
    ) -> Literal["complete", "incomplete", "pending"]
    # evaluates doc_table and validation_issues

    def parse_reply(
        self, reply_text: str,
        target_fields: list[FieldSchema],
        client: BaseLLMClient
    ) -> list[ExtractedField]
    # calls TextReader with XML isolation
    # source_trust hard-set to "user_input", confidence hard-capped at low

    def compare_fields(
        self, claim: Claim,
        new_fields: list[ExtractedField]
    ) -> dict[str, Literal["consistent", "inconsistent"]]
    # compares new_fields against claim.extracted_fields (unified_value only)
    # user_input fields never resolve existing ValidationIssues

    def log_reply(
        self, claim: Claim,
        message: str,
        compare_results: dict
    ) -> None
    # appends ConversationRound to claim.conversation_log

    def handle_reply(
        self, claim: Claim,
        reply_text: str,
        client: BaseLLMClient
    ) -> Claim
    # orchestrator: parse_reply → compare_fields → log_reply → determine_status
```

**Confidence score rules:**

Confidence is assigned by `DocReader` at extraction time and never modified afterwards.

| Condition | Confidence |
|-----------|-----------|
| Value found, format valid, source is clean PDF | `high` |
| Value found, format valid, source is scanned image | `medium` |
| Value found, source is customer reply | `low` (hard cap) |
| Value found, format validation failed | `low` |
| Value not found | `low` |

**Customer reply handling:**

```
reply_text
    │
    ▼
TextReader (XML-isolated, source_trust = "user_input")
    │  confidence hard-capped at low
    ▼
compare_fields (unified_value only)
    ├── consistent   → log_reply only
    └── inconsistent → ValidationIssue added, status → pending
```

---

### 5. DocReader

**File:** `doc_reader.py`

**Responsibility:** Reads a single file and returns extracted field values. Accepts `target_fields` so the VLM prompt is focused. Handles all file-type-specific preprocessing internally. Stateless — each call is fully independent.

**Retry logic** is centralised in `BaseDocReader.call_vlm()`. All subclasses call this method rather than the VLM API directly. This ensures retry behaviour is consistent and not duplicated across `PDFReader`, `ImageReader`, and `TextReader`.

**Known limitation — scanned PDFs:** Routing on file extension alone means a PDF with no text layer will be sent to `PDFReader` and may produce poor extractions. `PDFReader` mitigates this by checking extracted text volume after initial parsing. If the result falls below `pdf_text_threshold` (defined in `config/settings.yaml`), it falls back to `ImageReader` and logs the fallback in `DocRecord.status_reason`.

#### ADT

```python
class BaseDocReader:
    def read(
        self,
        file_path: str,
        target_fields: list[FieldSchema],
        client: BaseLLMClient       # injected from ClaimAgent.llm_clients
    ) -> list[ExtractedField]

    def call_vlm(
        self,
        prompt: str,
        files: list[str] | None,
        client: BaseLLMClient
    ) -> dict
    # centralised VLM call with exponential backoff (3 attempts)
    # raises ParseFailedError on final failure

class PDFReader(BaseDocReader):
    def read(self, file_path, target_fields, client: BaseLLMClient) -> list[ExtractedField]
    # checks extracted text volume against config pdf_text_threshold
    # falls back to ImageReader if below threshold

class ImageReader(BaseDocReader):
    def preprocess(self, file_path: str) -> str     # deskew, returns processed path
    def read(self, file_path, target_fields, client: BaseLLMClient) -> list[ExtractedField]

class TextReader(BaseDocReader):
    # wraps content in XML isolation tags before sending to VLM
    def read(self, file_path, target_fields, client: BaseLLMClient) -> list[ExtractedField]

class DocReaderFactory:
    @staticmethod
    def get_reader(file_path: str) -> BaseDocReader
    # .pdf  → PDFReader (with ImageReader fallback)
    # .png / .jpg → ImageReader
    # .txt  → TextReader
```

---

### 6. LLM Adapters

**File:** `llm_adapters.py`

**Responsibility:** Abstracts all VLM API calls behind a common interface.
Each adapter handles SDK initialization, prompt formatting, JSON schema
enforcement, and response parsing for its specific provider. No other
layer calls a VLM SDK directly.

#### ADT

```python
class BaseLLMClient:
    def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        files: list[str] | None = None
    ) -> dict
    # raises ParseFailedError on malformed response after retries

class GeminiAdapter(BaseLLMClient):
    # enforces structured output via response_mime_type + response_schema
    # handles Pydantic → Gemini-compatible schema conversion
    # note: not all Pydantic field types are supported; adapter holds escape hatch
    def generate(self, prompt, response_schema, files=None) -> dict

class QwenAdapter(BaseLLMClient):
    # enforces structured output via OpenAI-compatible response_format
    # handles Pydantic → JSON schema conversion
    def generate(self, prompt, response_schema, files=None) -> dict

class LLMClientFactory:
    @staticmethod
    def get_client(model_id: str) -> BaseLLMClient
    # "gemini" → GeminiAdapter
    # "qwen"   → QwenAdapter
    # unsupported model_id → raises ValueError
```

---

## Data Structures

### FieldSchema
```python
class FieldSchema(BaseModel):
    field_name: str
    data_type: str
    field_role: Literal["required", "optional"]
    validation_rule: str
    unify_instruction: str
    description: str
```

### ExtractedField
```python
class ExtractedField(BaseModel):
    field_name: str
    field_role: Literal["required", "optional", "discovered"]
    source_trust: Literal["document", "user_input"]
    origin_value: str | None        # raw value from VLM, kept for audit
    unified_value: str | None       # normalised value, used for validation
    data_type: str
    valid: bool
    validation_note: str | None     # reason if valid == False
    confidence: Literal["high", "medium", "low"]
    confidence_note: str | None     # reason if confidence != high
```

### DocRecord
```python
class DocRecord(BaseModel):
    file_name: str
    doc_type: Literal["police_report", "finance_agreement",
                      "settlement_breakdown", "customer_reply", "unknown"]
    doc_role: Literal["required", "optional", "other"]
    source_trust: Literal["document", "user_input"]
    parse_status: Literal["complete", "parse_failed", "unprocessed"]
    # "unprocessed" = not yet parsed; distinct from Claim.status "pending"
    doc_status: Literal["present", "missing", "duplicate"]
    duplicate_type: Literal["same_filename", "same_content"] | None
    # same_filename → incomplete; same_content → pending (data integrity concern)
    status_reason: str | None
    content_hash: str | None        # SHA-256, used for duplicate detection
    raw_text: str | None            # cached to avoid re-calling VLM
    fields: list[ExtractedField]
```

### ValidationIssue
```python
class ValidationIssue(BaseModel):
    issue_type: Literal["inconsistency", "missing", "invalid", "low_confidence"]
    field_name: str | None
    description: str
    sources: list[str]
    values: dict[str, str]          # {file_name: unified_value}
    resolved: bool
    resolved_by: Literal["upload", "human_verified"] | None
    resolved_at: str | None
```

### ConversationRound
```python
class ConversationRound(BaseModel):
    round: int
    timestamp: str
    direction: Literal["outbound", "inbound"]
    message: str
    triggered_by: str | None
    compare_results: dict[str, Literal["consistent", "inconsistent"]] | None
    # inbound only — keyed by field_name
```

### PriorityRecord
```python
class PriorityRecord(BaseModel):
    claim_id: str
    status: Literal["complete", "incomplete", "pending"]
    uploaded_at: str
    express: bool                   # True if incomplete and unresolved issues < 2
    priority_rank: int
    reason: str                     # populated from config/messages.yaml priority_reason
```

---

## Required Documents

| Doc Type | Role |
|----------|------|
| `police_report` | required |
| `finance_agreement` | required |
| `settlement_breakdown` | required |
| `customer_reply` | optional |

A claim cannot reach `complete` status if any required document is missing or has `parse_status != "complete"`.

---

## Error Handling Strategy

All unrecoverable errors result in `pending` status rather than silent failure. Every error is recorded in `DocRecord.status_reason` or `ValidationIssue.description` for human review.

Retry logic lives in `BaseDocReader.call_vlm()` and is not duplicated in subclasses.

| Error | Handling |
|-------|---------|
| VLM returns malformed JSON | Retry once via `call_vlm()`. If still malformed → `parse_status = "parse_failed"`, status → `pending` |
| File corrupted or unreadable | `parse_status = "parse_failed"`, status → `pending`, log in `status_reason` |
| API timeout | Exponential backoff, 3 attempts via `call_vlm()`. On final failure → `parse_status = "parse_failed"`, status → `pending` |
| PDF with no text layer | `PDFReader` detects low text volume (below `pdf_text_threshold` in `config/settings.yaml`) and falls back to `ImageReader` |
| Unknown doc type | `doc_type = "unknown"`, status → `pending`, no field extraction attempted |
| `model_id` not supported | `ValueError` raised at `LLMClientFactory.get_client()`, before any processing begins |

---

## Data Flow

```
Folder path
    │
    ▼
ClaimAgent
    │  loads workflow.yaml + messages.yaml
    │  instantiates llm_client from model_id (config/settings.yaml)
    │  calls chatbot.ask() / chatbot.display() as needed
    │
    ├── DocReaderFactory → PDFReader / ImageReader / TextReader
    │       └── BaseDocReader.call_vlm(client)   ← ClaimAgent.llm_client
    │               └── list[ExtractedField] → stored in DocRecord.fields
    │
    ├── ClaimParser.cross_validate
    │       └── list[ValidationIssue] → stored in Claim.validation_issues
    │
    ├── ClaimParser.determine_status
    │       └── complete / incomplete / pending → stored in Claim.status
    │
    ├── [if incomplete or pending]
    │       generate_customer_message (via messages.yaml template)
    │       → chatbot.display()
    │
    ├── [if customer replies]
    │       ClaimAgent.accept_reply
    │           → chatbot.ask() → reply_text
    │           → ClaimParser.handle_reply(claim, reply_text, client)   ← ClaimAgent.llm_client
    │               ├── parse_reply (TextReader, source_trust = "user_input")
    │               ├── compare_fields
    │               ├── log_reply → conversation_log
    │               └── determine_status → updated Claim.status
    │
    └── [after all claims processed]
            ClaimAgent.prioritize_claims
                └── list[PriorityRecord]
```

---

## Cache and File Structure

```
claims/
└── CLM-001/
    ├── police_report.pdf         ← original input, never modified
    ├── finance_agreement.png     ← original input, never modified
    ├── settlement_breakdown.pdf  ← original input, never modified
    ├── customer_reply.txt        ← original input, never modified (optional)
    └── .cache/
        └── claim_state.json      ← single source of truth; includes
                                     conversation_log, doc_table, all fields
```

`conversation_log` is serialised as part of `claim_state.json`. There is no separate `conversation.json`. A human-readable export can be generated from `claim_state.json` on demand if needed for review, but `claim_state.json` is the only authoritative record.

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/field_schema.json` | Field definitions, validation rules, unify instructions |
| `config/workflow.yaml` | Agent workflow rules and trigger conditions |
| `config/messages.yaml` | Outbound message templates and priority reason strings |
| `config/settings.yaml` | Runtime parameters including `pdf_text_threshold` and `model_id` (VLM provider selection) |

---

## Key Design Decisions

**ClaimAgent controls Chatbot.** `ClaimAgent` owns the conversation flow and calls `chatbot.ask()` / `chatbot.display()` when needed. `Chatbot` is a passive IO tool with no decision-making responsibility.

**Conditional dispatch, not hardcoded pipeline.** LangGraph conditional edges decide which tool to invoke based on `ClaimState`. Deterministic routing is justified in this domain — insurance claim processing has well-defined states and compliance requirements that make unpredictable LLM-driven tool selection inappropriate.

**`pending` over `needs_review`.** `pending` describes ownership (company staff must act), not just state. This maps directly to workflow routing.

**Duplicate routing is type-aware.** Same-filename duplicates route to `incomplete` (trivially resolvable by the user). Same-content duplicates with different filenames route to `pending` (potential data integrity issue requiring human review).

**Express flag is scoped to incomplete claims only.** `complete` claims need no special routing. `pending` claims cannot bypass human review. Express is only meaningful for `incomplete` claims with fewer than 2 unresolved issues.

**Confidence is immutable after extraction.** Assigned by `DocReader` once and never modified. Every value is traceable to its source document and extraction method.

**Customer replies cannot resolve inconsistencies.** This is a deliberate trust model decision, not an oversight. Even if a customer's reply contains a value consistent with one document, the underlying inconsistency between original documents still exists and requires human judgment. Accepting a reply as resolution would silently dismiss a potentially significant data conflict.

**Human input is never trusted.** Customer replies are parsed with `source_trust == "user_input"` and a hard confidence cap of `low`. Information from replies is recorded for audit but never used to resolve issues automatically.

**Workflow and messages are externalised.** `config/workflow.yaml` and `config/messages.yaml` control agent behaviour without requiring code changes. Priority reason strings follow the same pattern for consistency.

**Retry logic is centralised.** All VLM calls go through `BaseDocReader.call_vlm()`. Retry behaviour (exponential backoff, 3 attempts) is implemented once and inherited by all subclasses.

**LLM adapter layer isolates SDK dependencies.** No layer outside `llm_adapters.py` calls a VLM SDK directly. `BaseLLMClient` is the only interface the rest of the system depends on. Swapping providers or adding a new model requires only a new adapter class and updating `model_id` in `config/settings.yaml` — no changes to `DocReader`, `ClaimParser`, or `ClaimAgent`.

**`ParsedDocument` removed.** Fields and raw text are stored directly in `DocRecord`, eliminating a redundant intermediate structure.

**Single source of truth for claim state.** `claim_state.json` is the only persisted record. `conversation_log` is part of `Claim` and serialised within it. No separate `conversation.json` to avoid data drift between two files representing the same information.

**`DocRecord.parse_status` uses `unprocessed`, not `pending`.** At the doc level, `unprocessed` means not yet parsed. This is semantically distinct from `Claim.status = "pending"`, which means company staff must act. Using the same word for different concepts across levels would cause confusion.

---

## Scale-up Considerations

**Hybrid model routing:** The single `llm_client` on `ClaimAgent` can be extended to a role-keyed map (`dict[str, BaseLLMClient]`) when multi-model routing is needed — for example, routing vision-heavy tasks to Gemini and logic tasks to a different model. The `BaseLLMClient` interface already supports this; it requires only a `task_model_map` config entry and a small change to `ClaimAgent.__init__`. No other layer needs to change.

**Parallel document processing:** `DocReader` is stateless. Multiple files can be processed concurrently with `asyncio` or a thread pool.

**Confidence-based model fallback:** Low-confidence extractions could retry with a larger model before escalating to human review. Straightforward to add inside `DocReader` by passing a different `BaseLLMClient` on retry — no changes required outside that layer.

**Persistent state:** `claim_state.json` is structured to load into PostgreSQL or Redis without changes.

**API layer:** `Chatbot` can be replaced with a FastAPI endpoint. `Claim` serialises to JSON via Pydantic.

**Multi-claim parallelism:** A queue-based dispatcher could run multiple `ClaimAgent` instances concurrently with minimal architectural changes.

**Non-blocking reply handling:** Because `ClaimState` is fully serialisable, the system can be extended to suspend after sending a customer message and resume on an inbound webhook or queue event. The LangGraph checkpoint mechanism supports this without changes to `ClaimParser` or `Claim`.
