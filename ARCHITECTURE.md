# Architecture

## Overview

This system is a five-layer AI agent for processing insurance claims. Each layer has a single, clearly defined responsibility and communicates only with the layer directly below it.

```
Chatbot        (core/chatbot.py)      — IO only, no business logic
  └── ClaimAgent  (core/agent.py)     — LangGraph state machine controller
        └── ClaimParser (core/parser.py) — validation, status, reply handling
              └── DocReader (core/doc_reader.py) — file extraction via VLM
                    └── LLMAdapters (core/llm_adapters.py) — Gemini / Qwen clients
```

Shared utilities (YAML loading, schema loading) live in `core/utils.py`. Named tool functions (`validate_vin`, `check_field_consistency`, `classify_document`) live in `core/tools.py` and are called conditionally by `ClaimAgent` and `ClaimParser`.

**Entry point assumption:** Input files are assumed to have been uploaded to a local claim folder before processing begins.

```python
chatbot = Chatbot()
agent = ClaimAgent(chatbot=chatbot)
agent.process_claim("claims/CLM-001/")
```

---

## Layer Definitions

### 1. Chatbot

**File:** `core/chatbot.py`

**Responsibility:** Pure IO layer. Renders output and captures user input on demand. Contains no business logic and no decision-making. `ClaimAgent` owns the conversation flow and calls `Chatbot` when it needs to interact with the user.

**Design note:** Implemented as CLI for the current version. Swapping to Gradio or a REST API requires only replacing this class.

#### ADT

```python
class Chatbot:
    def ask(self, prompt: str) -> str          # called by ClaimAgent to get user input
    def display(self, message: str) -> None    # called by ClaimAgent to show output
```

---

### 2. ClaimAgent

**File:** `core/agent.py`

**Responsibility:** Controls the entire workflow via a LangGraph state machine. Decides which node to invoke next based on claim state. Reads workflow rules from `config/workflow.yaml` and message templates from `config/messages.yaml`. Does not read documents or validate fields directly.

**Framework:** LangGraph (`StateGraph` with conditional edges)

**LangGraph graph** (`ClaimAgent._build_graph`):
- Entry → `parse_documents` → `cross_validate`
- Conditional after `cross_validate`: if status is `incomplete` or `needs_review` → `generate_message` → `accept_reply`
- Conditional after `accept_reply`: if reply was not skipped and under `max_reply_rounds` → `cross_validate` again; else END
- Routing is deterministic/conditional, not LLM-driven (compliance requirement — insurance claim processing has well-defined states where unpredictable LLM-driven tool selection would be a liability)

**State:** `ClaimState` is a `TypedDict` holding `claim`, `folder_path`, `field_schemas`, `llm_client`, `chatbot`, `workflow_config`, `message_config`, `parser`, and `reply_skipped`. Note: `llm_client`, `chatbot`, and `parser` are non-serialisable objects; LangGraph checkpointing is not available without moving these to `ClaimAgent` instance variables.

#### ADT

```python
class ClaimAgent:
    chatbot: Chatbot
    llm_client: BaseLLMClient        # single client, model selected from config/settings.yaml
    workflow_config: dict            # loaded from config/workflow.yaml
    message_config: dict             # loaded from config/messages.yaml
    field_schemas: list[FieldSchema] # loaded from config/field_schema.json

    def __init__(self, chatbot: Chatbot) -> None
    def process_claim(self, folder_path: str, uploaded_at: str | None = None) -> Claim
    # uploaded_at: ISO 8601 string; falls back to folder mtime if None
    def prioritize_claims(self, claims: list[Claim]) -> list[PriorityRecord]
```

**Graph nodes:**

| Node | Trigger condition |
|------|------------------|
| `parse_documents` | entry point |
| `cross_validate` | after all documents parsed; also after each accepted reply |
| `generate_message` | status is `incomplete` or `needs_review` |
| `accept_reply` | after every outbound customer message |

**Customer message format:** `ClaimParser.build_customer_message` assembles a single unified email from body fragments defined in `config/messages.yaml`. One greeting, numbered issue list, one sign-off — regardless of how many issues are present. Warning-severity inconsistencies get a non-alarming fragment; blocking ones ask the customer to clarify.

**Claim prioritisation logic:**

```
Primary sort: status
  complete             → highest priority (ready to finalise immediately)
  needs_review         → second (staff must act — human review queue)
  incomplete (express) → third (≤1 unresolved issue, no missing docs — nearly resolved)
  incomplete           → lowest (waiting on customer; staff cannot act yet)

Secondary sort: uploaded_at (oldest first within same status group)
```

Priority reason strings are defined in `config/messages.yaml` under `priority_reason` — not generated ad hoc.

---

### 3. ClaimParser

**File:** `core/parser.py`

**Responsibility:** All processing logic that operates on a `Claim`. Each method has a single responsibility. `handle_reply` is the orchestrator for reply processing — it calls the smaller methods rather than implementing everything inline.

#### ADT

```python
class ClaimParser:

    def check_required_docs(
        self, claim: Claim
    ) -> list[str]
    # returns list of required doc types that are absent

    def cross_validate(
        self, claim: Claim
    ) -> list[ValidationIssue]
    # compares unified_value of each field across all document sources
    # only processes source_trust == "document"
    # never modifies confidence values
    # assigns severity: "blocking" or "warning" (see below)

    def determine_status(
        self, claim: Claim
    ) -> Literal["complete", "incomplete", "needs_review"]
    # evaluates doc_table and validation_issues
    # only "blocking" inconsistencies affect routing

    def extract_reply_fields(
        self, reply_text: str,
        schemas: list[FieldSchema],
        client: BaseLLMClient
    ) -> list[ExtractedField]
    # calls TextReader; source_trust hard-set to "user_input", confidence hard-capped at low

    def compare_fields(
        self, claim: Claim,
        new_fields: list[ExtractedField]
    ) -> dict[str, Literal["consistent", "inconsistent"]]
    # compares new_fields against claim.extracted_fields (unified_value only)

    def record_reply(
        self, claim: Claim,
        message: str,
        compare_results: dict
    ) -> None
    # appends ConversationRound to claim.conversation_log; increments reply_count

    def handle_reply(
        self, claim: Claim,
        reply_text: str,
        client: BaseLLMClient,
        schemas: list[FieldSchema]
    ) -> Claim
    # orchestrator: extract_reply_fields → compare_fields → record_reply → determine_status

    def build_customer_message(
        self, claim: Claim,
        message_config: dict
    ) -> str
    # assembles a single unified outbound email from issue_fragments in messages.yaml
```

**Confidence-aware inconsistency severity:**

`cross_validate` assigns `severity` to each `ValidationIssue`:

| Condition | Severity |
|-----------|---------|
| Two or more medium/high confidence sources disagree | `blocking` |
| Only low-confidence sources conflict, or a single low-confidence source disagrees with a high-confidence one | `warning` |

`blocking` inconsistencies affect claim routing. `warning` inconsistencies are logged for staff review but do not push the claim to `incomplete` or `needs_review`.

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
    ├── consistent   → record_reply only
    └── inconsistent → ValidationIssue added, status → needs_review
```

Customer replies **cannot resolve inconsistencies** — even a matching reply does not eliminate a conflict between original documents. This is a deliberate trust model decision.

---

### 4. DocReader

**File:** `core/doc_reader.py`

**Responsibility:** Reads a single file and returns a `DocRecord` containing extracted field values. Accepts `schemas` so the VLM prompt is focused on the fields that matter. Handles all file-type-specific preprocessing internally. Stateless — each call is fully independent.

**Dispatch:** `get_doc_reader(file_path)` selects the appropriate reader by file extension.

**PDF fallback:** `PDFReader` checks extracted text volume after initial parsing. If the result falls below `pdf_text_threshold` (in `config/settings.yaml`), it falls back to `ImageReader` and logs the fallback in `DocRecord.status_reason`.

#### ADT

```python
class BaseDocReader(ABC):
    def read(
        self,
        file_path: str,
        schemas: list[FieldSchema],
        client: BaseLLMClient
    ) -> DocRecord

    def call_vlm(
        self,
        prompt: str,
        files: list[str] | None,
        client: BaseLLMClient
    ) -> dict
    # delegates to client.generate(); raises ParseFailedError on failure

class PDFReader(BaseDocReader): ...
    # extracts text via pdfplumber; falls back to ImageReader if text is sparse

class ImageReader(BaseDocReader): ...
    # preprocesses (deskew) before sending to VLM
    # caps all extracted field confidence at "medium"

class TextReader(BaseDocReader): ...
    # wraps content in XML isolation tags before sending to VLM

def get_doc_reader(file_path: str) -> BaseDocReader:
    # .pdf        → PDFReader (with ImageReader fallback)
    # .png / .jpg / .jpeg → ImageReader
    # .txt        → TextReader
    # other       → raises UnsupportedFileTypeError
```

---

### 5. LLM Adapters

**File:** `core/llm_adapters.py`

**Responsibility:** Abstracts all VLM API calls behind a common interface. Each adapter handles SDK initialisation, prompt formatting, JSON schema enforcement, and response parsing for its specific provider. No other layer calls a VLM SDK directly.

**Retry logic:** `_with_retry` wraps every `generate` call in `GeminiAdapter` and `QwenAdapter`. It catches transient errors (rate-limit 429, server errors 5xx, `ResourceExhausted`, `ServiceUnavailable`) and retries with exponential backoff using parameters from `config/settings.yaml`. After exhausting all attempts, it raises `ParseFailedError`. `QwenLocalAdapter` (local, no network) is excluded from retry.

#### ADT

```python
class BaseLLMClient(ABC):
    def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        files: list[str] | None = None
    ) -> dict
    # raises ParseFailedError on malformed response or after retries exhausted

class GeminiAdapter(BaseLLMClient):
    # enforces structured output via response_mime_type + response_schema
    # flattens Pydantic schema to remove $ref / unsupported keywords before calling Gemini
    # retries on transient API errors via _with_retry

class QwenAdapter(BaseLLMClient):
    # enforces structured output via OpenAI-compatible response_format
    # retries on transient API errors via _with_retry

class QwenLocalAdapter(BaseLLMClient):
    # runs Qwen VL models locally via Hugging Face transformers
    # no retry (local inference, no network)

class LLMClientFactory:
    @staticmethod
    def get_client(model_id: str) -> BaseLLMClient:
    # "gemini"      → GeminiAdapter
    # "qwen"        → QwenAdapter
    # "qwen_local"  → QwenLocalAdapter
    # unsupported   → raises UnsupportedModelError
```

---

### 6. Tools

**File:** `core/tools.py`

**Responsibility:** Named, callable tool functions invoked conditionally during processing. Each call is logged to `Claim.tools_used` so the audit trail shows which tools ran and why. No tool modifies claim state directly — callers decide what to do with the result.

```python
def classify_document(file_name: str) -> dict
# infers doc type from filename keywords
# called by node_parse_documents for every file before reading

def validate_vin(vin: str) -> dict
# checks VIN length and character set
# called by ClaimParser.cross_validate when a VIN field is present

def check_field_consistency(field_name: str, values: dict[str, str]) -> dict
# compares values across sources; returns consistent/inconsistent + differing values
# called by ClaimParser.cross_validate for each multi-source field
```

Each function returns `{"tool": "<name>", "input": {...}, "result": {...}}` — the same dict appended to `Claim.tools_used`.

---

## Data Structures

### Claim
```python
class Claim(BaseModel):
    claim_id: str
    uploaded_at: str                              # ISO 8601
    status: Literal["complete", "incomplete", "needs_review"]
    doc_table: list[DocRecord]
    extracted_fields: dict[str, ExtractedField]  # highest-confidence wins per field
    validation_issues: list[ValidationIssue]
    conversation_log: list[ConversationRound]
    reply_count: int
    tools_used: list[dict]
    # each entry: {"tool": "<name>", "input": {...}, "result": {...}}
```

---

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
    # "unprocessed" = not yet parsed; distinct from Claim.status "needs_review"
    doc_status: Literal["present", "missing", "duplicate"]
    duplicate_type: Literal["same_filename", "same_content", "multiple_versions"] | None
    # same_filename → incomplete (customer-fixable)
    # same_content / multiple_versions → needs_review (human judgment needed)
    status_reason: str | None
    content_hash: str | None        # SHA-256, used for duplicate detection
    raw_text: str | None
    fields: list[ExtractedField]
```

### ValidationIssue
```python
class ValidationIssue(BaseModel):
    issue_type: Literal["inconsistency", "missing", "invalid", "low_confidence"]
    severity: Literal["blocking", "warning"]  # see cross_validate severity rules
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
    status: Literal["complete", "incomplete", "needs_review"]
    uploaded_at: str
    express: bool                   # True if incomplete and unresolved issues < 2 and no missing docs
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

## Status Aggregation Logic

`ClaimParser.determine_status` evaluates in this priority order:

```
# incomplete — customer-fixable only
1. any doc_status == "missing"                                → incomplete
2. any duplicate_type == "same_filename"                      → incomplete

# needs_review — all docs present but human judgment needed
3. any duplicate_type == "same_content" or "multiple_versions" → needs_review
4. any unresolved blocking inconsistency                      → needs_review
   (warning-severity inconsistencies do not affect routing)
5. any parse_status == "parse_failed"                         → needs_review
6. any doc_type == "unknown"                                  → needs_review
7. any unresolved low_confidence required field               → needs_review

# complete
8. all required docs present + all required fields valid      → complete
9. fallback                                                   → incomplete
```

**Status semantics:**

| Status | Meaning | Who acts |
|--------|---------|---------|
| `complete` | All required docs present, all fields valid, no unresolved blocking issues | Nobody — ready to finalise |
| `incomplete` | Customer-fixable problem: missing doc or same-filename duplicate | Customer |
| `needs_review` | All docs present but human judgment required (inconsistency, parse failure, unknown doc type, low confidence) | Company staff |

**Two-criterion split:** `incomplete` is reserved exclusively for cases the customer can fix themselves. Everything else that blocks completion goes to `needs_review`. This removes the `reply_count` branch from `determine_status` — whether a reply has been received does not change who needs to act.

---

## Error Handling

All unrecoverable errors result in `needs_review` status rather than silent failure. Every error is recorded in `DocRecord.status_reason` or `ValidationIssue.description` for human review.

| Error | Handling |
|-------|---------|
| Transient API error (rate-limit, 5xx) | Exponential backoff, up to `retry.max_attempts` attempts (`llm_adapters._with_retry`). On final failure → `ParseFailedError` → `parse_status = "parse_failed"`, claim → `needs_review` |
| VLM returns malformed JSON | `ParseFailedError` raised immediately (not retried) → `parse_status = "parse_failed"`, claim → `needs_review` |
| File corrupted or unreadable | `parse_status = "parse_failed"`, claim → `needs_review`, logged in `status_reason` |
| PDF with no text layer | `PDFReader` detects low text volume (below `pdf_text_threshold`) and falls back to `ImageReader` |
| Unknown doc type | `doc_type = "unknown"`, claim → `needs_review`, no field extraction attempted |
| Unsupported `model_id` | `UnsupportedModelError` raised at `LLMClientFactory.get_client()`, before any processing begins |

---

## Data Flow

```
Folder path
    │
    ▼
ClaimAgent
    │  loads workflow.yaml + messages.yaml + field_schema.json
    │  instantiates llm_client from model_id (config/settings.yaml)
    │  calls chatbot.ask() / chatbot.display() as needed
    │
    ├── node_parse_documents
    │       for each file:
    │           classify_document(file_name) → Claim.tools_used
    │           get_doc_reader() → PDFReader / ImageReader / TextReader
    │               └── BaseDocReader.read(schemas, llm_client)
    │                       └── DocRecord (with fields) → stored in Claim.doc_table
    │       merge extracted_fields into Claim.extracted_fields
    │       post-pass: flag multiple_versions duplicates
    │       insert missing-doc placeholder DocRecords
    │
    ├── node_cross_validate
    │       ClaimParser.cross_validate → list[ValidationIssue] (blocking or warning)
    │           check_field_consistency / validate_vin → Claim.tools_used
    │       ClaimParser.determine_status → complete / incomplete / needs_review
    │
    ├── [if incomplete or needs_review]
    │       node_generate_message
    │           ClaimParser.build_customer_message → single unified email (from messages.yaml fragments)
    │           → chatbot.display() + ConversationRound appended
    │
    ├── [waiting for customer reply]
    │       node_accept_reply
    │           chatbot.ask() → reply_text
    │           ClaimParser.handle_reply(claim, reply_text, llm_client, schemas)
    │               ├── extract_reply_fields (TextReader, source_trust = "user_input")
    │               ├── compare_fields
    │               ├── record_reply → conversation_log
    │               └── determine_status → updated Claim.status
    │           → loop back to node_cross_validate if under max_reply_rounds
    │
    └── [after processing]
            ClaimAgent.prioritize_claims → list[PriorityRecord]
```

---

## Cache and File Structure

```
claims/
└── CLM-001/
    ├── police_report.pdf          ← original input, never modified
    ├── finance_agreement.png      ← original input, never modified
    ├── settlement_breakdown.pdf   ← original input, never modified
    ├── customer_reply.txt         ← original input, never modified (optional)
    └── .cache/
        └── claim_state.json       ← full serialised Claim; single source of truth
```

`conversation_log` is serialised as part of `claim_state.json`. There is no separate conversation file — a single file eliminates the risk of drift between two representations of the same state.

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/field_schema.json` | Field definitions, validation rules, unify instructions |
| `config/workflow.yaml` | `max_reply_rounds` and other routing parameters |
| `config/messages.yaml` | `customer_email` wrapper template, `issue_fragments` body snippets, `priority_reason` strings |
| `config/settings.yaml` | `model_id`, model parameters, `pdf_text_threshold`, `retry` block |

---

## Key Design Decisions

**Conditional dispatch, not hardcoded pipeline.** LangGraph conditional edges decide which node to invoke based on `ClaimState`. Deterministic routing is justified in this domain — insurance claim processing has well-defined states and compliance requirements that make unpredictable LLM-driven tool selection inappropriate.

**Confidence is immutable after extraction.** Assigned by `DocReader` once and never modified. Every value is traceable to its source document and extraction method.

**Confidence-aware inconsistency routing.** Cross-validation assigns `blocking` severity only when two or more medium/high confidence sources disagree. Low-confidence conflicts are logged as `warning` and do not hold up the claim — the company is notified but processing continues.

**Customer replies cannot resolve inconsistencies.** Even if a customer's reply matches one document, the underlying conflict between original documents still exists and requires human judgment. Accepting a reply as resolution would silently dismiss a potentially significant data conflict.

**Human input is never trusted.** Customer replies are parsed with `source_trust = "user_input"` and a hard confidence cap of `low`. Reply data is recorded for audit but never used to auto-resolve issues.

**Single unified customer email.** All issues (missing docs, parse failures, inconsistencies) are collected and sent as one email per interaction cycle. Individual sections are body fragments in `config/messages.yaml`; the outer wrapper (greeting + sign-off) is a single template. This avoids sending multiple disjointed emails for the same claim in the same round.

**Status enum matches the spec.** `complete | incomplete | needs_review` map directly to who acts next: nobody, the customer, and company staff respectively.

**Duplicate routing is type-aware.** Same-filename duplicates → `incomplete` (trivially resolvable by the user). Same-content duplicates with different filenames, or multiple documents of the same required type → `needs_review` (potential data integrity concern requiring human judgment).

**Express flag is scoped to incomplete claims only.** `complete` claims need no special routing. `needs_review` claims cannot bypass human review. Express is only meaningful for `incomplete` claims with fewer than 2 unresolved issues and no missing documents.

**Retry is in the adapter layer, not DocReader.** Transient API failures (rate-limits, 5xx) are retried by `llm_adapters._with_retry` before they propagate. `DocReader` sees either a successful result or a `ParseFailedError` — it does not need to know about network-level retry logic.

**LLM adapter layer isolates SDK dependencies.** No layer outside `llm_adapters.py` calls a VLM SDK directly. Swapping providers or adding a new model requires only a new adapter class and updating `model_id` in `config/settings.yaml`.

**Single source of truth for claim state.** `claim_state.json` is the only persisted record. `conversation_log` is part of `Claim` and serialised within it.

**`DocRecord.parse_status` uses `unprocessed`, not `needs_review`.** At the doc level, `unprocessed` means not yet parsed — semantically distinct from `Claim.status = "needs_review"`, which means company staff must act.

---

## Scale-up Considerations

**Parallel document processing:** `DocReader` is stateless. Multiple files in the same claim can be processed concurrently with `asyncio` or a thread pool.

**Confidence-based model fallback:** Low-confidence extractions could retry with a larger model before escalating to human review. Straightforward to add inside `DocReader` by passing a different `BaseLLMClient` on retry — no changes required outside that layer.

**Multi-model routing:** The single `llm_client` on `ClaimAgent` can be extended to a role-keyed map (`dict[str, BaseLLMClient]`) when different tasks warrant different models (e.g., a vision model for PDFs and a text model for reply parsing). The `BaseLLMClient` interface already supports this; it requires a `task_model_map` config entry and a small change to `ClaimAgent.__init__`.

**Persistent state:** `claim_state.json` is structured to load directly into PostgreSQL or Redis without changes to the data model.

**API layer:** `Chatbot` can be replaced with a FastAPI endpoint. `Claim` already serialises to JSON via Pydantic.

**Non-blocking reply handling:** To support inbound webhook or queue resumption, `ClaimState` would need refactoring — `llm_client`, `chatbot`, and `parser` are not serialisable and must be moved to `ClaimAgent` instance variables before LangGraph checkpointing becomes viable.
