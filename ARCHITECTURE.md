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

Shared utilities (YAML loading, schema loading) live in `core/utils.py`. Named tool functions (`validate_vin`, `check_field_consistency`, `classify_document`) live in `core/tools.py`; `validate_vin` and `check_field_consistency` are dispatched at runtime by the byLLM ReAct loop (`run_cross_validation`), while `classify_document` is called directly by `ClaimAgent`. Priority scheduling logic (sorting, express eligibility) lives in `core/scheduler.py` — stateless functions with no dependency on the LLM or graph.

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
- Conditional after `cross_validate`:
  - status is `incomplete` AND `reply_queue` is non-empty → `accept_reply` directly (customer already replied — process before sending anything)
  - all other cases → `generate_message`
- Conditional after `generate_message`:
  - status is `incomplete` → `accept_reply`
  - status is `complete` or `needs_review` → END (message sent, no reply loop)
- Conditional after `accept_reply`: if reply was not skipped and under `max_reply_rounds` → `cross_validate` again; else END
- Routing is deterministic/conditional, not LLM-driven (compliance requirement — insurance claim processing has well-defined states where unpredictable LLM-driven tool selection would be a liability)

**State:** `ClaimState` is a `TypedDict` holding `claim`, `folder_path`, `field_schemas`, `llm_client`, `chatbot`, `workflow_config`, `message_config`, `parser`, `reply_skipped`, and `reply_queue`. `reply_queue` is a `list[str]` pre-loaded with `.txt` file contents during `parse_documents`; `node_accept_reply` drains it before falling back to stdin. Note: `llm_client`, `chatbot`, and `parser` are non-serialisable objects; LangGraph checkpointing is not available without moving these to `ClaimAgent` instance variables.

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
```

Priority scheduling is handled by standalone functions in `core/scheduler.py` (no LLM dependency, callable without a `ClaimAgent` instance):

```python
# core/scheduler.py
def prioritize_claims(claims: list[Claim], message_config: dict) -> list[PriorityRecord]
def _is_express_eligible(claim: Claim) -> bool
def _priority_key(claim: Claim) -> int
```

**Graph nodes:**

| Node | Trigger condition |
|------|------------------|
| `parse_documents` | entry point |
| `cross_validate` | after all documents parsed; also after each accepted reply |
| `generate_message` | all statuses (complete, incomplete, needs_review), UNLESS status is incomplete and reply_queue is non-empty |
| `accept_reply` | after `generate_message` when status is `incomplete`; also directly from `cross_validate` when `reply_queue` is non-empty |

**Customer message format:** `ClaimParser.build_customer_message_llm` composes the outbound email for all three statuses using an LLM guided by structured claim context and writing guidelines from `config/messages.yaml`. Message tone and content vary by status: `complete` congratulates and confirms no action needed; `needs_review` explains the uncertain item(s) in plain language, sets timeline expectations (3–5 days), and offers resubmission to speed up review; `incomplete` presents our best-guess values for confirmation and requests missing documents. If the LLM call fails, `ClaimParser.build_customer_message` (template-based) is used as a fallback.

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
        self, claim: Claim,
        field_schemas: list[FieldSchema] | None = None
    ) -> list[ValidationIssue]
    # compares unified_value of each field across all document sources
    # only processes source_trust == "document"
    # never modifies confidence values
    # assigns severity: "blocking" or "warning" (see below)
    # when field_schemas is provided, dispatches run_cross_validation (byLLM ReAct) to populate tools_used

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

    def build_customer_message_llm(
        self, claim: Claim,
        llm_client: BaseLLMClient,
        message_config: dict
    ) -> str
    # primary path: LLM-composed email guided by claim context + guidelines from messages.yaml
    # context includes conversation_summary, best-guess field values, and document status
    # raises on LLM failure → caller falls back to build_customer_message

    def build_customer_message(
        self, claim: Claim,
        message_config: dict
    ) -> str
    # fallback: assembles a unified email from issue_fragments templates in messages.yaml

    def update_conversation_summary(
        self, claim: Claim,
        llm_client: BaseLLMClient,
        message_config: dict
    ) -> None
    # called after every accepted reply; uses LLM to update claim.conversation_summary
    # summary tracks both sides: customer CONFIRMED/CORRECTED/PENDING items AND what we already
    # told the customer (WE_SAID), so the next message never contradicts a prior outbound message
    # best-effort: silently skips on failure so claim processing is never blocked

    def merge_summary_fields(
        self, claim: Claim,
        llm_client: BaseLLMClient,
        schemas: list[FieldSchema]
    ) -> None
    # called after update_conversation_summary; extracts fields from conversation_summary
    # and merges them into claim.extracted_fields
    # summary fields use source_trust="user_input", confidence="medium" (customer-confirmed answers
    # to direct questions); they fill in fields absent from documents and can override
    # low-confidence document extractions, but never override high-confidence document values

    def resolve_multiple_versions(
        self, claim: Claim
    ) -> None
    # compares fields between authoritative and duplicate (multiple_versions) DocRecords
    # adds blocking ValidationIssue if numeric fields differ by >10% or string fields mismatch
    # adds warning ValidationIssue if values are within tolerance
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

**PDF reading strategy:** `PDFReader` checks `client.supports_native_pdf()` to decide how to send the file. Gemini uploads the PDF directly via the Files API (native support — tables and layout fully visible). `QwenLocalAdapter` renders pages to images internally. For `QwenAdapter` (cloud API, no native PDF), `PDFReader` renders each page to a PNG temp file and passes them all, then cleans up.

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
    # passes PDF directly to VLM if client.supports_native_pdf(); otherwise renders pages to PNGs

class ImageReader(BaseDocReader): ...
    # preprocesses (deskew) before sending to VLM
    # caps all extracted field confidence at "medium"

class TextReader(BaseDocReader): ...
    # wraps content in XML isolation tags before sending to VLM

def get_doc_reader(file_path: str) -> BaseDocReader:
    # .pdf              → PDFReader (native PDF or page-image rendering per adapter)
    # .png / .jpg / .jpeg → ImageReader
    # .txt              → TextReader
    # other             → raises UnsupportedFileTypeError
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
    # structured extraction: enforces JSON schema; raises ParseFailedError on failure

    def generate_text(self, prompt: str) -> str
    # plain-text generation without JSON schema constraints; used for customer messages and summaries
    # temperature fixed at 0.4 for natural language variety

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
def classify_document(file_name: str, actual_type: str | None = None) -> dict
# infers doc type from filename keywords; records actual VLM-confirmed type when provided
# result includes inferred_doc_type, actual_doc_type (if given), and overridden flag
# called by node_parse_documents after reading each file

def validate_vin(vin: str) -> dict
# checks VIN length and character set
# exposed as a tool to the byLLM ReAct loop in run_cross_validation

def check_field_consistency(field_name: str, values_json: str) -> dict
# values_json: JSON string mapping source_document to value,
#   e.g. '{"police_report.pdf": "ABC123", "finance_agreement.pdf": "XYZ789"}'
# compares values across sources; returns consistent/inconsistent + unique_values list
# exposed as a tool to the byLLM ReAct loop in run_cross_validation

@dataclass
class ValidationReport:
    issues_found: list[str]   # human-readable issue descriptions from the LLM
    recommendation: str       # "complete" | "incomplete" | "needs_review"

@by(_tool_llm)  # byLLM ReAct loop — Gemini 2.5 Flash + [validate_vin, check_field_consistency]
def run_cross_validation(fields_by_source: dict, field_schemas: list) -> ValidationReport
# LLM-driven dispatcher: given fields and their per-source values, decides which tools to call.
# Called by ClaimParser.cross_validate when field_schemas is provided (production path).
# Skipped in unit tests (field_schemas=None guard) to avoid real API calls.
# Result logged as a single {"tool": "run_cross_validation", ...} entry in Claim.tools_used.
```

`classify_document` and `run_cross_validation` each append one entry to `Claim.tools_used`:
`{"tool": "<name>", "input": {...}, "result": {...}}`.
`run_cross_validation` records the aggregated LLM recommendation and issues list rather than one entry per field.

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
    conversation_log: list[ConversationRound]    # full raw message history (outbound + inbound)
    conversation_summary: str                    # LLM-maintained running summary; updated after each reply
    reply_count: int
    tools_used: list[dict]
    # each entry: {"tool": "<name>", "input": {...}, "result": {...}}
    next_action: NextAction | None               # set by node_generate_message after each outbound message
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
    source_doc: str | None          # file_name of the winning source document
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
    # same_content → needs_review (human judgment needed)
    # multiple_versions → resolve_multiple_versions compares fields; result may add blocking issue → needs_review
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
3. any duplicate_type == "same_content"                       → needs_review
   (multiple_versions is NOT a direct trigger; resolve_multiple_versions may add a blocking issue → rule 4)
4. any unresolved blocking inconsistency                      → needs_review
   (warning-severity inconsistencies do not affect routing)
5. any parse_status == "parse_failed"                         → needs_review
6. any unresolved low_confidence required field               → needs_review

# complete
7. all required docs present + all required fields valid      → complete
8. fallback                                                   → incomplete
```

**Note on unknown doc type:** `doc_type == "unknown"` on a non-required document does **not** trigger `needs_review` — extra/supplementary docs the VLM can't classify are ignored as long as all required docs are present. Required docs classified as `unknown` are caught upstream by the missing-placeholder mechanism (rule 1).

**Status semantics:**

| Status | Meaning | Who acts |
|--------|---------|---------|
| `complete` | All required docs present, all fields valid, no unresolved blocking issues | Nobody — ready to finalise |
| `incomplete` | Customer-fixable problem: missing doc or same-filename duplicate | Customer |
| `needs_review` | All docs present but human judgment required (inconsistency, parse failure, low confidence) | Company staff |

**Two-criterion split:** `incomplete` is reserved exclusively for cases the customer can fix themselves. Everything else that blocks completion goes to `needs_review`. This removes the `reply_count` branch from `determine_status` — whether a reply has been received does not change who needs to act.

---

## Error Handling

All unrecoverable errors result in `needs_review` status rather than silent failure. Every error is recorded in `DocRecord.status_reason` or `ValidationIssue.description` for human review.

| Error | Handling |
|-------|---------|
| Transient API error (rate-limit, 5xx) | Exponential backoff, up to `retry.max_attempts` attempts (`llm_adapters._with_retry`). On final failure → `ParseFailedError` → `parse_status = "parse_failed"`, claim → `needs_review` |
| VLM returns malformed JSON | `ParseFailedError` raised immediately (not retried) → `parse_status = "parse_failed"`, claim → `needs_review` |
| File corrupted or unreadable | `parse_status = "parse_failed"`, claim → `needs_review`, logged in `status_reason` |
| PDF (any type) | `PDFReader` sends the file visually — native upload for Gemini, page-image rendering for Qwen; tables and layout are always visible to the VLM |
| Unknown doc type (non-required) | `doc_type = "unknown"`, logged in `Claim.tools_used`; does not affect routing if required docs are present |
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
    │       ClaimParser.cross_validate(claim, field_schemas) → list[ValidationIssue] (blocking or warning)
    │           run_cross_validation (byLLM ReAct: validate_vin + check_field_consistency) → Claim.tools_used
    │       ClaimParser.determine_status → complete / incomplete / needs_review
    │
    ├── [all statuses, unless incomplete AND reply_queue non-empty]
    │       node_generate_message
    │           ClaimParser.build_customer_message_llm (LLM-composed, guidelines + claim context)
    │               context: status, conversation_summary, best-guess field values, document status
    │               tone varies by status: complete=congratulate, needs_review=explain+timeline,
    │                                      incomplete=confirm values + request missing docs
    │           fallback on failure: ClaimParser.build_customer_message (template)
    │           → chatbot.display() + ConversationRound(direction="outbound") appended
    │           if complete or needs_review → END
    │
    ├── [if incomplete: after generate_message, or directly from cross_validate if reply_queue non-empty]
    │       node_accept_reply
    │           if reply_queue non-empty:
    │               pop reply_text from queue; chatbot.display("[Auto-loaded]" preview)
    │           else:
    │               chatbot.ask() → reply_text  (returns "" on EOF for non-interactive runs)
    │           ClaimParser.handle_reply(claim, reply_text, llm_client, schemas)
    │               ├── extract_reply_fields (TextReader, source_trust = "user_input", confidence = low)
    │               ├── compare_fields
    │               ├── record_reply → ConversationRound(direction="inbound") appended
    │               └── determine_status → updated Claim.status
    │           ClaimParser.update_conversation_summary(claim, llm_client, message_config)
    │               └── LLM updates claim.conversation_summary — tracks both customer answers
    │                   (CONFIRMED/CORRECTED/PENDING) and what we told the customer (WE_SAID)
    │           ClaimParser.merge_summary_fields(claim, llm_client, schemas)
    │               └── extracts fields from summary → merges into claim.extracted_fields
    │                   at confidence="medium" (fills gaps; overrides low-confidence doc values)
    │           → loop back to node_cross_validate if under max_reply_rounds
    │
    └── [after processing]
            scheduler.prioritize_claims(claims, message_config) → list[PriorityRecord]
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

`conversation_log` and `conversation_summary` are both serialised as part of `claim_state.json`. There is no separate conversation file — a single file eliminates the risk of drift between two representations of the same state.

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/field_schema.json` | Field definitions, validation rules, unify instructions |
| `config/workflow.yaml` | `max_reply_rounds` and other routing parameters |
| `config/messages.yaml` | `customer_message_guideline` (LLM prompt for outbound messages), `conversation_summary_guideline` (LLM prompt for summary updates), `customer_email` + `issue_fragments` (template fallback), `priority_reason` strings |
| `config/settings.yaml` | `model_id`, model parameters, `retry` block |

---

## Key Design Decisions

**Conditional dispatch, not hardcoded pipeline.** LangGraph conditional edges decide which node to invoke based on `ClaimState`. Deterministic routing is justified in this domain — insurance claim processing has well-defined states and compliance requirements that make unpredictable LLM-driven tool selection a liability.

**Cross-validation tool dispatch is LLM-driven via byLLM.** `validate_vin` and `check_field_consistency` are exposed as tools to a `@by(llm)` decorated `run_cross_validation` function (Gemini 2.5 Flash, max 10 ReAct iterations). At runtime, the LLM decides which tools to call and in what order — it is not hardcoded Python logic. The Python `ClaimParser.cross_validate` loop still generates `ValidationIssue` objects (with severity, resubmit_doc, etc.) for deterministic routing compliance; `run_cross_validation` is responsible only for populating `Claim.tools_used` with the LLM-selected audit trail. `classify_document` remains Python-conditional (once per file, no LLM decision needed).

**Confidence is immutable after extraction.** Assigned by `DocReader` once and never modified. Every value is traceable to its source document and extraction method.

**Confidence-aware inconsistency routing.** Cross-validation assigns `blocking` severity only when two or more medium/high confidence sources disagree. Low-confidence conflicts are logged as `warning` and do not hold up the claim — the company is notified but processing continues.

**Customer replies cannot resolve inconsistencies.** Even if a customer's reply matches one document, the underlying conflict between original documents still exists and requires human judgment. Accepting a reply as resolution would silently dismiss a potentially significant data conflict.

**Customer reply confidence is hard-capped at low; summary fields are elevated to medium.** Raw customer replies use `source_trust = "user_input"` and `confidence = "low"` — they are recorded for audit but cannot override any document-extracted value. However, after each reply the LLM distils the conversation into `conversation_summary` and `merge_summary_fields` extracts fields from that summary at `confidence = "medium"`. The elevation is justified because summary fields represent explicit answers to direct questions (CONFIRMED/CORRECTED in the summary), not unsolicited raw text. Medium confidence can override a low-confidence document extraction (e.g., blurry OCR) but never a high-confidence one.

**LLM-composed customer messages with template fallback.** The primary path (`build_customer_message_llm`) passes structured claim context — status, document status, best-guess field values, and the running `conversation_summary` — to the LLM with guidelines in `config/messages.yaml`. Message tone and content are status-driven: `complete` congratulates and confirms no action needed; `needs_review` explains the specific uncertain item(s) in plain language, sets a 3–5 business day timeline expectation, and offers resubmission as an option; `incomplete` presents our best-guess values for confirmation and requests missing documents. A template-based fallback (`build_customer_message`) fires if the LLM call fails.

**Running conversation summary tracks both sides.** `Claim.conversation_summary` is an LLM-maintained rolling summary updated after every customer reply. It tracks customer answers (CONFIRMED/CORRECTED/PENDING) and what we have already told the customer (WE_SAID), so subsequent messages never contradict prior outbound messages or re-ask for information already provided. The raw `conversation_log` is preserved for audit; the summary exists solely to keep the message context window small and consistent as the conversation grows.

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
