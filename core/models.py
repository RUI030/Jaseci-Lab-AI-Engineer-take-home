from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


# --- Schema Models ---

class FieldSchema(BaseModel):
    field_name: str
    data_type: str
    field_role: Literal["required", "optional"]
    validation_rule: str
    validation_pattern: str | None = None  # regex applied after LLM extraction; None = no structural check
    unify_instruction: str
    description: str


# --- Document & Field Models ---

class ExtractedField(BaseModel):
    field_name: str
    field_role: Literal["required", "optional", "discovered"]
    source_trust: Literal["document", "user_input"]
    origin_value: str | None = None
    unified_value: str | None = None
    data_type: str
    valid: bool
    validation_note: str | None = None
    confidence: Literal["high", "medium", "low"]
    confidence_note: str | None = None


class DocRecord(BaseModel):
    file_name: str
    doc_type: Literal[
        "police_report",
        "finance_agreement",
        "settlement_breakdown",
        "customer_reply",
        "unknown",
    ]
    doc_role: Literal["required", "optional", "other"]
    source_trust: Literal["document", "user_input"]
    parse_status: Literal["complete", "parse_failed", "unprocessed"] = "unprocessed"
    doc_status: Literal["present", "missing", "duplicate"] = "present"
    duplicate_type: Literal["same_filename", "same_content", "multiple_versions"] | None = None
    status_reason: str | None = None
    content_hash: str | None = None
    raw_text: str | None = None
    fields: list[ExtractedField] = Field(default_factory=list)


# --- Validation & Conversation Models ---

class ValidationIssue(BaseModel):
    issue_type: Literal["inconsistency", "missing", "invalid", "low_confidence"]
    severity: Literal["blocking", "warning"] = "blocking"
    field_name: str | None = None
    description: str
    sources: list[str] = Field(default_factory=list)
    values: dict[str, str] = Field(default_factory=dict)
    resubmit_doc: str | None = None  # file_name of lower-confidence doc to request resubmission
    resolved: bool = False
    resolved_by: Literal["upload", "human_verified"] | None = None
    resolved_at: str | None = None


class ConversationRound(BaseModel):
    round: int
    timestamp: str
    direction: Literal["outbound", "inbound"]
    message: str
    triggered_by: str | None = None
    compare_results: dict[str, Literal["consistent", "inconsistent"]] | None = None


# --- Claim & Priority Models ---

class Claim(BaseModel):
    claim_id: str
    status: Literal["complete", "incomplete", "needs_review"] = "needs_review"
    reply_count: int = 0
    uploaded_at: str
    doc_table: list[DocRecord] = Field(default_factory=list)
    extracted_fields: dict[str, ExtractedField] = Field(default_factory=dict)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    conversation_log: list[ConversationRound] = Field(default_factory=list)
    conversation_summary: str = ""
    tools_used: list[dict] = Field(default_factory=list)
    # each entry: {"tool": "<name>", "input": {...}, "result": {...}}


class PriorityRecord(BaseModel):
    claim_id: str
    status: Literal["complete", "incomplete", "needs_review"]
    uploaded_at: str
    express: bool
    priority_rank: int
    reason: str
