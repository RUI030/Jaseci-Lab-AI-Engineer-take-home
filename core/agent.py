from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import yaml
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from core.chatbot import Chatbot
from core.doc_reader import DocReaderFactory
from core.llm_adapters import BaseLLMClient, LLMClientFactory
from core.models import (
    Claim,
    ConversationRound,
    DocRecord,
    ExtractedField,
    FieldSchema,
    PriorityRecord,
)
from core.parser import ClaimParser


def _load_yaml(relative_path: str) -> dict:
    base = Path(__file__).parent.parent
    with open(base / relative_path) as f:
        return yaml.safe_load(f)


def _load_field_schemas() -> list[FieldSchema]:
    base = Path(__file__).parent.parent
    data = json.loads((base / "config" / "field_schema.json").read_text())
    return [FieldSchema(**item) for item in data]


def _sha256_of(path: str) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- LangGraph state ---

class ClaimState(TypedDict):
    claim: Claim
    folder_path: str
    field_schemas: list[FieldSchema]
    llm_client: Any
    chatbot: Any
    workflow_config: dict
    message_config: dict
    parser: Any
    reply_skipped: bool  # True when user pressed Enter without a reply


# --- Graph nodes ---

def node_parse_documents(state: ClaimState) -> ClaimState:
    claim = state["claim"]
    folder = Path(state["folder_path"])
    schemas = state["field_schemas"]
    client: BaseLLMClient = state["llm_client"]
    parser: ClaimParser = state["parser"]

    # Collect all non-hidden files
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )

    seen_hashes: dict[str, str] = {}  # hash -> file_name
    seen_names: dict[str, str] = {}   # name -> file_path

    for file_path in files:
        file_name = file_path.name

        # --- duplicate-by-filename check ---
        if file_name in seen_names:
            record = DocRecord(
                file_name=file_name,
                doc_type="unknown",
                doc_role="other",
                source_trust="document",
                parse_status="unprocessed",
                doc_status="duplicate",
                duplicate_type="same_filename",
                status_reason=f"Duplicate filename; original: {seen_names[file_name]}",
            )
            claim.doc_table.append(record)
            continue
        seen_names[file_name] = str(file_path)

        # --- compute hash for same-content detection ---
        try:
            content_hash = _sha256_of(str(file_path))
        except Exception:
            content_hash = None

        if content_hash and content_hash in seen_hashes:
            record = DocRecord(
                file_name=file_name,
                doc_type="unknown",
                doc_role="other",
                source_trust="document",
                parse_status="unprocessed",
                doc_status="duplicate",
                duplicate_type="same_content",
                content_hash=content_hash,
                status_reason=(
                    f"Content identical to already-processed file: {seen_hashes[content_hash]}"
                ),
            )
            claim.doc_table.append(record)
            continue

        if content_hash:
            seen_hashes[content_hash] = file_name

        # --- read the document ---
        try:
            reader = DocReaderFactory.get_reader(str(file_path))
            record = reader.read(str(file_path), schemas, client)
        except (ValueError, Exception) as exc:
            record = DocRecord(
                file_name=file_name,
                doc_type="unknown",
                doc_role="other",
                source_trust="document",
                parse_status="parse_failed",
                content_hash=content_hash,
                status_reason=str(exc),
            )
        claim.doc_table.append(record)

    # --- merge extracted_fields (highest confidence wins per field) ---
    CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}
    for record in claim.doc_table:
        for field in record.fields:
            existing = claim.extracted_fields.get(field.field_name)
            if existing is None:
                claim.extracted_fields[field.field_name] = field
            elif CONFIDENCE_RANK.get(field.confidence, 0) > CONFIDENCE_RANK.get(
                existing.confidence, 0
            ):
                claim.extracted_fields[field.field_name] = field

    # --- placeholder DocRecords for missing required docs ---
    required_types = {"police_report", "finance_agreement", "settlement_breakdown"}
    present_types = {r.doc_type for r in claim.doc_table if r.doc_status == "present"}
    for doc_type in required_types - present_types:
        claim.doc_table.append(
            DocRecord(
                file_name=f"[missing] {doc_type}",
                doc_type=doc_type,
                doc_role="required",
                source_trust="document",
                parse_status="unprocessed",
                doc_status="missing",
                status_reason="Required document not found in claim folder.",
            )
        )

    return state


def node_cross_validate(state: ClaimState) -> ClaimState:
    claim = state["claim"]
    parser: ClaimParser = state["parser"]
    parser.cross_validate(claim)
    claim.status = parser.determine_status(claim)
    return state


def _build_customer_message(claim: Claim, message_config: dict) -> str:
    templates = message_config.get("templates", {})
    parts: list[str] = []

    missing_docs = [
        r.doc_type
        for r in claim.doc_table
        if r.doc_status == "missing"
    ]
    if missing_docs:
        tmpl = templates.get("missing_document", "Missing documents: {missing_docs}")
        parts.append(
            tmpl.format(
                claim_id=claim.claim_id,
                missing_docs=", ".join(missing_docs),
            )
        )

    parse_failed_docs = [
        r.file_name
        for r in claim.doc_table
        if r.parse_status == "parse_failed"
    ]
    if parse_failed_docs:
        tmpl = templates.get("parse_failed", "Parse failed: {failed_docs}")
        parts.append(
            tmpl.format(
                claim_id=claim.claim_id,
                failed_docs=", ".join(parse_failed_docs),
            )
        )

    unresolved_inconsistencies = [
        vi for vi in claim.validation_issues
        if vi.issue_type == "inconsistency" and not vi.resolved
    ]
    for vi in unresolved_inconsistencies:
        tmpl = templates.get("field_inconsistency", "Inconsistency in {field_name}")
        details = "; ".join(f"{k}: {v}" for k, v in vi.values.items())
        parts.append(
            tmpl.format(
                claim_id=claim.claim_id,
                field_name=vi.field_name or "unknown field",
                inconsistency_details=details,
            )
        )

    if claim.status == "pending" and not parts:
        tmpl = templates.get("pending_review", "Claim {claim_id} is under review.")
        parts.append(tmpl.format(claim_id=claim.claim_id))

    return "\n\n---\n\n".join(parts) if parts else (
        templates.get("complete", "Claim {claim_id} is complete.").format(
            claim_id=claim.claim_id
        )
    )


def node_generate_message(state: ClaimState) -> ClaimState:
    claim = state["claim"]
    chatbot: Chatbot = state["chatbot"]
    message_config: dict = state["message_config"]

    message = _build_customer_message(claim, message_config)
    chatbot.display(f"\n[Message to customer]\n{message}")

    claim.conversation_log.append(
        ConversationRound(
            round=len(claim.conversation_log) + 1,
            timestamp=datetime.now(timezone.utc).isoformat(),
            direction="outbound",
            message=message,
            triggered_by=claim.status,
        )
    )
    return state


def node_accept_reply(state: ClaimState) -> ClaimState:
    claim = state["claim"]
    chatbot: Chatbot = state["chatbot"]
    parser: ClaimParser = state["parser"]
    client: BaseLLMClient = state["llm_client"]

    reply_text = chatbot.ask("\nPlease type the customer's reply (or press Enter to skip): ")
    if reply_text.strip():
        parser.handle_reply(claim, reply_text, client)
        state["reply_skipped"] = False
    else:
        state["reply_skipped"] = True
    return state


def _route_after_validate(state: ClaimState) -> str:
    status = state["claim"].status
    if status in ("incomplete", "pending"):
        return "generate_message"
    return END


def _route_after_reply(state: ClaimState) -> str:
    if state.get("reply_skipped", False):
        return END
    claim = state["claim"]
    workflow = state["workflow_config"]
    max_rounds = workflow.get("routing", {}).get("max_reply_rounds", 3)
    if claim.reply_count < max_rounds:
        return "cross_validate"
    return END


class ClaimAgent:
    def __init__(self, chatbot: Chatbot) -> None:
        self.chatbot = chatbot
        self.workflow_config = _load_yaml("config/workflow.yaml")
        self.message_config = _load_yaml("config/messages.yaml")
        self.field_schemas = _load_field_schemas()

        settings = _load_yaml("config/settings.yaml")
        model_id = settings.get("model_id", "gemini")
        self.llm_client: BaseLLMClient = LLMClientFactory.get_client(model_id)

        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ClaimState)
        graph.add_node("parse_documents", node_parse_documents)
        graph.add_node("cross_validate", node_cross_validate)
        graph.add_node("generate_message", node_generate_message)
        graph.add_node("accept_reply", node_accept_reply)

        graph.set_entry_point("parse_documents")
        graph.add_edge("parse_documents", "cross_validate")
        graph.add_conditional_edges(
            "cross_validate",
            _route_after_validate,
            {"generate_message": "generate_message", END: END},
        )
        graph.add_edge("generate_message", "accept_reply")
        graph.add_conditional_edges(
            "accept_reply",
            _route_after_reply,
            {"cross_validate": "cross_validate", END: END},
        )
        return graph.compile()

    def process_claim(
        self, folder_path: str, uploaded_at: str | None = None
    ) -> Claim:
        folder = Path(folder_path)
        claim_id = folder.name

        if uploaded_at is None:
            mtime = folder.stat().st_mtime
            uploaded_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

        claim = Claim(claim_id=claim_id, uploaded_at=uploaded_at)

        initial_state: ClaimState = {
            "claim": claim,
            "folder_path": str(folder),
            "field_schemas": self.field_schemas,
            "llm_client": self.llm_client,
            "chatbot": self.chatbot,
            "workflow_config": self.workflow_config,
            "message_config": self.message_config,
            "parser": ClaimParser(),
            "reply_skipped": False,
        }

        final_state = self._graph.invoke(initial_state)
        final_claim: Claim = final_state["claim"]

        # Persist state to .cache/claim_state.json
        cache_dir = folder / ".cache"
        cache_dir.mkdir(exist_ok=True)
        (cache_dir / "claim_state.json").write_text(
            final_claim.model_dump_json(indent=2)
        )

        return final_claim

    def accept_reply(self, claim: Claim) -> Claim:
        """Standalone entry point for async/webhook reply handling."""
        reply_text = self.chatbot.ask(
            f"\nReply for claim {claim.claim_id} (or Enter to skip): "
        )
        if reply_text.strip():
            ClaimParser().handle_reply(claim, reply_text, self.llm_client)
        return claim

    def prioritize_claims(self, claims: list[Claim]) -> list[PriorityRecord]:
        """Sort claims by status priority, then by upload time; assign express flag."""
        priority_reasons = self.message_config.get("priority_reason", {})
        STATUS_ORDER = {"complete": 0, "incomplete": 1, "pending": 2}

        complete = sorted(
            [c for c in claims if c.status == "complete"],
            key=lambda c: c.uploaded_at,
        )
        incomplete_all = sorted(
            [c for c in claims if c.status == "incomplete"],
            key=lambda c: c.uploaded_at,
        )
        pending = sorted(
            [c for c in claims if c.status == "pending"],
            key=lambda c: c.uploaded_at,
        )

        incomplete_express = [
            c for c in incomplete_all
            if sum(1 for vi in c.validation_issues if not vi.resolved) < 2
        ]
        incomplete_standard = [
            c for c in incomplete_all
            if c not in incomplete_express
        ]

        ordered = (
            [(c, False) for c in complete]
            + [(c, True) for c in incomplete_express]
            + [(c, False) for c in incomplete_standard]
            + [(c, False) for c in pending]
        )

        records: list[PriorityRecord] = []
        for rank, (claim, express) in enumerate(ordered, start=1):
            if claim.status == "complete":
                reason = priority_reasons.get(
                    "complete_oldest",
                    "All documents present and valid — ready to finalize.",
                )
            elif claim.status == "incomplete" and express:
                reason = priority_reasons.get(
                    "incomplete_express",
                    "Express routing — fewer than 2 unresolved issues.",
                )
            elif claim.status == "incomplete":
                reason = priority_reasons.get(
                    "incomplete_standard",
                    "Incomplete claim — customer notification required.",
                )
            else:
                reason = priority_reasons.get(
                    "pending_oldest",
                    "Requires human review.",
                )
            records.append(
                PriorityRecord(
                    claim_id=claim.claim_id,
                    status=claim.status,
                    uploaded_at=claim.uploaded_at,
                    express=express,
                    priority_rank=rank,
                    reason=reason,
                )
            )
        return records
