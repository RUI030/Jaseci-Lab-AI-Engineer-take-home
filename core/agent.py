from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from core.chatbot import Chatbot
from core.doc_reader import get_doc_reader, _hash_file
from core.llm_adapters import BaseLLMClient, LLMClientFactory
from core.models import (
    Claim,
    ConversationRound,
    DocRecord,
    FieldSchema,
    PriorityRecord,
)
from core.parser import REQUIRED_DOC_TYPES, ClaimParser
from core.utils import load_yaml, load_field_schemas


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

    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )

    seen_hashes: dict[str, str] = {}  # hash -> file_name
    seen_names: dict[str, str] = {}   # name -> file_path

    # H2: track required types that were attempted but may have failed to parse,
    # so the missing-placeholder loop does not double-count them as "missing".
    attempted_required_types: set[str] = set()

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
            content_hash = _hash_file(str(file_path))
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
            reader = get_doc_reader(str(file_path))
            record = reader.read(str(file_path), schemas, client)
            if record.doc_type in REQUIRED_DOC_TYPES:
                attempted_required_types.add(record.doc_type)
        except Exception as exc:  # L4: ValueError ⊂ Exception, single catch is enough
            # Use filename to infer intended required type before discarding it
            for req_type in REQUIRED_DOC_TYPES:
                if req_type in file_name.lower().replace("-", "_"):
                    attempted_required_types.add(req_type)
                    break
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
    confidence_rank = {"high": 3, "medium": 2, "low": 1}  # L16: lowercase local var
    for record in claim.doc_table:
        for field in record.fields:
            existing = claim.extracted_fields.get(field.field_name)
            if existing is None:
                claim.extracted_fields[field.field_name] = field
            elif confidence_rank.get(field.confidence, 0) > confidence_rank.get(
                existing.confidence, 0
            ):
                claim.extracted_fields[field.field_name] = field

    # --- placeholder DocRecords for missing required docs ---
    # H2: only mark as missing if the type was neither successfully parsed
    # nor attempted (attempted-but-parse_failed is handled by rule #6).
    present_types = {
        r.doc_type for r in claim.doc_table
        if r.doc_status == "present" and r.parse_status == "complete"
    }
    for doc_type in set(REQUIRED_DOC_TYPES) - present_types - attempted_required_types:
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
    """Assemble a single unified email from individual issue fragments."""
    fragments = message_config.get("issue_fragments")
    if not fragments:
        raise KeyError("messages.yaml is missing required 'issue_fragments' section")
    email_tmpl = message_config.get("customer_email")
    if not email_tmpl:
        raise KeyError("messages.yaml is missing required 'customer_email' template")

    issue_lines: list[str] = []

    missing_docs = [r.doc_type for r in claim.doc_table if r.doc_status == "missing"]
    if missing_docs:
        issue_lines.append(
            fragments["missing_document"].format(
                missing_docs=", ".join(missing_docs)
            ).strip()
        )

    parse_failed_docs = [r.file_name for r in claim.doc_table if r.parse_status == "parse_failed"]
    if parse_failed_docs:
        issue_lines.append(
            fragments["parse_failed"].format(
                failed_docs=", ".join(parse_failed_docs)
            ).strip()
        )

    for vi in claim.validation_issues:
        if vi.issue_type != "inconsistency" or vi.resolved:
            continue
        if vi.severity == "blocking":
            if vi.resubmit_doc:
                issue_lines.append(
                    fragments["resubmit_document"].format(
                        field_name=vi.field_name or "unknown field",
                        resubmit_doc=vi.resubmit_doc,
                    ).strip()
                )
            else:
                details = "\n".join(f"  - {k}: {v}" for k, v in vi.values.items())
                issue_lines.append(
                    fragments["field_inconsistency"].format(
                        field_name=vi.field_name or "unknown field",
                        inconsistency_details=details,
                    ).strip()
                )
        else:
            issue_lines.append(
                fragments["field_inconsistency_warning"].format(
                    field_name=vi.field_name or "unknown field",
                ).strip()
            )

    if claim.status == "pending" and not issue_lines:
        issue_lines.append(fragments["pending_review"].strip())

    issues_body = "\n\n".join(f"{i + 1}. {line}" for i, line in enumerate(issue_lines))
    return email_tmpl.format(claim_id=claim.claim_id, issues_body=issues_body).strip()


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
    skipped = True
    if reply_text.strip():
        parser.handle_reply(claim, reply_text, client, state["field_schemas"])
        skipped = False
    else:
        # Customer skipped — if any open resubmit request exists, treat as refusal → human review
        has_resubmit = any(
            vi.resubmit_doc and not vi.resolved
            for vi in claim.validation_issues
        )
        if has_resubmit:
            claim.status = "pending"

    state["reply_skipped"] = skipped
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


# --- H6 + L9: express eligibility and single-pass prioritization ---

def _is_express_eligible(claim: Claim) -> bool:
    """Return True when the claim qualifies for express routing.

    Requires fewer than 2 unresolved issues AND no missing-doc placeholders.
    Missing docs are excluded because they represent absent required files, not
    customer-fixable issues that express routing is designed for.
    """
    return (
        sum(1 for vi in claim.validation_issues if not vi.resolved) < 2
        and not any(r.doc_status == "missing" for r in claim.doc_table)
    )


def _priority_key(claim: Claim) -> int:
    """Return sort key for prioritize_claims (lower = higher priority)."""
    if claim.status == "complete":
        return 0
    if claim.status == "incomplete" and _is_express_eligible(claim):
        return 1
    if claim.status == "incomplete":
        return 2
    return 3  # pending


class ClaimAgent:
    def __init__(self, chatbot: Chatbot) -> None:
        self.chatbot = chatbot
        self.workflow_config = load_yaml("config/workflow.yaml")
        self.message_config = load_yaml("config/messages.yaml")
        self.field_schemas = load_field_schemas()

        settings = load_yaml("config/settings.yaml")
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
        self,
        folder_path: str,
        uploaded_at: str | None = None,
        use_cache: bool = True,
    ) -> Claim:
        folder = Path(folder_path)
        claim_id = folder.name

        cache_path = folder / ".cache" / "claim_state.json"
        if use_cache and cache_path.exists():
            cached = Claim.model_validate_json(cache_path.read_text())
            if cached.status == "complete":
                self.chatbot.display(f"  [cache] {claim_id} is complete — loaded from cache.")
                return cached
            self.chatbot.display(
                f"  [cache] {claim_id} has status '{cached.status}' — re-processing."
            )

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

    def prioritize_claims(self, claims: list[Claim]) -> list[PriorityRecord]:
        """Sort claims by status priority then upload time; assign express flag.

        Priority groups (lowest number = highest priority):
          0 complete → 1 incomplete-express → 2 incomplete-standard → 3 pending
        Within each group, oldest upload_at wins.
        """
        priority_reasons = self.message_config.get("priority_reason", {})

        ordered = sorted(claims, key=lambda c: (_priority_key(c), c.uploaded_at))

        records: list[PriorityRecord] = []
        for rank, claim in enumerate(ordered, start=1):
            express = claim.status == "incomplete" and _is_express_eligible(claim)
            if claim.status == "complete":
                reason = priority_reasons.get(
                    "complete_oldest",
                    "All documents present and valid — ready to finalize.",
                )
            elif express:
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
