from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from typing import Literal

from core.doc_reader import TextReader
from core.llm_adapters import BaseLLMClient
from core.models import (
    Claim,
    ConversationRound,
    ExtractedField,
    FieldSchema,
    ValidationIssue,
)

REQUIRED_DOC_TYPES = ["police_report", "finance_agreement", "settlement_breakdown"]


class ClaimParser:
    def check_required_docs(self, claim: Claim) -> list[str]:
        """Return list of required doc types that are absent or not present."""
        present = {
            r.doc_type
            for r in claim.doc_table
            if r.doc_status == "present" and r.doc_type != "unknown"
        }
        return [t for t in REQUIRED_DOC_TYPES if t not in present]

    def cross_validate(self, claim: Claim) -> list[ValidationIssue]:
        """Compare unified_value of each field across all document sources.

        Only considers fields where source_trust == 'document'. Never modifies
        confidence values. Returns only the new ValidationIssues added this round.

        Severity rules:
          - 'blocking': two or more medium/high confidence sources disagree.
          - 'warning':  only low-confidence sources are involved in the conflict,
                        or a single low-confidence source disagrees with a
                        high/medium-confidence consensus. Company is notified but
                        the claim is not held up.
        """
        _CONF_RANK = {"high": 2, "medium": 1, "low": 0}

        # field_name -> {file_name: (unified_value, confidence)}
        field_data: dict[str, dict[str, tuple[str, str]]] = {}
        for record in claim.doc_table:
            if record.parse_status != "complete":
                continue
            for field in record.fields:
                if field.source_trust != "document" or field.unified_value is None:
                    continue
                field_data.setdefault(field.field_name, {})[record.file_name] = (
                    field.unified_value,
                    field.confidence,
                )

        new_issues: list[ValidationIssue] = []
        for field_name, source_map in field_data.items():
            values_only = {f: v for f, (v, _) in source_map.items()}
            if len(set(values_only.values())) <= 1:
                continue

            already = any(
                vi.field_name == field_name
                and vi.issue_type == "inconsistency"
                and not vi.resolved
                for vi in claim.validation_issues
            )
            if already:
                continue

            # Count how many medium/high confidence sources have a differing value
            # from the majority (highest-confidence) value.
            by_conf = sorted(
                source_map.items(),
                key=lambda kv: _CONF_RANK.get(kv[1][1], 0),
                reverse=True,
            )
            top_value = by_conf[0][1][0]
            strong_dissenters = sum(
                1
                for _, (v, c) in source_map.items()
                if v != top_value and _CONF_RANK.get(c, 0) >= 1
            )
            severity = "blocking" if strong_dissenters >= 1 else "warning"

            issue = ValidationIssue(
                issue_type="inconsistency",
                severity=severity,
                field_name=field_name,
                description=(
                    f"Field '{field_name}' has different values across documents: "
                    + ", ".join(
                        f"{f}: {v} (confidence={c})" for f, (v, c) in source_map.items()
                    )
                ),
                sources=list(values_only.keys()),
                values=values_only,
            )
            claim.validation_issues.append(issue)
            new_issues.append(issue)

        return new_issues

    def determine_status(
        self, claim: Claim
    ) -> Literal["complete", "incomplete", "pending"]:
        """Evaluate doc_table and validation_issues to assign claim status.

        Priority order matches ARCHITECTURE.md status aggregation logic.
        """
        # 1. Missing required document
        if any(r.doc_status == "missing" for r in claim.doc_table):
            return "incomplete"

        # 2. same_filename duplicate
        if any(
            r.doc_status == "duplicate" and r.duplicate_type == "same_filename"
            for r in claim.doc_table
        ):
            return "incomplete"

        # 3. same_content duplicate
        if any(
            r.doc_status == "duplicate" and r.duplicate_type == "same_content"
            for r in claim.doc_table
        ):
            return "pending"

        # Single pass over validation_issues for all unresolved types
        blocking_inconsistencies = []
        low_conf_issues = []
        for vi in claim.validation_issues:
            if vi.resolved:
                continue
            if vi.issue_type == "inconsistency" and vi.severity == "blocking":
                blocking_inconsistencies.append(vi)
            elif vi.issue_type == "low_confidence":
                low_conf_issues.append(vi)
        # Warning-severity inconsistencies are logged for staff review but do not
        # affect routing — only blocking ones change the claim status.

        # 4/5. Blocking inconsistency — incomplete before first reply, pending after
        if blocking_inconsistencies:
            return "pending" if claim.reply_count > 0 else "incomplete"

        # 6. Parse failed
        if any(r.parse_status == "parse_failed" for r in claim.doc_table):
            return "pending"

        # 7. Unknown doc type
        if any(r.doc_type == "unknown" for r in claim.doc_table):
            return "pending"

        # 8. Unresolved low-confidence required field
        if low_conf_issues:
            return "pending"

        # Also check extracted_fields directly for low-confidence required fields
        for field in claim.extracted_fields.values():
            if field.field_role == "required" and field.confidence == "low":
                return "pending"

        # 9. Defensive check for required docs absent from doc_table entirely.
        # D5: Step 1 already catches doc_status == "missing" placeholders, which
        # node_parse_documents always inserts for required types not found on disk.
        # This step only fires if a required type somehow has no record at all
        # (e.g., if a caller bypassed node_parse_documents). Kept as a safety net.
        missing = self.check_required_docs(claim)
        if missing:
            return "incomplete"

        # 10. All required docs present and parsed + all required fields valid
        required_complete = all(
            r.parse_status == "complete"
            for r in claim.doc_table
            if r.doc_role == "required" and r.doc_status == "present"
        )
        required_fields_valid = all(
            f.valid
            for f in claim.extracted_fields.values()
            if f.field_role == "required"
        )
        if required_complete and required_fields_valid:
            return "complete"

        # Fallback: something unresolved
        return "incomplete"

    def extract_reply_fields(
        self,
        reply_text: str,
        schemas: list[FieldSchema],
        client: BaseLLMClient,
    ) -> list[ExtractedField]:
        """Extract fields from a customer reply; hard-cap source_trust and confidence.

        R1: import os moved to module top.
        R2: renamed parse_reply → extract_reply_fields (does field extraction, not just parsing).
        R3: parameter renamed target_fields → schemas.
        """
        # R1: os imported at module top
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(reply_text)
                tmp_path = tmp.name
            record = TextReader().read(tmp_path, schemas, client)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
        fields = record.fields

        # Override trust and confidence for all fields from customer replies
        for field in fields:
            field.source_trust = "user_input"
            field.confidence = "low"
            note = "Source is customer reply — confidence hard-capped at low."
            field.confidence_note = (
                (field.confidence_note + " " if field.confidence_note else "") + note
            ).strip()

        return fields

    def compare_fields(
        self,
        claim: Claim,
        new_fields: list[ExtractedField],
    ) -> dict[str, Literal["consistent", "inconsistent"]]:
        """Compare new_fields unified_value against claim.extracted_fields.

        New fields not in existing dict are treated as consistent (no prior conflict).
        Never modifies ValidationIssue.resolved.
        """
        results: dict[str, Literal["consistent", "inconsistent"]] = {}
        for field in new_fields:
            existing = claim.extracted_fields.get(field.field_name)
            if existing is None or existing.unified_value is None or field.unified_value is None:
                results[field.field_name] = "consistent"
                continue
            if existing.unified_value == field.unified_value:
                results[field.field_name] = "consistent"
            else:
                results[field.field_name] = "inconsistent"
        return results

    def record_reply(
        self,
        claim: Claim,
        message: str,
        compare_results: dict[str, Literal["consistent", "inconsistent"]],
    ) -> None:
        """Append an inbound ConversationRound and increment reply_count.

        R2: renamed log_reply → record_reply ("log" implies a logger; this appends to the conversation list).
        """
        claim.reply_count += 1
        claim.conversation_log.append(
            ConversationRound(
                round=len(claim.conversation_log) + 1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                direction="inbound",
                message=message,
                compare_results=compare_results,
            )
        )

    def handle_reply(
        self,
        claim: Claim,
        reply_text: str,
        client: BaseLLMClient,
        schemas: list[FieldSchema],
    ) -> Claim:
        """Orchestrate reply processing: extract fields → compare → record → determine status.

        R3: parameter renamed field_schemas → schemas for consistency.
        """
        unresolved_fields = {
            vi.field_name
            for vi in claim.validation_issues
            if not vi.resolved and vi.field_name
        }
        focused_schemas = (
            [s for s in schemas if s.field_name in unresolved_fields]
            if unresolved_fields
            else schemas
        )

        new_fields = self.extract_reply_fields(reply_text, focused_schemas, client)
        compare_results = self.compare_fields(claim, new_fields)

        # Add ValidationIssue for any new inconsistencies introduced by the reply
        for field_name, result in compare_results.items():
            if result == "inconsistent":
                already = any(
                    vi.field_name == field_name
                    and vi.issue_type == "inconsistency"
                    and not vi.resolved
                    for vi in claim.validation_issues
                )
                if not already:
                    existing = claim.extracted_fields.get(field_name)
                    new_val = next(
                        (f.unified_value for f in new_fields if f.field_name == field_name),
                        None,
                    )
                    claim.validation_issues.append(
                        ValidationIssue(
                            issue_type="inconsistency",
                            field_name=field_name,
                            description=(
                                f"Customer reply value for '{field_name}' conflicts "
                                f"with document value."
                            ),
                            sources=["document", "user_input"],
                            values={
                                "document": (existing.unified_value or "") if existing else "",
                                "user_input": new_val or "",
                            },
                        )
                    )

        self.record_reply(claim, reply_text, compare_results)
        claim.status = self.determine_status(claim)
        return claim
