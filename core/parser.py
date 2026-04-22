from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
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
        """
        # Gather all doc-sourced extracted fields grouped by field_name
        field_values: dict[str, dict[str, str]] = {}  # field_name -> {file_name: unified_value}
        for record in claim.doc_table:
            if record.parse_status != "complete":
                continue
            for field in record.fields:
                if field.source_trust != "document":
                    continue
                if field.unified_value is None:
                    continue
                field_values.setdefault(field.field_name, {})[record.file_name] = (
                    field.unified_value
                )

        new_issues: list[ValidationIssue] = []
        for field_name, source_map in field_values.items():
            distinct_values = set(source_map.values())
            if len(distinct_values) <= 1:
                continue
            # Check if an equivalent issue already exists (avoid duplicates)
            already = any(
                vi.field_name == field_name
                and vi.issue_type == "inconsistency"
                and not vi.resolved
                for vi in claim.validation_issues
            )
            if already:
                continue
            issue = ValidationIssue(
                issue_type="inconsistency",
                field_name=field_name,
                description=(
                    f"Field '{field_name}' has different values across documents: "
                    + ", ".join(f"{f}: {v}" for f, v in source_map.items())
                ),
                sources=list(source_map.keys()),
                values=dict(source_map),
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

        unresolved_inconsistencies = [
            vi
            for vi in claim.validation_issues
            if vi.issue_type == "inconsistency" and not vi.resolved
        ]

        # 4/5. Inconsistency — incomplete before first reply, pending after
        if unresolved_inconsistencies:
            return "pending" if claim.reply_count > 0 else "incomplete"

        # 6. Parse failed
        if any(r.parse_status == "parse_failed" for r in claim.doc_table):
            return "pending"

        # 7. Unknown doc type
        if any(r.doc_type == "unknown" for r in claim.doc_table):
            return "pending"

        # 8. Unresolved low-confidence required field
        low_conf_issues = [
            vi
            for vi in claim.validation_issues
            if vi.issue_type == "low_confidence" and not vi.resolved
        ]
        if low_conf_issues:
            return "pending"

        # Also check extracted_fields directly for low-confidence required fields
        for field in claim.extracted_fields.values():
            if field.field_role == "required" and field.confidence == "low":
                return "pending"

        # 9. Check missing required docs (via check_required_docs helper)
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

    def parse_reply(
        self,
        reply_text: str,
        target_fields: list[FieldSchema],
        client: BaseLLMClient,
    ) -> list[ExtractedField]:
        """Parse a customer reply via TextReader; hard-cap source_trust and confidence."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(reply_text)
            tmp_path = tmp.name

        record = TextReader().read(tmp_path, target_fields, client)
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
            if existing is None or existing.unified_value is None:
                results[field.field_name] = "consistent"
                continue
            if field.unified_value is None:
                results[field.field_name] = "consistent"
                continue
            if existing.unified_value == field.unified_value:
                results[field.field_name] = "consistent"
            else:
                results[field.field_name] = "inconsistent"
        return results

    def log_reply(
        self,
        claim: Claim,
        message: str,
        compare_results: dict[str, Literal["consistent", "inconsistent"]],
    ) -> None:
        """Append an inbound ConversationRound and increment reply_count."""
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
    ) -> Claim:
        """Orchestrate reply processing: parse → compare → log → determine status."""
        # Load field schemas scoped to unresolved issues
        schema_path = Path(__file__).parent.parent / "config" / "field_schema.json"
        all_schemas = [
            FieldSchema(**item)
            for item in json.loads(schema_path.read_text())
        ]

        unresolved_fields = {
            vi.field_name
            for vi in claim.validation_issues
            if not vi.resolved and vi.field_name
        }
        target_fields = (
            [s for s in all_schemas if s.field_name in unresolved_fields]
            if unresolved_fields
            else all_schemas
        )

        new_fields = self.parse_reply(reply_text, target_fields, client)
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
                                "document": existing.unified_value if existing else "",
                                "user_input": new_val or "",
                            },
                        )
                    )

        self.log_reply(claim, reply_text, compare_results)
        claim.status = self.determine_status(claim)
        return claim
