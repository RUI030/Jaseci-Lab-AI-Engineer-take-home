from __future__ import annotations

import re
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
from core.tools import check_field_consistency, validate_vin

_CONF_WEIGHT: dict[str, int] = {"high": 3, "medium": 2, "low": 1}
_VIN_PATTERN = r"^[A-Z0-9]{17}$"   # standard 17-char alphanumeric VIN (case-insensitive)
_MONEY_PCT_THRESHOLD = 10.0         # percentage difference above which money values are "significant"


def _pct_diff(a: str, b: str) -> float:
    """Percentage difference between two numeric strings relative to the larger value."""
    try:
        fa, fb = float(a), float(b)
        denom = max(abs(fa), abs(fb))
        return 0.0 if denom == 0 else abs(fa - fb) / denom * 100.0
    except ValueError:
        return 100.0

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

        Algorithm (per field):
          1. Weighted vote: high=3, medium=2, low=1. If the winning value holds
             a strict weight majority (>50%), it is trusted → severity='warning'.
          2. No clear majority, exactly 2 sources:
             - VIN (string): char diff > 6 → 'blocking' + resubmit lower-confidence doc.
             - number: diff > 10% → 'blocking' + resubmit; diff ≤ 10% → 'warning',
               note to trust the smaller value.
             - other types → 'blocking' + resubmit lower-confidence doc (if identifiable).
          3. No clear majority, 3+ sources → 'blocking'.
        """
        # field_name -> {file_name: (unified_value, confidence, data_type)}
        field_data: dict[str, dict[str, tuple[str, str, str]]] = {}
        for record in claim.doc_table:
            if record.parse_status != "complete":
                continue
            for field in record.fields:
                if field.source_trust != "document" or field.unified_value is None:
                    continue
                field_data.setdefault(field.field_name, {})[record.file_name] = (
                    field.unified_value,
                    field.confidence,
                    field.data_type,
                )

        new_issues: list[ValidationIssue] = []
        for field_name, source_map in field_data.items():
            values_only = {f: v for f, (v, _, _) in source_map.items()}
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

            # Log tool calls for this multi-source field
            claim.tools_used.append(check_field_consistency(field_name, values_only))

            # --- weighted voting ---
            weighted: dict[str, int] = {}
            for _, (val, conf, _) in source_map.items():
                weighted[val] = weighted.get(val, 0) + _CONF_WEIGHT.get(conf, 1)
            total_weight = sum(weighted.values())
            winner_value = max(weighted, key=lambda v: weighted[v])
            winner_weight = weighted[winner_value]

            resubmit_doc: str | None = None

            if winner_weight * 2 > total_weight:
                # Clear weighted majority — low-confidence minority overruled
                severity: Literal["blocking", "warning"] = "warning"
                overruled = ", ".join(
                    f"{f}: {v} (confidence={c})"
                    for f, (v, c, _) in source_map.items()
                    if v != winner_value
                )
                description = (
                    f"Field '{field_name}' has conflicting values; "
                    f"trusted value '{winner_value}' (confidence-weighted majority). "
                    f"Overruled: {overruled}."
                )
            else:
                # No clear majority — apply significance checks
                items = list(source_map.items())
                data_type = items[0][1][2]  # same field → same data_type

                resubmit_doc = None

                if data_type == "string" and "vin" in field_name.lower():
                    # Validate each unique VIN value and log the results
                    for vin_val in {v for _, (v, _, _) in source_map.items()}:
                        claim.tools_used.append(validate_vin(vin_val))
                    # Regex validity is a stronger signal than confidence weights for VINs.
                    # Filter to sources whose value passes the VIN format check first;
                    # then weighted-vote among those. If none pass, block.
                    valid_sources = {
                        f: (v, c)
                        for f, (v, c, _) in source_map.items()
                        if re.fullmatch(_VIN_PATTERN, v, re.IGNORECASE)
                    }
                    if not valid_sources:
                        # No structurally valid VIN anywhere — must resubmit
                        min_w = min(_CONF_WEIGHT.get(c, 1) for _, (_, c, _) in source_map.items())
                        lowest = [f for f, (_, c, _) in source_map.items() if _CONF_WEIGHT.get(c, 1) == min_w]
                        resubmit_doc = lowest[0] if len(lowest) == 1 else None
                        severity = "blocking"
                        description = (
                            f"Field '{field_name}': no source contains a valid 17-character "
                            "alphanumeric VIN. Values: "
                            + ", ".join(f"'{f}': '{v}'" for f, (v, _, _) in source_map.items())
                            + "."
                        )
                    elif len(valid_sources) == 1:
                        # Exactly one structurally valid VIN — trust it unconditionally
                        trusted_file, (trusted_value, _) = next(iter(valid_sources.items()))
                        invalid = [
                            (f, v) for f, (v, _, _) in source_map.items()
                            if f not in valid_sources
                        ]
                        severity = "warning"
                        description = (
                            f"Field '{field_name}': trusted value '{trusted_value}' "
                            f"from '{trusted_file}' (only structurally valid VIN). "
                            + "; ".join(
                                f"'{f}': '{v}' is not a valid 17-char VIN"
                                for f, v in invalid
                            )
                            + "."
                        )
                    else:
                        # Multiple valid VINs — weighted vote among regex-passing sources
                        valid_weighted: dict[str, int] = {}
                        for _, (v, c) in valid_sources.items():
                            valid_weighted[v] = valid_weighted.get(v, 0) + _CONF_WEIGHT.get(c, 1)
                        vin_winner = max(valid_weighted, key=lambda v: valid_weighted[v])
                        severity = "warning"
                        description = (
                            f"Field '{field_name}': multiple valid VINs found; "
                            f"confidence-weighted trusted value is '{vin_winner}'. "
                            "Sources: "
                            + ", ".join(
                                f"'{f}': '{v}' (confidence={c})"
                                for f, (v, c) in valid_sources.items()
                            )
                            + "."
                        )

                elif len(items) == 2:
                    f1, (v1, c1, _) = items[0]
                    f2, (v2, c2, _) = items[1]
                    w1 = _CONF_WEIGHT.get(c1, 1)
                    w2 = _CONF_WEIGHT.get(c2, 1)
                    lower_conf_doc = f1 if w1 < w2 else (f2 if w2 < w1 else None)

                    if data_type == "number":
                        pct = _pct_diff(v1, v2)
                        if pct <= _MONEY_PCT_THRESHOLD:
                            try:
                                smaller = str(min(float(v1), float(v2)))
                            except ValueError:
                                smaller = v1
                            severity = "warning"
                            description = (
                                f"Field '{field_name}' differs by {pct:.1f}% "
                                f"({f1}: '{v1}', {f2}: '{v2}') — within {_MONEY_PCT_THRESHOLD}% "
                                f"tolerance; conservative value '{smaller}' recommended."
                            )
                        else:
                            severity = "blocking"
                            resubmit_doc = lower_conf_doc
                            description = (
                                f"Field '{field_name}' has a significant discrepancy "
                                f"({pct:.1f}%): {f1}: '{v1}', {f2}: '{v2}'."
                            )
                    else:
                        severity = "blocking"
                        resubmit_doc = lower_conf_doc
                        description = (
                            f"Field '{field_name}' has conflicting values: "
                            f"{f1}: '{v1}' (confidence={c1}), {f2}: '{v2}' (confidence={c2})."
                        )

                else:
                    # 3+ sources, no weighted majority
                    severity = "blocking"
                    description = (
                        f"Field '{field_name}' has conflicting values across "
                        f"{len(items)} sources with no clear majority: "
                        + ", ".join(
                            f"{f}: '{v}' (confidence={c})"
                            for f, (v, c, _) in source_map.items()
                        )
                        + "."
                    )

            issue = ValidationIssue(
                issue_type="inconsistency",
                severity=severity,
                field_name=field_name,
                description=description,
                sources=list(values_only.keys()),
                values=values_only,
                resubmit_doc=resubmit_doc,
            )
            claim.validation_issues.append(issue)
            new_issues.append(issue)

        return new_issues

    def determine_status(
        self, claim: Claim
    ) -> Literal["complete", "incomplete", "needs_review"]:
        """Evaluate doc_table and validation_issues to assign claim status.

        Priority order matches ARCHITECTURE.md status aggregation logic.
        Two-criterion split:
          incomplete   = customer must act (missing doc, same-filename duplicate)
          needs_review = human must act (everything else wrong)
          complete     = nothing wrong
        """
        # 1. Missing required document — customer can fix
        if any(r.doc_status == "missing" for r in claim.doc_table):
            return "incomplete"

        # 2. Same-filename duplicate — customer can fix by resubmitting
        if any(
            r.doc_status == "duplicate" and r.duplicate_type == "same_filename"
            for r in claim.doc_table
        ):
            return "incomplete"

        # 3. Same-content duplicate or multiple versions of the same doc type — human judgment needed
        if any(
            r.doc_status == "duplicate" and r.duplicate_type in ("same_content", "multiple_versions")
            for r in claim.doc_table
        ):
            return "needs_review"

        # 4. Unresolved blocking inconsistency
        # Warning-severity inconsistencies are logged for staff but do not affect routing.
        blocking_inconsistencies = [
            vi for vi in claim.validation_issues
            if not vi.resolved and vi.issue_type == "inconsistency" and vi.severity == "blocking"
        ]
        if blocking_inconsistencies:
            return "needs_review"

        # 5. Parse failed
        if any(r.parse_status == "parse_failed" for r in claim.doc_table):
            return "needs_review"

        # 6. Unknown doc type
        if any(r.doc_type == "unknown" for r in claim.doc_table):
            return "needs_review"

        # 7. Unresolved low-confidence required field
        low_conf_issues = [
            vi for vi in claim.validation_issues
            if not vi.resolved and vi.issue_type == "low_confidence"
        ]
        if low_conf_issues:
            return "needs_review"
        for field in claim.extracted_fields.values():
            if field.field_role == "required" and field.confidence == "low":
                return "needs_review"

        # 8. Defensive check for required docs absent from doc_table entirely.
        # Step 1 already catches missing-doc placeholders that node_parse_documents inserts.
        # This only fires if a caller bypassed node_parse_documents entirely — kept as a safety net.
        missing = self.check_required_docs(claim)
        if missing:
            return "incomplete"

        # 9. All required docs present and parsed + all required fields valid
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
        """Extract fields from a customer reply; hard-cap source_trust and confidence."""
        record = TextReader().read_text(reply_text, schemas, client)
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
        schemas: list[FieldSchema],
    ) -> Claim:
        """Orchestrate reply processing: extract fields → compare → record → determine status."""
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

    def build_customer_message(self, claim: Claim, message_config: dict) -> str:
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

        if claim.status == "needs_review" and not issue_lines:
            issue_lines.append(fragments["pending_review"].strip())

        issues_body = "\n\n".join(f"{i + 1}. {line}" for i, line in enumerate(issue_lines))
        return email_tmpl.format(claim_id=claim.claim_id, issues_body=issues_body).strip()
