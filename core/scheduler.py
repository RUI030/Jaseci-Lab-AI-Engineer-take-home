from __future__ import annotations

from core.models import Claim, PriorityRecord


def _is_express_eligible(claim: Claim) -> bool:
    """Return True when the claim qualifies for express routing.

    Requires fewer than 2 unresolved issues AND no missing-doc placeholders.
    """
    return (
        sum(1 for vi in claim.validation_issues if not vi.resolved) < 2
        and not any(r.doc_status == "missing" for r in claim.doc_table)
    )


def _priority_key(claim: Claim) -> int:
    """Return sort key (lower = higher priority)."""
    if claim.status == "complete":
        return 0
    if claim.status == "needs_review":
        return 1
    if claim.status == "incomplete" and _is_express_eligible(claim):
        return 2
    return 3


def prioritize_claims(
    claims: list[Claim], message_config: dict
) -> list[PriorityRecord]:
    """Sort claims by status priority then upload time; assign express flag.

    Priority groups (lowest number = highest priority):
      0 complete → 1 needs_review → 2 incomplete-express → 3 incomplete-standard
    Within each group, oldest uploaded_at wins.
    """
    priority_reasons = message_config.get("priority_reason", {})
    ordered = sorted(claims, key=lambda c: (_priority_key(c), c.uploaded_at))

    records: list[PriorityRecord] = []
    for rank, claim in enumerate(ordered, start=1):
        express = claim.status == "incomplete" and _is_express_eligible(claim)
        if claim.status == "complete":
            reason = priority_reasons.get(
                "complete_oldest",
                "All documents present and valid — ready to finalize.",
            )
        elif claim.status == "needs_review":
            reason = priority_reasons.get(
                "pending_oldest",
                "Requires human review.",
            )
        elif express:
            reason = priority_reasons.get(
                "incomplete_express",
                "Express routing — fewer than 2 unresolved issues.",
            )
        else:
            reason = priority_reasons.get(
                "incomplete_standard",
                "Incomplete claim — customer notification required.",
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
