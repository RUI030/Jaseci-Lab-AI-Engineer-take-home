#!/usr/bin/env python3
"""Generate a standalone Bootstrap HTML claim report from cached claim state files.

Reads .cache/claim_state.json for every claim folder — no API key required.
Run main.py first if caches are missing.

Usage (from any working directory):
    python demo/gen_visual.py
    python demo/gen_visual.py --claims-dir path/to/claims
    python demo/gen_visual.py --output /tmp/report.html

Defaults:
    --claims-dir  <project_root>/claims/
    --output      <project_root>/demo/claim_report.html
"""
from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent   # project root (this file lives in demo/)
sys.path.insert(0, str(ROOT))

from core.models import Claim, PriorityRecord
from core.utils import load_yaml


# ── Priority helpers (mirrors agent.py logic, no LLM init needed) ────────────

def _is_express_eligible(claim: Claim) -> bool:
    return (
        sum(1 for vi in claim.validation_issues if not vi.resolved) < 2
        and not any(r.doc_status == "missing" for r in claim.doc_table)
    )


def _priority_key(claim: Claim) -> int:
    if claim.status == "complete":
        return 0
    if claim.status == "needs_review":
        return 1  # human must act — staff can do something now
    if claim.status == "incomplete" and _is_express_eligible(claim):
        return 2  # waiting on customer, but nearly resolved
    return 3  # incomplete — waiting on customer, nothing staff can do yet


def _prioritize(claims: list[Claim]) -> list[PriorityRecord]:
    reasons = load_yaml("config/messages.yaml").get("priority_reason", {})
    ordered = sorted(claims, key=lambda c: (_priority_key(c), c.uploaded_at))
    records: list[PriorityRecord] = []
    for rank, claim in enumerate(ordered, start=1):
        express = claim.status == "incomplete" and _is_express_eligible(claim)
        if claim.status == "complete":
            reason = reasons.get("complete_oldest", "All documents present and valid — ready to finalise.")
        elif claim.status == "needs_review":
            reason = reasons.get("pending_oldest", "Requires human review.")
        elif express:
            reason = reasons.get("incomplete_express", "Express routing — fewer than 2 unresolved issues.")
        else:
            reason = reasons.get("incomplete_standard", "Incomplete claim — customer notification required.")
        records.append(PriorityRecord(
            claim_id=claim.claim_id,
            status=claim.status,
            uploaded_at=claim.uploaded_at,
            express=express,
            priority_rank=rank,
            reason=reason,
        ))
    return records


# ── Bootstrap variant maps ────────────────────────────────────────────────────

_STATUS_V  = {"complete": "success", "incomplete": "danger", "needs_review": "warning"}
_CONF_V    = {"high": "success", "medium": "warning", "low": "danger"}
_DOCST_V   = {"present": "success", "missing": "danger", "duplicate": "warning"}
_SEV_V     = {"blocking": "danger", "warning": "warning"}


def _e(text: object) -> str:
    return html.escape(str(text))


def _badge(text: str, variant: str) -> str:
    return f'<span class="badge text-bg-{variant}">{_e(text)}</span>'


# ── Per-claim card ────────────────────────────────────────────────────────────

def _render_claim(claim: Claim) -> str:
    sv = _STATUS_V.get(claim.status, "secondary")
    p: list[str] = []

    # header
    p.append(f"""
    <div class="card mb-4 border-{sv}">
      <div class="card-header bg-{sv} bg-opacity-10 d-flex justify-content-between align-items-center">
        <h5 class="mb-0 fw-bold font-monospace">{_e(claim.claim_id)}</h5>
        <div>
          {_badge(claim.status.upper(), sv)}
          <small class="text-muted ms-2">uploaded {_e(claim.uploaded_at[:10])}</small>
          <small class="text-muted ms-2">· {claim.reply_count} reply round(s)</small>
        </div>
      </div>
      <div class="card-body">
    """)

    # documents
    p.append('<h6 class="mt-1">Documents</h6>')
    p.append("""<table class="table table-sm table-hover table-bordered small mb-3">
      <thead class="table-light"><tr>
        <th>File</th><th>Type</th><th>Status</th><th>Parse</th>
      </tr></thead><tbody>""")
    for r in claim.doc_table:
        dv = _DOCST_V.get(r.doc_status, "secondary")
        dup = f' <small class="text-muted">({_e(r.duplicate_type)})</small>' if r.duplicate_type else ""
        reason_tip = f' title="{_e(r.status_reason)}"' if r.status_reason else ""
        parse_cell = (
            '<span class="text-success">complete</span>' if r.parse_status == "complete"
            else f'<span class="text-danger">{_e(r.parse_status)}</span>'
        )
        p.append(f"""<tr{reason_tip}>
          <td class="font-monospace">{_e(r.file_name)}</td>
          <td>{_e(r.doc_type)}</td>
          <td>{_badge(r.doc_status, dv)}{dup}</td>
          <td>{parse_cell}</td>
        </tr>""")
    p.append("</tbody></table>")

    # extracted fields
    p.append('<h6>Extracted Fields <small class="text-muted fw-normal">(highest-confidence wins per field)</small></h6>')
    if claim.extracted_fields:
        p.append("""<table class="table table-sm table-hover table-bordered small mb-3">
          <thead class="table-light"><tr>
            <th>Field</th><th>Value</th><th>Confidence</th><th>Valid</th>
          </tr></thead><tbody>""")
        for f in claim.extracted_fields.values():
            cv = _CONF_V.get(f.confidence, "secondary")
            val = _e(f.unified_value) if f.unified_value else '<em class="text-muted">not found</em>'
            valid_icon = "✓" if f.valid else "✗"
            valid_cls  = "text-success" if f.valid else "text-danger"
            note = ""
            if f.validation_note or f.confidence_note:
                tip = _e((f.validation_note or "") + (" | " + f.confidence_note if f.confidence_note else ""))
                note = f' <small class="text-muted" title="{tip}">ⓘ</small>'
            p.append(f"""<tr>
              <td>{_e(f.field_name)}{note} <small class="text-muted">({_e(f.field_role)})</small></td>
              <td class="font-monospace">{val}</td>
              <td>{_badge(f.confidence, cv)}</td>
              <td class="{valid_cls} fw-bold">{valid_icon}</td>
            </tr>""")
        p.append("</tbody></table>")
    else:
        p.append('<p class="text-muted fst-italic small">No fields extracted.</p>')

    # validation issues
    if claim.validation_issues:
        p.append('<h6>Validation Issues</h6><ul class="list-group list-group-flush mb-3">')
        for vi in claim.validation_issues:
            vv = _SEV_V.get(vi.severity, "secondary")
            resolved = ' <span class="badge text-bg-success ms-1">resolved</span>' if vi.resolved else ""
            resubmit = (
                f' <span class="badge text-bg-warning ms-1">resubmit: {_e(vi.resubmit_doc)}</span>'
                if vi.resubmit_doc else ""
            )
            p.append(f"""<li class="list-group-item list-group-item-{vv} py-1 small">
              {_badge(vi.severity, vv)}
              <strong class="ms-1">{_e(vi.field_name or vi.issue_type)}</strong>:
              {_e(vi.description)}{resubmit}{resolved}
            </li>""")
        p.append("</ul>")

    # decision summary
    missing  = [r.doc_type for r in claim.doc_table if r.doc_status == "missing"]
    failed   = [r.file_name for r in claim.doc_table if r.parse_status == "parse_failed"]
    blocking = [vi for vi in claim.validation_issues if vi.severity == "blocking" and not vi.resolved]
    warnings = [vi for vi in claim.validation_issues if vi.severity == "warning" and not vi.resolved]
    reasons: list[str] = []
    if missing:  reasons.append(f'Missing: <strong>{_e(", ".join(missing))}</strong>')
    if failed:   reasons.append(f'Parse failed: <strong>{_e(", ".join(failed))}</strong>')
    if blocking: reasons.append(f'{len(blocking)} blocking issue{"s" if len(blocking) > 1 else ""}')
    if warnings: reasons.append(f'{len(warnings)} warning(s) — staff notified')
    if not reasons and claim.status == "complete":
        reasons.append("All required documents present and all fields valid.")
    decision = " &nbsp;·&nbsp; ".join(reasons) if reasons else "No issues detected."
    p.append(f"""<div class="alert alert-{sv} py-2 small mb-2">
      <strong>Decision:</strong> {_badge(claim.status.upper(), sv)} — {decision}
    </div>""")

    # full conversation log (outbound + inbound)
    if claim.conversation_log:
        p.append(f'<h6 class="mt-3">Conversation Log ({len(claim.conversation_log)} round(s))</h6>')
        for cr in claim.conversation_log:
            if cr.direction == "outbound":
                label = "→ Agent → Customer"
                bg = "bg-light"
                border = "border-primary"
                label_cls = "text-primary"
            else:
                label = "← Customer → Agent"
                bg = "bg-white"
                border = "border-success"
                label_cls = "text-success"
            p.append(
                f'<div class="border-start border-3 {border} ps-2 mb-2">'
                f'<small class="{label_cls} fw-semibold">{label}</small>'
                f'<pre class="{bg} border rounded p-2 small mt-1 mb-0">{_e(cr.message)}</pre>'
                f'</div>'
            )

    # tools used
    if claim.tools_used:
        p.append(f'<h6 class="mt-3">Tools Used ({len(claim.tools_used)})</h6>')
        p.append('<ul class="list-unstyled small font-monospace mb-0">')
        for t in claim.tools_used:
            result_summary = ", ".join(f"{k}={v}" for k, v in t.get("result", {}).items())
            p.append(f'<li class="text-muted">· {_e(t["tool"])}({_e(str(t.get("input", {})))}) → {_e(result_summary)}</li>')
        p.append("</ul>")

    p.append("</div></div>")  # close card-body + card
    return "\n".join(p)


# ── Priority ranking table ────────────────────────────────────────────────────

def _render_priority(records: list[PriorityRecord]) -> str:
    rows: list[str] = []
    for r in records:
        sv = _STATUS_V.get(r.status, "secondary")
        express = '<span class="badge text-bg-primary">⚡ EXPRESS</span>' if r.express else ""
        rows.append(f"""<tr>
          <td class="text-center fw-bold">#{r.priority_rank}</td>
          <td class="font-monospace fw-semibold">{_e(r.claim_id)}</td>
          <td>{_badge(r.status.upper(), sv)}</td>
          <td>{express}</td>
          <td>{_e(r.uploaded_at[:10])}</td>
          <td class="text-muted small">{_e(r.reason)}</td>
        </tr>""")
    return f"""
    <div class="card mb-5">
      <div class="card-header fw-semibold">Priority Ranking</div>
      <div class="card-body p-0">
        <table class="table table-sm table-hover mb-0">
          <thead class="table-light"><tr>
            <th>Rank</th><th>Claim</th><th>Status</th>
            <th>Express</th><th>Uploaded</th><th>Reason</th>
          </tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
      </div>
    </div>"""


# ── Full HTML page ────────────────────────────────────────────────────────────

def _build_page(claims: list[Claim], records: list[PriorityRecord]) -> str:
    complete   = sum(1 for c in claims if c.status == "complete")
    incomplete = sum(1 for c in claims if c.status == "incomplete")
    review     = sum(1 for c in claims if c.status == "needs_review")

    cards = "\n".join(_render_claim(c) for c in claims)
    priority_section = _render_priority(records)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Claim Processing Report</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body  {{ background: #f8f9fa; }}
    pre   {{ white-space: pre-wrap; word-break: break-word; font-size: 0.82em; }}
    .font-monospace {{ font-size: 0.88em; }}
    small[title] {{ cursor: help; }}
  </style>
</head>
<body>
<div class="container py-5">

  <div class="d-flex align-items-baseline gap-3 mb-1">
    <h1 class="mb-0">Claim Processing Report</h1>
  </div>
  <p class="text-muted mb-4">
    {len(claims)} claim(s) &nbsp;·&nbsp;
    <span class="text-success fw-semibold">{complete} complete</span> &nbsp;·&nbsp;
    <span class="text-danger fw-semibold">{incomplete} incomplete</span> &nbsp;·&nbsp;
    <span class="text-warning fw-semibold">{review} needs_review</span>
  </p>

  {priority_section}

  <h3 class="mb-3">Claim Details</h3>
  {cards}

</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a Bootstrap HTML claim report from cache.")
    ap.add_argument("--claims-dir", default=str(ROOT / "claims"),
                    help="Path to claims directory (default: <project_root>/claims/)")
    ap.add_argument("--output", default=str(ROOT / "demo" / "claim_report.html"),
                    help="Output HTML file (default: <project_root>/demo/claim_report.html)")
    args = ap.parse_args()

    claims_dir = Path(args.claims_dir)
    if not claims_dir.is_dir():
        print(f"Error: '{claims_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    claim_dirs = sorted(p for p in claims_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    claims: list[Claim] = []
    for d in claim_dirs:
        cache = d / ".cache" / "claim_state.json"
        if cache.exists():
            claims.append(Claim.model_validate_json(cache.read_text()))
            print(f"  [cache]  {d.name}")
        else:
            print(f"  [skip]   {d.name} — no cache found (run: python main.py --all)")

    if not claims:
        print("\nNo cached claims found. Run first:  python main.py --all")
        sys.exit(1)

    records = _prioritize(claims)

    out = Path(args.output)
    out.write_text(_build_page(claims, records), encoding="utf-8")
    print(f"\nReport written → {out.resolve()}")
    print(f"Open:  xdg-open {out}  (Linux)  |  open {out}  (macOS)")


if __name__ == "__main__":
    main()
