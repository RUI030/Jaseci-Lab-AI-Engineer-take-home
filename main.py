"""
AI Insurance Claims Processing Agent — CLI entry point.

Usage:
    python main.py --claim claims/CLM-001        # process one claim (interactive)
    python main.py --all                          # process all 5 claims + prioritization
    python main.py --batch claims/               # same as --all but with custom directory
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from core.agent import ClaimAgent
from core.chatbot import Chatbot

load_dotenv()


def _print_claim_summary(claim) -> None:
    print(f"\n{'='*60}")
    print(f"  {claim.claim_id}  |  status: {claim.status.upper()}")
    print(f"{'='*60}")

    # Documents
    print("\nDocuments:")
    for rec in claim.doc_table:
        status_tag = f"[{rec.doc_status}]" if rec.doc_status != "present" else ""
        parse_tag = f"[{rec.parse_status}]" if rec.parse_status != "complete" else ""
        tags = " ".join(filter(None, [status_tag, parse_tag]))
        print(f"  {rec.file_name:<40} {rec.doc_type:<25} {tags}")

    # Extracted fields
    if claim.extracted_fields:
        print("\nExtracted Fields:")
        for name, field in claim.extracted_fields.items():
            val = field.unified_value or "(not found)"
            conf = field.confidence
            valid_tag = "✓" if field.valid else "✗"
            print(f"  {name:<30} {val:<30} conf={conf} {valid_tag}")

    # Issues
    if claim.validation_issues:
        print("\nValidation Issues:")
        for vi in claim.validation_issues:
            resolved_tag = "[resolved]" if vi.resolved else ""
            print(f"  [{vi.issue_type}] {vi.description} {resolved_tag}")

    # Conversation log
    if claim.conversation_log:
        print(f"\nConversation ({len(claim.conversation_log)} rounds):")
        for cr in claim.conversation_log:
            direction_tag = "→ OUT" if cr.direction == "outbound" else "← IN"
            preview = cr.message[:80].replace("\n", " ")
            suffix = "..." if len(cr.message) > 80 else ""
            print(f"  Round {cr.round} {direction_tag}: {preview}{suffix}")

    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AI Insurance Claims Processing Agent"
    )
    parser.add_argument(
        "--claim", metavar="PATH", help="Process a single claim folder (e.g. claims/CLM-001)"
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all claims in --batch directory"
    )
    parser.add_argument(
        "--batch", default="claims/", metavar="DIR", help="Claims directory (default: claims/)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output raw JSON instead of summary"
    )
    args = parser.parse_args()

    if not args.claim and not args.all:
        parser.print_help()
        return 1

    chatbot = Chatbot()
    agent = ClaimAgent(chatbot=chatbot)

    if args.claim:
        claim = agent.process_claim(args.claim)
        if args.json:
            print(claim.model_dump_json(indent=2))
        else:
            _print_claim_summary(claim)
        return 0

    # --all: process every folder in --batch
    batch_dir = Path(args.batch)
    if not batch_dir.is_dir():
        print(f"Error: '{batch_dir}' is not a directory", file=sys.stderr)
        return 1

    claim_dirs = sorted(
        p for p in batch_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not claim_dirs:
        print(f"No claim folders found in '{batch_dir}'", file=sys.stderr)
        return 1

    claims = []
    for claim_dir in claim_dirs:
        chatbot.display(f"\nProcessing {claim_dir.name}...")
        try:
            claim = agent.process_claim(str(claim_dir))
            claims.append(claim)
            if args.json:
                print(claim.model_dump_json(indent=2))
            else:
                _print_claim_summary(claim)
        except Exception as exc:
            print(f"  ERROR processing {claim_dir.name}: {exc}", file=sys.stderr)

    if not claims:
        return 1

    # Prioritization summary
    records = agent.prioritize_claims(claims)
    print("\n" + "="*60)
    print("  PRIORITY ORDER")
    print("="*60)
    for rec in records:
        express_tag = " [EXPRESS]" if rec.express else ""
        print(
            f"  {rec.priority_rank}. {rec.claim_id:<12} [{rec.status}]{express_tag}"
        )
        print(f"     {rec.reason}")

    if args.json:
        print(
            json.dumps(
                {"processing_order": [r.model_dump() for r in records]}, indent=2
            )
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
