from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from core.models import (
    Claim,
    DocRecord,
    ExtractedField,
    FieldSchema,
    ValidationIssue,
)
from core.parser import ClaimParser


def _make_claim(**kwargs) -> Claim:
    defaults = dict(claim_id="CLM-TEST", uploaded_at="2024-01-01T00:00:00")
    defaults.update(kwargs)
    return Claim(**defaults)


def _make_doc(
    file_name="police_report.pdf",
    doc_type="police_report",
    doc_role="required",
    parse_status="complete",
    doc_status="present",
    duplicate_type=None,
    fields=None,
) -> DocRecord:
    return DocRecord(
        file_name=file_name,
        doc_type=doc_type,
        doc_role=doc_role,
        source_trust="document",
        parse_status=parse_status,
        doc_status=doc_status,
        duplicate_type=duplicate_type,
        fields=fields or [],
    )


def _make_field(
    field_name="VIN",
    unified_value="1HGCM82633A004352",
    confidence="high",
    source_trust="document",
    valid=True,
    field_role="required",
) -> ExtractedField:
    return ExtractedField(
        field_name=field_name,
        field_role=field_role,
        source_trust=source_trust,
        unified_value=unified_value,
        data_type="string",
        valid=valid,
        confidence=confidence,
    )


parser = ClaimParser()


# --- check_required_docs ---

def test_check_required_docs_missing_police_report():
    claim = _make_claim(
        doc_table=[
            _make_doc("finance_agreement.pdf", "finance_agreement"),
            _make_doc("settlement_breakdown.pdf", "settlement_breakdown"),
        ]
    )
    missing = parser.check_required_docs(claim)
    assert missing == ["police_report"]


def test_check_required_docs_all_present():
    claim = _make_claim(
        doc_table=[
            _make_doc("police_report.pdf", "police_report"),
            _make_doc("finance_agreement.pdf", "finance_agreement"),
            _make_doc("settlement_breakdown.pdf", "settlement_breakdown"),
        ]
    )
    assert parser.check_required_docs(claim) == []


def test_check_required_docs_duplicate_not_counted():
    claim = _make_claim(
        doc_table=[
            _make_doc(
                "police_report.pdf", "police_report", doc_status="duplicate",
                duplicate_type="same_filename"
            ),
            _make_doc("finance_agreement.pdf", "finance_agreement"),
            _make_doc("settlement_breakdown.pdf", "settlement_breakdown"),
        ]
    )
    # duplicate doc has doc_status != "present", so police_report is missing
    missing = parser.check_required_docs(claim)
    assert "police_report" in missing


# --- cross_validate ---

def test_cross_validate_no_issues_when_values_match():
    vin_field = _make_field("VIN", "SAMEVIN12345678AB")
    claim = _make_claim(
        doc_table=[
            _make_doc("police_report.pdf", "police_report", fields=[vin_field]),
            _make_doc(
                "finance_agreement.pdf", "finance_agreement",
                fields=[_make_field("VIN", "SAMEVIN12345678AB")]
            ),
        ]
    )
    new_issues = parser.cross_validate(claim)
    assert new_issues == []
    assert claim.validation_issues == []


def test_cross_validate_creates_inconsistency_on_vin_mismatch():
    claim = _make_claim(
        doc_table=[
            _make_doc("police_report.pdf", "police_report",
                      fields=[_make_field("VIN", "AAA00000000000001")]),
            _make_doc("finance_agreement.pdf", "finance_agreement",
                      fields=[_make_field("VIN", "BBB00000000000002")]),
        ]
    )
    new_issues = parser.cross_validate(claim)
    assert len(new_issues) == 1
    assert new_issues[0].issue_type == "inconsistency"
    assert new_issues[0].field_name == "VIN"
    assert not new_issues[0].resolved


def test_cross_validate_ignores_user_input_fields():
    user_field = _make_field("VIN", "DIFFERENT123456789", source_trust="user_input")
    doc_field = _make_field("VIN", "DOCUMENT12345678A")
    claim = _make_claim(
        doc_table=[
            _make_doc("police_report.pdf", "police_report", fields=[doc_field]),
            _make_doc("reply.txt", "customer_reply", fields=[user_field]),
        ]
    )
    new_issues = parser.cross_validate(claim)
    assert new_issues == []


def test_cross_validate_no_duplicate_issues():
    claim = _make_claim(
        doc_table=[
            _make_doc("police_report.pdf", "police_report",
                      fields=[_make_field("VIN", "AAA00000000000001")]),
            _make_doc("finance_agreement.pdf", "finance_agreement",
                      fields=[_make_field("VIN", "BBB00000000000002")]),
        ]
    )
    parser.cross_validate(claim)
    parser.cross_validate(claim)  # second call should not add duplicate
    assert len(claim.validation_issues) == 1


# --- determine_status ---

def test_determine_status_missing_doc():
    claim = _make_claim(
        doc_table=[_make_doc("police_report.pdf", "police_report", doc_status="missing")]
    )
    assert parser.determine_status(claim) == "incomplete"


def test_determine_status_same_filename_duplicate():
    claim = _make_claim(
        doc_table=[
            _make_doc("settlement_breakdown.pdf", "settlement_breakdown",
                      doc_status="duplicate", duplicate_type="same_filename")
        ]
    )
    assert parser.determine_status(claim) == "incomplete"


def test_determine_status_same_content_duplicate():
    claim = _make_claim(
        doc_table=[
            _make_doc("settlement_breakdown.pdf", "settlement_breakdown",
                      doc_status="duplicate", duplicate_type="same_content")
        ]
    )
    assert parser.determine_status(claim) == "pending"


def test_determine_status_parse_failed():
    claim = _make_claim(
        doc_table=[
            _make_doc("police_report.pdf", "police_report", parse_status="parse_failed")
        ]
    )
    assert parser.determine_status(claim) == "pending"


def test_determine_status_unknown_doc_type():
    claim = _make_claim(
        doc_table=[
            _make_doc("unknown.png", "unknown", doc_role="other")
        ]
    )
    assert parser.determine_status(claim) == "pending"


def test_determine_status_unresolved_inconsistency_no_reply():
    claim = _make_claim(
        validation_issues=[
            ValidationIssue(
                issue_type="inconsistency",
                description="VIN mismatch",
                resolved=False,
            )
        ]
    )
    assert parser.determine_status(claim) == "incomplete"


def test_determine_status_unresolved_inconsistency_after_reply():
    claim = _make_claim(
        reply_count=1,
        validation_issues=[
            ValidationIssue(
                issue_type="inconsistency",
                description="VIN mismatch",
                resolved=False,
            )
        ]
    )
    assert parser.determine_status(claim) == "pending"


def test_determine_status_complete():
    claim = _make_claim(
        doc_table=[
            _make_doc("police_report.pdf", "police_report"),
            _make_doc("finance_agreement.pdf", "finance_agreement"),
            _make_doc("settlement_breakdown.pdf", "settlement_breakdown"),
        ],
        extracted_fields={
            "VIN": _make_field("VIN", "1HGCM82633A004352", valid=True),
            "date_of_loss": _make_field("date_of_loss", "2024-01-15", valid=True,
                                        field_role="required"),
            "insurance_payout": _make_field("insurance_payout", "24500.00",
                                            valid=True, field_role="required"),
        },
    )
    assert parser.determine_status(claim) == "complete"


# --- compare_fields ---

def test_compare_fields_consistent():
    claim = _make_claim(
        extracted_fields={"VIN": _make_field("VIN", "1HGCM82633A004352")}
    )
    new = [_make_field("VIN", "1HGCM82633A004352", source_trust="user_input")]
    results = parser.compare_fields(claim, new)
    assert results["VIN"] == "consistent"


def test_compare_fields_inconsistent():
    claim = _make_claim(
        extracted_fields={"VIN": _make_field("VIN", "1HGCM82633A004352")}
    )
    new = [_make_field("VIN", "DIFFERENT00000000A", source_trust="user_input")]
    results = parser.compare_fields(claim, new)
    assert results["VIN"] == "inconsistent"


def test_compare_fields_new_field_treated_as_consistent():
    claim = _make_claim()
    new = [_make_field("date_of_loss", "2024-01-15", source_trust="user_input",
                       field_role="required")]
    results = parser.compare_fields(claim, new)
    assert results["date_of_loss"] == "consistent"


# --- record_reply ---

def test_record_reply_appends_round_and_increments_count():
    claim = _make_claim()
    parser.record_reply(claim, "My VIN is 1HGCM82633A004352", {"VIN": "consistent"})
    assert claim.reply_count == 1
    assert len(claim.conversation_log) == 1
    round_ = claim.conversation_log[0]
    assert round_.direction == "inbound"
    assert round_.compare_results == {"VIN": "consistent"}
    assert round_.round == 1


def test_record_reply_multiple_rounds():
    claim = _make_claim()
    parser.record_reply(claim, "First reply", {})
    parser.record_reply(claim, "Second reply", {"date_of_loss": "inconsistent"})
    assert claim.reply_count == 2
    assert claim.conversation_log[1].round == 2


# --- handle_reply (integration with mocked TextReader) ---

def test_handle_reply_consistent_reply_keeps_incomplete():
    claim = _make_claim(
        status="incomplete",
        doc_table=[
            _make_doc("police_report.pdf", "police_report", doc_status="missing")
        ],
        extracted_fields={"VIN": _make_field("VIN", "1HGCM82633A004352")},
    )
    mock_client = MagicMock()
    with patch("core.parser.TextReader") as MockTextReader:
        mock_record = MagicMock()
        mock_record.fields = [
            _make_field("VIN", "1HGCM82633A004352", source_trust="document")
        ]
        MockTextReader.return_value.read.return_value = mock_record

        result = parser.handle_reply(claim, "My VIN is 1HGCM82633A004352", mock_client, [])

    assert result.reply_count == 1
    assert len(result.conversation_log) == 1
    # still incomplete because police_report is missing
    assert result.status == "incomplete"


def test_handle_reply_inconsistent_adds_validation_issue():
    claim = _make_claim(
        status="incomplete",
        doc_table=[
            _make_doc("police_report.pdf", "police_report"),
            _make_doc("finance_agreement.pdf", "finance_agreement"),
            _make_doc("settlement_breakdown.pdf", "settlement_breakdown"),
        ],
        extracted_fields={"VIN": _make_field("VIN", "1HGCM82633A004352")},
    )
    mock_client = MagicMock()
    with patch("core.parser.TextReader") as MockTextReader:
        mock_record = MagicMock()
        mock_record.fields = [
            _make_field("VIN", "DIFFERENTVIN000000", source_trust="document")
        ]
        MockTextReader.return_value.read.return_value = mock_record

        result = parser.handle_reply(claim, "My VIN is DIFFERENTVIN000000", mock_client, [])

    inconsistencies = [
        vi for vi in result.validation_issues
        if vi.issue_type == "inconsistency" and vi.field_name == "VIN"
    ]
    assert len(inconsistencies) == 1
    assert result.status == "pending"  # reply_count > 0 + unresolved inconsistency
