import json
from pathlib import Path

import pytest
from core.models import (
    Claim,
    ConversationRound,
    DocRecord,
    ExtractedField,
    FieldSchema,
    PriorityRecord,
    ValidationIssue,
)


# --- FieldSchema ---

def test_field_schema_from_json():
    data = json.loads(Path("config/field_schema.json").read_text())
    schemas = [FieldSchema(**item) for item in data]
    assert len(schemas) == 4
    names = {s.field_name for s in schemas}
    assert names == {"VIN", "date_of_loss", "insurance_payout", "outstanding_loan_balance"}


def test_field_schema_roles():
    data = json.loads(Path("config/field_schema.json").read_text())
    schemas = {s["field_name"]: FieldSchema(**s) for s in data}
    assert schemas["VIN"].field_role == "required"
    assert schemas["outstanding_loan_balance"].field_role == "optional"


# --- ExtractedField ---

def test_extracted_field_document_trust():
    f = ExtractedField(
        field_name="VIN",
        field_role="required",
        source_trust="document",
        origin_value="1HGCM82633A004352",
        unified_value="1HGCM82633A004352",
        data_type="string",
        valid=True,
        confidence="high",
    )
    assert f.source_trust == "document"
    assert f.confidence == "high"
    assert f.confidence_note is None


def test_extracted_field_user_input():
    f = ExtractedField(
        field_name="VIN",
        field_role="required",
        source_trust="user_input",
        unified_value="1HGCM82633A004352",
        data_type="string",
        valid=True,
        confidence="low",
        confidence_note="Source is customer reply — hard cap low",
    )
    assert f.source_trust == "user_input"
    assert f.confidence == "low"


def test_extracted_field_not_found():
    f = ExtractedField(
        field_name="VIN",
        field_role="required",
        source_trust="document",
        origin_value=None,
        unified_value=None,
        data_type="string",
        valid=False,
        validation_note="Field not found in document",
        confidence="low",
    )
    assert f.unified_value is None
    assert f.valid is False


# --- DocRecord ---

@pytest.mark.parametrize("parse_status", ["complete", "parse_failed", "unprocessed"])
def test_doc_record_parse_status(parse_status):
    dr = DocRecord(
        file_name="test.pdf",
        doc_type="police_report",
        doc_role="required",
        source_trust="document",
        parse_status=parse_status,
    )
    assert dr.parse_status == parse_status


@pytest.mark.parametrize("doc_status", ["present", "missing", "duplicate"])
def test_doc_record_doc_status(doc_status):
    dr = DocRecord(
        file_name="test.pdf",
        doc_type="police_report",
        doc_role="required",
        source_trust="document",
        doc_status=doc_status,
    )
    assert dr.doc_status == doc_status


def test_doc_record_defaults():
    dr = DocRecord(
        file_name="report.pdf",
        doc_type="police_report",
        doc_role="required",
        source_trust="document",
    )
    assert dr.parse_status == "unprocessed"
    assert dr.doc_status == "present"
    assert dr.fields == []
    assert dr.duplicate_type is None


def test_doc_record_duplicate_types():
    for dt in ("same_filename", "same_content"):
        dr = DocRecord(
            file_name="f.pdf",
            doc_type="settlement_breakdown",
            doc_role="required",
            source_trust="document",
            doc_status="duplicate",
            duplicate_type=dt,
        )
        assert dr.duplicate_type == dt


# --- ValidationIssue ---

def test_validation_issue_defaults():
    vi = ValidationIssue(
        issue_type="inconsistency",
        description="VIN mismatch",
    )
    assert vi.resolved is False
    assert vi.resolved_by is None
    assert vi.sources == []
    assert vi.values == {}


def test_validation_issue_with_values():
    vi = ValidationIssue(
        issue_type="inconsistency",
        field_name="VIN",
        description="VIN differs between police report and finance agreement",
        sources=["police_report.pdf", "finance_agreement.pdf"],
        values={"police_report.pdf": "ABC123", "finance_agreement.pdf": "XYZ999"},
    )
    assert len(vi.sources) == 2
    assert vi.values["police_report.pdf"] == "ABC123"


# --- ConversationRound ---

def test_conversation_round_outbound():
    cr = ConversationRound(
        round=1,
        timestamp="2024-01-01T00:00:00",
        direction="outbound",
        message="Please provide your VIN.",
    )
    assert cr.direction == "outbound"
    assert cr.compare_results is None


def test_conversation_round_inbound():
    cr = ConversationRound(
        round=2,
        timestamp="2024-01-02T00:00:00",
        direction="inbound",
        message="My VIN is 1HGCM82633A004352",
        compare_results={"VIN": "consistent"},
    )
    assert cr.direction == "inbound"
    assert cr.compare_results == {"VIN": "consistent"}


# --- Claim ---

def test_claim_defaults():
    c = Claim(claim_id="CLM-001", uploaded_at="2024-01-01T00:00:00")
    assert c.status == "needs_review"
    assert c.reply_count == 0
    assert c.doc_table == []
    assert c.extracted_fields == {}
    assert c.validation_issues == []
    assert c.conversation_log == []


def test_claim_round_trip():
    c = Claim(
        claim_id="CLM-001",
        status="incomplete",
        uploaded_at="2024-01-01T00:00:00",
        doc_table=[
            DocRecord(
                file_name="police_report.pdf",
                doc_type="police_report",
                doc_role="required",
                source_trust="document",
                parse_status="complete",
            )
        ],
    )
    restored = Claim.model_validate(c.model_dump())
    assert restored == c
    assert restored.doc_table[0].file_name == "police_report.pdf"


# --- PriorityRecord ---

def test_priority_record_express():
    pr = PriorityRecord(
        claim_id="CLM-002",
        status="incomplete",
        uploaded_at="2024-01-01T00:00:00",
        express=True,
        priority_rank=1,
        reason="Express routing — fewer than 2 unresolved issues",
    )
    assert pr.express is True


def test_priority_record_no_express():
    pr = PriorityRecord(
        claim_id="CLM-003",
        status="needs_review",
        uploaded_at="2024-01-01T00:00:00",
        express=False,
        priority_rank=5,
        reason="Requires human review",
    )
    assert pr.express is False
    assert pr.priority_rank == 5
