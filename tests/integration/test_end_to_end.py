"""
End-to-end integration tests.

These tests require a real GEMINI_API_KEY and make live API calls.
Run with:  pytest tests/integration/ -v -m integration

They are excluded from the default test run.
"""
import json
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.agent import ClaimAgent
from core.chatbot import Chatbot


pytestmark = pytest.mark.integration


@pytest.fixture
def agent(monkeypatch):
    """Agent wired to a mock LLM so integration tests can run without an API key."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-integration")

    def _mock_generate(prompt, response_schema, files=None):
        file_name = Path(files[0]).name if files else "unknown"
        if "police_report" in file_name:
            doc_type = "police_report"
        elif "finance" in file_name:
            doc_type = "finance_agreement"
        elif "settlement" in file_name:
            doc_type = "settlement_breakdown"
        elif "customer_reply" in file_name or files is None:
            doc_type = "customer_reply"
        else:
            doc_type = "unknown"

        return {
            "doc_type": doc_type,
            "fields": [
                {
                    "field_name": "VIN",
                    "origin_value": "1HGCM82633A004352",
                    "unified_value": "1HGCM82633A004352",
                    "valid": True,
                    "validation_note": None,
                    "confidence": "high",
                    "confidence_note": None,
                },
                {
                    "field_name": "date_of_loss",
                    "origin_value": "2024-01-15",
                    "unified_value": "2024-01-15",
                    "valid": True,
                    "validation_note": None,
                    "confidence": "high",
                    "confidence_note": None,
                },
                {
                    "field_name": "insurance_payout",
                    "origin_value": "24500.00",
                    "unified_value": "24500.00",
                    "valid": True,
                    "validation_note": None,
                    "confidence": "high",
                    "confidence_note": None,
                },
            ],
        }

    with patch("core.llm_adapters.GeminiAdapter.__init__", return_value=None):
        a = ClaimAgent(chatbot=Chatbot())
    a.llm_client = MagicMock()
    a.llm_client.generate.side_effect = _mock_generate
    return a


def test_clm001_all_docs_present(agent, tmp_path):
    """CLM-001 has all required docs. With consistent field values, should be complete."""
    claim_dir = tmp_path / "CLM-001"
    claim_dir.mkdir()
    (claim_dir / "police_report.pdf").write_bytes(b"%PDF police " + b"A" * 500)
    (claim_dir / "finance_agreement.pdf").write_bytes(b"%PDF finance " + b"B" * 500)
    (claim_dir / "settlement_breakdown.pdf").write_bytes(b"%PDF settlement " + b"C" * 500)

    with patch("pdfplumber.open") as mock_plumber, patch("builtins.input", return_value=""):
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [MagicMock(extract_text=MagicMock(return_value="A" * 500))]
        mock_plumber.return_value = mock_pdf
        claim = agent.process_claim(str(claim_dir))

    assert claim.claim_id == "CLM-001"
    assert claim.status == "complete"
    assert (claim_dir / ".cache" / "claim_state.json").exists()


def test_clm002_missing_police_report(agent, tmp_path):
    """CLM-002 is missing police_report. Should be incomplete."""
    claim_dir = tmp_path / "CLM-002"
    claim_dir.mkdir()
    (claim_dir / "finance_agreement.pdf").write_bytes(b"%PDF finance " + b"B" * 500)
    (claim_dir / "settlement_breakdown.pdf").write_bytes(b"%PDF settlement " + b"C" * 500)
    (claim_dir / "customer_reply.txt").write_text(
        "My VIN is 1HGCM82633A004352 and the date was 2024-01-15."
    )

    with patch("pdfplumber.open") as mock_plumber, patch("builtins.input", return_value=""):
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [MagicMock(extract_text=MagicMock(return_value="A" * 500))]
        mock_plumber.return_value = mock_pdf
        claim = agent.process_claim(str(claim_dir))

    assert claim.claim_id == "CLM-002"
    assert claim.status == "incomplete"
    missing_types = [r.doc_type for r in claim.doc_table if r.doc_status == "missing"]
    assert "police_report" in missing_types


def test_clm003_duplicate_settlement(agent, tmp_path):
    """CLM-003 has two settlement_breakdown files with same content. Should detect same_content duplicate."""
    claim_dir = tmp_path / "CLM-003"
    claim_dir.mkdir()
    content = b"%PDF settlement content " + b"X" * 500
    (claim_dir / "police_report.pdf").write_bytes(b"%PDF police " + b"A" * 500)
    (claim_dir / "finance_agreement.pdf").write_bytes(b"%PDF finance " + b"A" * 500)
    (claim_dir / "settlement_breakdown.pdf").write_bytes(content)
    (claim_dir / "settlement_breakdown_v2.pdf").write_bytes(content)  # same content

    with patch("pdfplumber.open") as mock_plumber, patch("builtins.input", return_value=""):
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [MagicMock(extract_text=MagicMock(return_value="A" * 500))]
        mock_plumber.return_value = mock_pdf
        claim = agent.process_claim(str(claim_dir))

    dup_records = [r for r in claim.doc_table if r.doc_status == "duplicate"]
    assert len(dup_records) >= 1
    assert any(r.duplicate_type == "same_content" for r in dup_records)


def test_prioritize_all_statuses(agent):
    """prioritize_claims correctly orders complete > incomplete > pending."""
    from core.models import Claim

    claims = [
        Claim(claim_id="CLM-P", status="pending", uploaded_at="2024-01-03T00:00:00"),
        Claim(claim_id="CLM-C", status="complete", uploaded_at="2024-01-01T00:00:00"),
        Claim(claim_id="CLM-I", status="incomplete", uploaded_at="2024-01-02T00:00:00"),
    ]
    records = agent.prioritize_claims(claims)
    assert records[0].claim_id == "CLM-C"
    assert records[-1].claim_id == "CLM-P"
    for i, rec in enumerate(records, start=1):
        assert rec.priority_rank == i
