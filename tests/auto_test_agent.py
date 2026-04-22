import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.agent import ClaimAgent
from core.chatbot import Chatbot
from core.models import Claim, DocRecord, ExtractedField, ValidationIssue
from core.scheduler import prioritize_claims


def _make_claim(status="incomplete", reply_count=0, uploaded_at="2024-01-01T00:00:00",
                claim_id="CLM-TEST", validation_issues=None) -> Claim:
    return Claim(
        claim_id=claim_id,
        status=status,
        uploaded_at=uploaded_at,
        reply_count=reply_count,
        validation_issues=validation_issues or [],
    )


def _make_agent_with_mock_llm(monkeypatch) -> ClaimAgent:
    with patch("core.llm_adapters.LLMClientFactory.get_client", return_value=MagicMock()):
        agent = ClaimAgent(chatbot=Chatbot())
    return agent


# --- __init__ / config loading ---

def test_agent_init_loads_config(monkeypatch):
    with patch("core.llm_adapters.LLMClientFactory.get_client", return_value=MagicMock()):
        agent = ClaimAgent(chatbot=Chatbot())
    assert agent.workflow_config is not None
    assert agent.message_config is not None
    assert len(agent.field_schemas) == 4


def test_agent_init_uses_model_from_settings(monkeypatch):
    import yaml
    from pathlib import Path
    settings_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    model_id = yaml.safe_load(settings_path.read_text())["model_id"]

    with patch("core.llm_adapters.LLMClientFactory.get_client") as mock_factory:
        mock_factory.return_value = MagicMock()
        agent = ClaimAgent(chatbot=Chatbot())
    mock_factory.assert_called_once_with(model_id)


# --- process_claim ---

def test_process_claim_returns_claim_with_correct_id(monkeypatch, tmp_path):
    # Create a minimal claim folder with one text file
    claim_dir = tmp_path / "CLM-999"
    claim_dir.mkdir()
    (claim_dir / "reply.txt").write_text("Hello, my VIN is 1HGCM82633A004352")

    mock_client = MagicMock()
    mock_client.generate.return_value = {
        "doc_type": "customer_reply",
        "fields": [],
    }

    with patch("core.llm_adapters.LLMClientFactory.get_client", return_value=mock_client):
        agent = ClaimAgent(chatbot=Chatbot())

    with patch("builtins.input", return_value=""):  # skip interactive reply
        claim = agent.process_claim(str(claim_dir))

    assert claim.claim_id == "CLM-999"
    assert claim.status in ("complete", "incomplete", "needs_review")


def test_process_claim_saves_cache(monkeypatch, tmp_path):
    claim_dir = tmp_path / "CLM-CACHE"
    claim_dir.mkdir()
    (claim_dir / "note.txt").write_text("some content")

    mock_client = MagicMock()
    mock_client.generate.return_value = {"doc_type": "unknown", "fields": []}

    with patch("core.llm_adapters.LLMClientFactory.get_client", return_value=mock_client):
        agent = ClaimAgent(chatbot=Chatbot())

    with patch("builtins.input", return_value=""):
        agent.process_claim(str(claim_dir))

    cache_file = claim_dir / ".cache" / "claim_state.json"
    assert cache_file.exists()
    data = json.loads(cache_file.read_text())
    assert data["claim_id"] == "CLM-CACHE"


# --- duplicate detection ---

def test_process_claim_detects_same_filename_duplicate(monkeypatch, tmp_path):
    claim_dir = tmp_path / "CLM-DUP1"
    claim_dir.mkdir()
    # Write same filename twice — simulated by having the same name via subdirectory trick.
    # We'll patch folder.iterdir() to return two Path objects with the same name.
    (claim_dir / "police_report.pdf").write_bytes(b"%PDF file1")

    mock_client = MagicMock()
    mock_client.generate.return_value = {"doc_type": "police_report", "fields": []}

    with patch("core.llm_adapters.LLMClientFactory.get_client", return_value=mock_client):
        agent = ClaimAgent(chatbot=Chatbot())

    # Simulate two files with the same name by patching Path.iterdir
    original_file = claim_dir / "police_report.pdf"
    duplicate_file = claim_dir / "police_report.pdf"  # same object

    with patch("pdfplumber.open") as mock_plumber, patch("builtins.input", return_value=""):
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [MagicMock(extract_text=MagicMock(return_value="A" * 500))]
        mock_plumber.return_value = mock_pdf

        with patch.object(Path, "iterdir", return_value=iter([original_file, duplicate_file])):
            claim = agent.process_claim(str(claim_dir))

    dup_records = [r for r in claim.doc_table if r.doc_status == "duplicate"]
    assert len(dup_records) == 1
    assert dup_records[0].duplicate_type == "same_filename"


def test_process_claim_detects_same_content_duplicate(monkeypatch, tmp_path):
    claim_dir = tmp_path / "CLM-DUP2"
    claim_dir.mkdir()

    content = b"identical content bytes"
    (claim_dir / "settlement_breakdown.pdf").write_bytes(content)
    (claim_dir / "settlement_breakdown_v2.pdf").write_bytes(content)

    mock_client = MagicMock()
    mock_client.generate.return_value = {"doc_type": "settlement_breakdown", "fields": []}

    with patch("core.llm_adapters.LLMClientFactory.get_client", return_value=mock_client):
        agent = ClaimAgent(chatbot=Chatbot())

    with patch("pdfplumber.open") as mock_plumber, patch("builtins.input", return_value=""):
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [MagicMock(extract_text=MagicMock(return_value="A" * 500))]
        mock_plumber.return_value = mock_pdf

        claim = agent.process_claim(str(claim_dir))

    dup_records = [r for r in claim.doc_table if r.doc_status == "duplicate"]
    assert len(dup_records) == 1
    assert dup_records[0].duplicate_type == "same_content"


# --- prioritize_claims ---

def test_prioritize_claims_order(monkeypatch):
    agent = _make_agent_with_mock_llm(monkeypatch)

    claims = [
        _make_claim("needs_review", claim_id="CLM-003", uploaded_at="2024-01-03T00:00:00"),
        _make_claim("complete", claim_id="CLM-001", uploaded_at="2024-01-01T00:00:00"),
        _make_claim("incomplete", claim_id="CLM-002", uploaded_at="2024-01-02T00:00:00"),
    ]
    records = prioritize_claims(claims, agent.message_config)

    assert records[0].claim_id == "CLM-001"  # complete first
    assert records[0].priority_rank == 1
    assert records[1].claim_id == "CLM-003"  # needs_review second (staff can act)
    assert records[-1].claim_id == "CLM-002"  # incomplete last (waiting on customer)
    assert records[-1].priority_rank == 3


def test_prioritize_claims_express_flag(monkeypatch):
    agent = _make_agent_with_mock_llm(monkeypatch)

    express_claim = _make_claim(
        "incomplete", claim_id="CLM-EXPRESS",
        validation_issues=[
            ValidationIssue(issue_type="missing", description="Missing doc", resolved=False)
        ]
    )
    standard_claim = _make_claim(
        "incomplete", claim_id="CLM-STANDARD",
        uploaded_at="2023-12-31T00:00:00",  # older, but more issues
        validation_issues=[
            ValidationIssue(issue_type="inconsistency", description="VIN mismatch", resolved=False),
            ValidationIssue(issue_type="inconsistency", description="Date mismatch", resolved=False),
        ]
    )
    records = prioritize_claims([standard_claim, express_claim], agent.message_config)

    express_records = [r for r in records if r.express]
    assert len(express_records) == 1
    assert express_records[0].claim_id == "CLM-EXPRESS"
    # Express incomplete beats standard incomplete regardless of upload time
    assert express_records[0].priority_rank < next(
        r.priority_rank for r in records if r.claim_id == "CLM-STANDARD"
    )


def test_prioritize_claims_oldest_first_within_same_status(monkeypatch):
    agent = _make_agent_with_mock_llm(monkeypatch)

    claims = [
        _make_claim("needs_review", claim_id="CLM-NEW", uploaded_at="2024-06-01T00:00:00"),
        _make_claim("needs_review", claim_id="CLM-OLD", uploaded_at="2024-01-01T00:00:00"),
    ]
    records = prioritize_claims(claims, agent.message_config)
    assert records[0].claim_id == "CLM-OLD"


def test_process_claim_txt_auto_loaded_as_reply(monkeypatch, tmp_path):
    """A .txt file in the claim folder is queued as a customer reply, not sent to the VLM."""
    claim_dir = tmp_path / "CLM-TXT"
    claim_dir.mkdir()
    (claim_dir / "customer_reply.txt").write_text("My VIN is 1HGCM82633A004352")

    mock_client = MagicMock()
    # Only called during handle_reply (reply text processing), NOT for the .txt file itself
    mock_client.generate.return_value = {"doc_type": "unknown", "fields": []}

    with patch("core.llm_adapters.LLMClientFactory.get_client", return_value=mock_client):
        agent = ClaimAgent(chatbot=Chatbot())

    with patch("builtins.input", return_value=""):
        claim = agent.process_claim(str(claim_dir))

    txt_records = [r for r in claim.doc_table if r.file_name == "customer_reply.txt"]
    assert len(txt_records) == 1
    assert txt_records[0].doc_type == "customer_reply"
    # Reply was auto-consumed — conversation log should have an inbound round
    assert any(r.direction == "inbound" for r in claim.conversation_log)


def test_process_claim_txt_not_classified_by_vlm(monkeypatch, tmp_path):
    """VLM must not be called for .txt file classification — only for reply processing."""
    claim_dir = tmp_path / "CLM-TXT2"
    claim_dir.mkdir()
    (claim_dir / "note.txt").write_text("some content")

    mock_client = MagicMock()
    mock_client.generate.return_value = {"doc_type": "unknown", "fields": []}

    with patch("core.llm_adapters.LLMClientFactory.get_client", return_value=mock_client):
        agent = ClaimAgent(chatbot=Chatbot())

    with patch("builtins.input", return_value=""):
        agent.process_claim(str(claim_dir))

    # generate should only be called once: during handle_reply (reply text extraction)
    # NOT once for the .txt file itself
    assert mock_client.generate.call_count <= 1


def test_prioritize_claims_reason_populated(monkeypatch):
    agent = _make_agent_with_mock_llm(monkeypatch)
    claims = [_make_claim("complete", claim_id="CLM-001")]
    records = prioritize_claims(claims, agent.message_config)
    assert records[0].reason != ""
