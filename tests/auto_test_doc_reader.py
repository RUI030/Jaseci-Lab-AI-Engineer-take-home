import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.doc_reader import (
    get_doc_reader,
    ImageReader,
    PDFReader,
    TextReader,
    BaseDocReader,
)
from core.models import DocRecord


# Minimal concrete subclass so we can test call_vlm without triggering ABC enforcement
class _ConcreteReader(BaseDocReader):
    def read(self, file_path, schemas, client):  # pragma: no cover
        return DocRecord(
            file_name="stub.txt",
            doc_type="unknown",
            doc_role="other",
            source_trust="document",
            parse_status="unprocessed",
        )
from core.exceptions import ParseFailedError
from core.models import FieldSchema


FIELD_SCHEMAS = [
    FieldSchema(
        field_name="VIN",
        data_type="string",
        field_role="required",
        validation_rule="Exactly 17 alphanumeric characters",
        unify_instruction="Uppercase, remove whitespace",
        description="Vehicle Identification Number",
    ),
    FieldSchema(
        field_name="date_of_loss",
        data_type="date",
        field_role="required",
        validation_rule="Valid date YYYY-MM-DD",
        unify_instruction="Normalize to ISO 8601",
        description="Date of the loss event",
    ),
]

MOCK_RESPONSE = {
    "doc_type": "police_report",
    "fields": [
        {
            "field_name": "VIN",
            "origin_value": "1HGCM82633A004352",
            "unified_value": "1HGCM82633A004352",
            "valid": True,
            "validation_note": None,
            "confidence": "high",
            "confidence_note": None,
        }
    ],
}


# --- get_doc_reader routing ---

@pytest.mark.parametrize(
    "path,expected",
    [
        ("report.pdf", PDFReader),
        ("photo.png", ImageReader),
        ("scan.jpg", ImageReader),
        ("scan.jpeg", ImageReader),
        ("email.txt", TextReader),
    ],
)
def test_factory_routing(path, expected):
    reader = get_doc_reader(path)
    assert isinstance(reader, expected)


def test_factory_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported file type"):
        get_doc_reader("data.csv")


# --- BaseDocReader.call_vlm retry logic ---

def test_call_vlm_succeeds_on_third_attempt():
    reader = _ConcreteReader()
    mock_client = MagicMock()
    mock_client.generate.side_effect = [
        ParseFailedError("fail 1"),
        ParseFailedError("fail 2"),
        {"doc_type": "police_report", "fields": []},
    ]
    with patch("time.sleep"):
        result = reader.call_vlm("prompt", None, mock_client)
    assert result["doc_type"] == "police_report"
    assert mock_client.generate.call_count == 3


def test_call_vlm_raises_after_all_failures():
    reader = _ConcreteReader()
    mock_client = MagicMock()
    mock_client.generate.side_effect = ParseFailedError("always fails")
    with patch("time.sleep"):
        with pytest.raises(ParseFailedError):
            reader.call_vlm("prompt", None, mock_client)
    assert mock_client.generate.call_count == 3


def test_call_vlm_succeeds_first_attempt():
    reader = _ConcreteReader()
    mock_client = MagicMock()
    mock_client.generate.return_value = {"doc_type": "unknown", "fields": []}
    result = reader.call_vlm("prompt", None, mock_client)
    assert result == {"doc_type": "unknown", "fields": []}
    mock_client.generate.assert_called_once()


# --- TextReader ---

def test_text_reader_returns_doc_record(tmp_path):
    txt_file = tmp_path / "reply.txt"
    txt_file.write_text("My VIN is 1HGCM82633A004352 and the date was 2024-03-15.")

    mock_client = MagicMock()
    mock_client.generate.return_value = MOCK_RESPONSE

    record = TextReader().read(str(txt_file), FIELD_SCHEMAS, mock_client)

    assert record.file_name == "reply.txt"
    assert record.parse_status == "complete"
    assert record.doc_type == "police_report"
    assert len(record.fields) == 1
    assert record.fields[0].field_name == "VIN"
    assert record.raw_text is not None


def test_text_reader_parse_failed(tmp_path):
    txt_file = tmp_path / "bad.txt"
    txt_file.write_text("some content")

    mock_client = MagicMock()
    mock_client.generate.side_effect = ParseFailedError("VLM broken")

    with patch("time.sleep"):
        record = TextReader().read(str(txt_file), FIELD_SCHEMAS, mock_client)

    assert record.parse_status == "parse_failed"
    assert "VLM broken" in (record.status_reason or "")


def test_text_reader_wraps_xml(tmp_path):
    txt_file = tmp_path / "msg.txt"
    txt_file.write_text("Hello, my loan balance is $5000.")

    mock_client = MagicMock()
    mock_client.generate.return_value = {"doc_type": "customer_reply", "fields": []}

    TextReader().read(str(txt_file), FIELD_SCHEMAS, mock_client)

    call_args = mock_client.generate.call_args
    prompt_arg = call_args[0][0]
    assert "<document>" in prompt_arg
    assert "</document>" in prompt_arg


# --- PDFReader fallback to ImageReader ---

def test_pdf_reader_falls_back_when_text_short(tmp_path):
    pdf_file = tmp_path / "scanned.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 minimal")  # not a real PDF, just for path

    mock_client = MagicMock()
    mock_client.generate.return_value = MOCK_RESPONSE

    with patch("pdfplumber.open") as mock_plumber, patch.object(
        ImageReader, "read", return_value=MagicMock(
            file_name="scanned.pdf",
            parse_status="complete",
            doc_type="police_report",
            doc_role="required",
            source_trust="document",
            content_hash="abc",
            fields=[],
            status_reason=None,
        )
    ) as mock_image_read:
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_pdf.pages = [MagicMock(extract_text=MagicMock(return_value="Hi"))]
        mock_plumber.return_value = mock_pdf

        record = PDFReader().read(str(pdf_file), FIELD_SCHEMAS, mock_client)

    mock_image_read.assert_called_once()
    assert "falling back to ImageReader" in (record.status_reason or "")


def test_pdf_reader_parse_failed(tmp_path):
    pdf_file = tmp_path / "report.pdf"
    pdf_file.write_bytes(b"%PDF fake")

    mock_client = MagicMock()
    mock_client.generate.side_effect = ParseFailedError("broken")

    with patch("pdfplumber.open") as mock_plumber, patch("time.sleep"):
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)
        long_text = "A" * 500
        mock_pdf.pages = [MagicMock(extract_text=MagicMock(return_value=long_text))]
        mock_plumber.return_value = mock_pdf

        record = PDFReader().read(str(pdf_file), FIELD_SCHEMAS, mock_client)

    assert record.parse_status == "parse_failed"


# --- ImageReader confidence cap ---

def test_image_reader_caps_confidence_to_medium(tmp_path):
    img_file = tmp_path / "photo.png"
    img_file.write_bytes(b"fake png")

    mock_client = MagicMock()
    mock_client.generate.return_value = {
        "doc_type": "police_report",
        "fields": [
            {
                "field_name": "VIN",
                "origin_value": "1HGCM82633A004352",
                "unified_value": "1HGCM82633A004352",
                "valid": True,
                "validation_note": None,
                "confidence": "high",
                "confidence_note": None,
            }
        ],
    }

    with patch.object(ImageReader, "preprocess", return_value=str(img_file)):
        record = ImageReader().read(str(img_file), FIELD_SCHEMAS, mock_client)

    assert record.fields[0].confidence == "medium"
    assert "Capped to medium" in (record.fields[0].confidence_note or "")
