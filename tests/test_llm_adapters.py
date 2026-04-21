import json
import os
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from core.exceptions import ParseFailedError, UnsupportedModelError
from core.llm_adapters import GeminiAdapter, LLMClientFactory, QwenAdapter


class _SampleSchema(BaseModel):
    doc_type: Literal["police_report", "unknown"]
    value: str


# --- LLMClientFactory ---

def test_factory_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    with patch("core.llm_adapters.GeminiAdapter.__init__", return_value=None):
        client = LLMClientFactory.get_client("gemini")
    assert isinstance(client, GeminiAdapter)


def test_factory_qwen(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "fake-key")
    with patch("core.llm_adapters.QwenAdapter.__init__", return_value=None):
        client = LLMClientFactory.get_client("qwen")
    assert isinstance(client, QwenAdapter)


def test_factory_unsupported():
    with pytest.raises(UnsupportedModelError):
        LLMClientFactory.get_client("unknown-model")


# --- GeminiAdapter ---

def _make_gemini_adapter(monkeypatch) -> GeminiAdapter:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    with patch("google.genai.Client"):
        adapter = GeminiAdapter(model="gemini-2.0-flash", temperature=0.0)
    adapter._client = MagicMock()
    adapter._types = MagicMock()
    adapter._types.GenerateContentConfig = MagicMock(return_value=MagicMock())
    return adapter


def test_gemini_generate_valid_json(monkeypatch):
    adapter = _make_gemini_adapter(monkeypatch)
    payload = {"doc_type": "police_report", "value": "abc"}

    mock_response = MagicMock()
    mock_response.text = json.dumps(payload)
    adapter._client.models.generate_content.return_value = mock_response

    result = adapter.generate("Extract fields", _SampleSchema, files=None)
    assert result == payload


def test_gemini_generate_malformed_json(monkeypatch):
    adapter = _make_gemini_adapter(monkeypatch)

    mock_response = MagicMock()
    mock_response.text = "not valid json {{{"
    adapter._client.models.generate_content.return_value = mock_response

    with pytest.raises(ParseFailedError):
        adapter.generate("Extract fields", _SampleSchema, files=None)


def test_gemini_generate_with_files(monkeypatch):
    adapter = _make_gemini_adapter(monkeypatch)
    payload = {"doc_type": "police_report", "value": "xyz"}

    mock_uploaded = MagicMock()
    mock_uploaded.state.name = "ACTIVE"
    adapter._client.files.upload.return_value = mock_uploaded

    mock_response = MagicMock()
    mock_response.text = json.dumps(payload)
    adapter._client.models.generate_content.return_value = mock_response

    result = adapter.generate("Extract fields", _SampleSchema, files=["some/path.pdf"])
    assert result["doc_type"] == "police_report"
    adapter._client.files.upload.assert_called_once_with(path="some/path.pdf")


def test_gemini_generate_no_files(monkeypatch):
    adapter = _make_gemini_adapter(monkeypatch)
    payload = {"doc_type": "unknown", "value": ""}

    mock_response = MagicMock()
    mock_response.text = json.dumps(payload)
    adapter._client.models.generate_content.return_value = mock_response

    result = adapter.generate("Extract fields", _SampleSchema)
    assert result["doc_type"] == "unknown"
    adapter._client.files.upload.assert_not_called()


# --- QwenAdapter ---

def _make_qwen_adapter(monkeypatch) -> QwenAdapter:
    monkeypatch.setenv("QWEN_API_KEY", "fake-key")
    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        adapter = QwenAdapter(model="qwen-vl-max", base_url="https://example.com")
    adapter._client = MagicMock()
    return adapter


def test_qwen_generate_valid_json(monkeypatch):
    adapter = _make_qwen_adapter(monkeypatch)
    payload = {"doc_type": "police_report", "value": "abc"}

    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(payload)
    adapter._client.chat.completions.create.return_value = MagicMock(
        choices=[mock_choice]
    )

    result = adapter.generate("Extract fields", _SampleSchema, files=None)
    assert result == payload


def test_qwen_generate_malformed_json(monkeypatch):
    adapter = _make_qwen_adapter(monkeypatch)

    mock_choice = MagicMock()
    mock_choice.message.content = "oops not json"
    adapter._client.chat.completions.create.return_value = MagicMock(
        choices=[mock_choice]
    )

    with pytest.raises(ParseFailedError):
        adapter.generate("Extract fields", _SampleSchema, files=None)
