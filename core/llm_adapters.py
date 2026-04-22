from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from core.exceptions import ParseFailedError, UnsupportedModelError


def _load_settings() -> dict:
    path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _strip_schema_keywords(schema: dict) -> dict:
    """Remove JSON Schema keywords that Gemini's response_schema does not support."""
    unsupported = {"title", "$defs", "$schema", "additionalProperties", "default"}
    if isinstance(schema, dict):
        return {
            k: _strip_schema_keywords(v)
            for k, v in schema.items()
            if k not in unsupported
        }
    if isinstance(schema, list):
        return [_strip_schema_keywords(item) for item in schema]
    return schema


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        files: list[str] | None = None,
    ) -> dict:
        """Call the VLM and return a parsed dict matching response_schema."""


class GeminiAdapter(BaseLLMClient):
    def __init__(self, model: str, temperature: float = 0.0) -> None:
        from google import genai
        from google.genai import types as genai_types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY environment variable not set")
        self._client = genai.Client(api_key=api_key)
        self._types = genai_types
        self._model_name = model
        self._temperature = temperature

    def _upload_file(self, file_path: str):
        """Upload a file and wait until it reaches ACTIVE state (max 30 s)."""
        uploaded = self._client.files.upload(path=file_path)
        deadline = time.time() + 30
        while uploaded.state.name != "ACTIVE":
            if time.time() > deadline:
                raise ParseFailedError(
                    f"Gemini file upload timed out for {file_path}"
                )
            time.sleep(2)
            uploaded = self._client.files.get(name=uploaded.name)
        return uploaded

    def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        files: list[str] | None = None,
    ) -> dict:
        contents: list[Any] = []
        if files:
            for path in files:
                contents.append(self._upload_file(path))
        contents.append(prompt)

        config = self._types.GenerateContentConfig(
            temperature=self._temperature,
            response_mime_type="application/json",
            response_schema=_strip_schema_keywords(
                response_schema.model_json_schema()
            ),
        )
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=config,
        )
        try:
            return json.loads(response.text)
        except (json.JSONDecodeError, AttributeError) as exc:
            raise ParseFailedError(
                f"Gemini returned malformed JSON: {exc}"
            ) from exc


class QwenAdapter(BaseLLMClient):
    def __init__(self, model: str, base_url: str, temperature: float = 0.0) -> None:
        from openai import OpenAI

        api_key = os.environ.get("QWEN_API_KEY", "")
        resolved_url = os.environ.get("QWEN_BASE_URL", base_url)
        self._client = OpenAI(api_key=api_key, base_url=resolved_url)
        self._model = model
        self._temperature = temperature

    def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        files: list[str] | None = None,
    ) -> dict:
        import base64
        import mimetypes

        messages: list[dict] = []
        if files:
            content_parts: list[dict] = []
            for path in files:
                mime, _ = mimetypes.guess_type(path)
                mime = mime or "application/octet-stream"
                data = base64.b64encode(Path(path).read_bytes()).decode()
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{data}"},
                    }
                )
            content_parts.append({"type": "text", "text": prompt})
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": prompt})

        schema = response_schema.model_json_schema()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema.get("title", "Response"), "schema": schema},
            },
        )
        raw = response.choices[0].message.content or ""
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ParseFailedError(
                f"Qwen returned malformed JSON: {exc}"
            ) from exc


class LLMClientFactory:
    @staticmethod
    def get_client(model_id: str) -> BaseLLMClient:
        settings = _load_settings()
        if model_id == "gemini":
            cfg = settings.get("gemini", {})
            return GeminiAdapter(
                model=cfg.get("model", "gemini-2.0-flash"),
                temperature=cfg.get("temperature", 0.0),
            )
        if model_id == "qwen":
            cfg = settings.get("qwen", {})
            return QwenAdapter(
                model=cfg.get("model", "qwen-vl-max"),
                base_url=cfg.get("base_url", ""),
                temperature=cfg.get("temperature", 0.0),
            )
        raise UnsupportedModelError(
            f"Unsupported model_id '{model_id}'. Supported: gemini, qwen"
        )
