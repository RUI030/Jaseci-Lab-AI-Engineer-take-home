from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from core.exceptions import ParseFailedError, UnsupportedModelError
from core.utils import load_yaml


def _load_settings() -> dict:
    return load_yaml("config/settings.yaml")


def _retry_config() -> tuple[int, float, float]:
    """Return (max_attempts, base_delay, max_delay) from settings."""
    cfg = _load_settings().get("retry", {})
    return (
        int(cfg.get("max_attempts", 3)),
        float(cfg.get("base_delay_seconds", 2.0)),
        float(cfg.get("max_delay_seconds", 30.0)),
    )


def _with_retry(fn, *args, **kwargs):
    """Call fn with exponential backoff on transient errors.

    Retries on: rate-limit (429), server errors (5xx), and provider-specific
    ResourceExhausted / ServiceUnavailable exceptions. All other errors surface
    immediately. After exhausting retries, raises ParseFailedError.
    """
    max_attempts, base_delay, max_delay = _retry_config()
    delay = base_delay
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except ParseFailedError:
            raise
        except Exception as exc:
            err_str = str(exc).lower()
            transient = (
                "429" in err_str
                or "quota" in err_str
                or "resource_exhausted" in err_str
                or "resourceexhausted" in err_str
                or "service_unavailable" in err_str
                or "serviceunavailable" in err_str
                or "500" in err_str
                or "503" in err_str
                or "too many requests" in err_str
            )
            if not transient:
                raise
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(min(delay, max_delay))
                delay *= 2

    raise ParseFailedError(
        f"API call failed after {max_attempts} attempts: {last_exc}"
    ) from last_exc


def _prepare_gemini_schema(schema: dict) -> dict:
    """Resolve $ref references inline and strip keywords Gemini does not support.

    Gemini's response_schema rejects $ref / $defs and a few JSON Schema keywords.
    This function inlines every $ref so the schema is fully self-contained, then
    strips the unsupported keys in one pass.
    """
    _UNSUPPORTED = {"title", "$schema", "additionalProperties", "default"}
    defs: dict = schema.get("$defs", {})

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                return _resolve(defs.get(ref_name, {}))
            return {
                k: _resolve(v)
                for k, v in node.items()
                if k not in _UNSUPPORTED and k != "$defs"
            }
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(schema)


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        files: list[str] | None = None,
    ) -> dict:
        """Call the VLM and return a parsed dict matching response_schema."""

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """Generate a plain-text response without JSON schema constraints."""


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
        uploaded = self._client.files.upload(file=file_path)
        deadline = time.time() + 30
        while uploaded.state.name != "ACTIVE":
            if time.time() > deadline:
                raise ParseFailedError(
                    f"Gemini file upload timed out for {file_path}"
                )
            time.sleep(2)
            uploaded = self._client.files.get(name=uploaded.name)
        return uploaded

    def _call_api(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        files: list[str] | None,
    ) -> dict:
        contents: list[Any] = []
        if files:
            for path in files:
                contents.append(self._upload_file(path))
        contents.append(prompt)

        config = self._types.GenerateContentConfig(
            temperature=self._temperature,
            response_mime_type="application/json",
            response_schema=_prepare_gemini_schema(
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

    def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        files: list[str] | None = None,
    ) -> dict:
        return _with_retry(self._call_api, prompt, response_schema, files)

    def generate_text(self, prompt: str) -> str:
        def _call():
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[prompt],
                config=self._types.GenerateContentConfig(temperature=0.4),
            )
            return (response.text or "").strip()
        return _with_retry(_call)


class QwenAdapter(BaseLLMClient):
    def __init__(self, model: str, base_url: str, temperature: float = 0.0) -> None:
        from openai import OpenAI

        api_key = os.environ.get("QWEN_API_KEY", "")
        resolved_url = os.environ.get("QWEN_BASE_URL", base_url)
        self._client = OpenAI(api_key=api_key, base_url=resolved_url)
        self._model = model
        self._temperature = temperature

    def _call_api(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        files: list[str] | None,
    ) -> dict:
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

    def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        files: list[str] | None = None,
    ) -> dict:
        return _with_retry(self._call_api, prompt, response_schema, files)

    def generate_text(self, prompt: str) -> str:
        def _call():
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            return (response.choices[0].message.content or "").strip()
        return _with_retry(_call)


class QwenLocalAdapter(BaseLLMClient):
    """Runs a Qwen VL model locally via Hugging Face transformers.

    Supports any Qwen vision-language model family (Qwen2-VL, Qwen2.5-VL, Qwen3-VL, …).
    Set the HuggingFace Hub model ID in config/settings.yaml under qwen_local.model.
    """

    def __init__(self, model: str, device: str = "auto", temperature: float = 0.0) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import torch

        self._model_name = model
        self._temperature = temperature
        self._torch = torch

        self._processor = AutoProcessor.from_pretrained(model)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
        )

    def _build_messages(self, prompt: str, files: list[str] | None) -> list[dict]:
        from PIL import Image

        content: list[dict] = []
        if files:
            for path in files:
                ext = Path(path).suffix.lower()
                if ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    content.append({"type": "image", "image": Image.open(path).convert("RGB")})
                elif ext == ".pdf":
                    try:
                        import pdfplumber
                        with pdfplumber.open(path) as pdf:
                            for page in pdf.pages:
                                img = page.to_image(resolution=150).original
                                content.append({"type": "image", "image": img.convert("RGB")})
                    except Exception as exc:
                        raise ParseFailedError(
                            f"Failed to render PDF pages for local model: {exc}"
                        ) from exc
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def generate(
        self,
        prompt: str,
        response_schema: type[BaseModel],
        files: list[str] | None = None,
    ) -> dict:
        from qwen_vl_utils import process_vision_info

        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        full_prompt = (
            f"{prompt}\n\n"
            f"You MUST respond with valid JSON only — no markdown, no explanation.\n"
            f"The JSON must exactly match this schema:\n{schema_json}"
        )

        messages = self._build_messages(full_prompt, files)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        do_sample = self._temperature > 0.0
        gen_kwargs: dict = {"max_new_tokens": 1024, "do_sample": do_sample}
        if do_sample:
            gen_kwargs["temperature"] = self._temperature
        output_ids = self._model.generate(**inputs, **gen_kwargs)
        # Strip input tokens from output
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]
        raw = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # Strip markdown code fences if present (handles ```json, ```jsonc, etc.)
        if raw.startswith("```"):
            lines = raw.splitlines()
            inner = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
            raw = "\n".join(inner).strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ParseFailedError(
                f"QwenLocal returned malformed JSON: {exc}\nRaw output: {raw[:300]}"
            ) from exc

    def generate_text(self, prompt: str) -> str:
        from qwen_vl_utils import process_vision_info

        messages = self._build_messages(prompt, None)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        gen_kwargs: dict = {"max_new_tokens": 512, "do_sample": True, "temperature": 0.4}
        output_ids = self._model.generate(**inputs, **gen_kwargs)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


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
        if model_id == "qwen_local":
            cfg = settings.get("qwen_local", {})
            return QwenLocalAdapter(
                model=cfg.get("model", "Qwen/Qwen2.5-VL-3B-Instruct"),
                device=cfg.get("device", "auto"),
                temperature=cfg.get("temperature", 0.0),
            )
        raise UnsupportedModelError(
            f"Unsupported model_id '{model_id}'. Supported: gemini, qwen, qwen_local"
        )
