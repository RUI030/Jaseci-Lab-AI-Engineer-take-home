from __future__ import annotations

import hashlib
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from core.exceptions import ParseFailedError
from core.llm_adapters import BaseLLMClient
from core.models import DocRecord, ExtractedField, FieldSchema


# D3: Cache the threshold at module level — no reason to re-read YAML on every PDF parse.
_PDF_TEXT_THRESHOLD: int | None = None


def _pdf_text_threshold() -> int:
    global _PDF_TEXT_THRESHOLD
    if _PDF_TEXT_THRESHOLD is None:
        path = Path(__file__).parent.parent / "config" / "settings.yaml"
        with open(path) as f:
            _PDF_TEXT_THRESHOLD = yaml.safe_load(f).get("pdf_text_threshold", 100)
    return _PDF_TEXT_THRESHOLD


# R2: Renamed _sha256 → _hash_file (callers want a content hash, not SHA-256 specifically)
def _hash_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- Internal VLM response schema ---

class _SingleFieldExtraction(BaseModel):
    field_name: str
    origin_value: str | None = None
    unified_value: str | None = None
    valid: bool
    validation_note: str | None = None
    confidence: Literal["high", "medium", "low"]
    confidence_note: str | None = None


class _ExtractionResponse(BaseModel):
    fields: list[_SingleFieldExtraction]
    doc_type: Literal[
        "police_report",
        "finance_agreement",
        "settlement_breakdown",
        "customer_reply",
        "unknown",
    ]


_DOC_ROLE = {
    "police_report": "required",
    "finance_agreement": "required",
    "settlement_breakdown": "required",
    "customer_reply": "optional",
    "unknown": "other",
}


# R3: Unified parameter name to `schemas` (was `target_fields`)
def _build_prompt(schemas: list[FieldSchema], context_note: str = "") -> str:
    field_lines = "\n".join(
        f"  - {f.field_name} ({f.data_type}): {f.description}\n"
        f"    Validation: {f.validation_rule}\n"
        f"    Unify instruction: {f.unify_instruction}"
        for f in schemas
    )
    context_block = f"\nContext: {context_note}\n" if context_note else ""
    return (
        f"You are an AI assistant extracting structured data from an insurance document.{context_block}\n"
        "Instructions:\n"
        "1. Identify the document type.\n"
        "2. Extract the following fields if present:\n"
        f"{field_lines}\n\n"
        "For each field:\n"
        "  - Set origin_value to the raw text found (null if not found).\n"
        "  - Set unified_value to the normalised value per the unify instruction (null if not found).\n"
        "  - Set valid=true only if the value satisfies the validation rule.\n"
        "  - Set confidence: high (clean PDF), medium (scanned image), low (not found / format error).\n"
        "  - Provide a confidence_note or validation_note when the value is not high or not valid.\n"
        "Return ONLY valid JSON matching the required schema."
    )


def _response_to_extracted_fields(
    response: dict,
    source_trust: Literal["document", "user_input"],
    schemas: list[FieldSchema],
) -> list[ExtractedField]:
    schema_map = {s.field_name: s for s in schemas}
    result: list[ExtractedField] = []
    for item in response.get("fields", []):
        schema = schema_map.get(item["field_name"])
        role = schema.field_role if schema else "discovered"
        data_type = schema.data_type if schema else "string"
        result.append(
            ExtractedField(
                field_name=item["field_name"],
                field_role=role,
                source_trust=source_trust,
                origin_value=item.get("origin_value"),
                unified_value=item.get("unified_value"),
                data_type=data_type,
                valid=item.get("valid", False),
                validation_note=item.get("validation_note"),
                confidence=item.get("confidence", "low"),
                confidence_note=item.get("confidence_note"),
            )
        )
    return result


# --- Base ---

class BaseDocReader(ABC):
    def call_vlm(
        self,
        prompt: str,
        files: list[str] | None,
        client: BaseLLMClient,
    ) -> dict:
        last_exc: Exception | None = None
        delays = [0, 1, 2]
        for delay in delays:
            if delay:
                time.sleep(delay)
            try:
                return client.generate(prompt, _ExtractionResponse, files=files)
            except Exception as exc:  # T1: ParseFailedError ⊂ Exception; both branches did the same thing
                last_exc = exc
        raise ParseFailedError(
            f"VLM call failed after {len(delays)} attempts: {last_exc}"
        ) from last_exc

    # T2: shared helper — eliminates copy-pasted DocRecord construction across all three readers
    def _failed_record(
        self,
        file_path: str,
        exc: Exception,
        content_hash: str | None = None,
        raw_text: str | None = None,
    ) -> DocRecord:
        return DocRecord(
            file_name=Path(file_path).name,
            doc_type="unknown",
            doc_role="other",
            source_trust="document",
            parse_status="parse_failed",
            status_reason=str(exc),
            content_hash=content_hash,
            raw_text=raw_text,
        )

    @abstractmethod  # D2: enforce subclass contract via ABC instead of pragma: no cover
    def read(
        self,
        file_path: str,
        schemas: list[FieldSchema],
        client: BaseLLMClient,
    ) -> DocRecord: ...


# --- PDF ---

class PDFReader(BaseDocReader):
    def read(
        self,
        file_path: str,
        schemas: list[FieldSchema],
        client: BaseLLMClient,
    ) -> DocRecord:
        import pdfplumber

        threshold = _pdf_text_threshold()
        status_reason: str | None = None

        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)

        content_hash = _hash_file(file_path)

        if len(text.strip()) < threshold:
            status_reason = (
                f"PDF text volume ({len(text.strip())} chars) below threshold "
                f"({threshold}); falling back to ImageReader."
            )
            reader = ImageReader()
            record = reader.read(file_path, schemas, client)
            record.status_reason = status_reason
            record.content_hash = content_hash
            return record

        prompt = _build_prompt(schemas, context_note="This is a machine-readable PDF.")
        try:
            response = self.call_vlm(prompt, files=[file_path], client=client)
            parse_status = "complete"
        except ParseFailedError as exc:
            return self._failed_record(file_path, exc, content_hash, text)

        fields = _response_to_extracted_fields(response, "document", schemas)
        doc_type = response.get("doc_type", "unknown")
        return DocRecord(
            file_name=Path(file_path).name,
            doc_type=doc_type,
            doc_role=_DOC_ROLE.get(doc_type, "other"),
            source_trust="document",
            parse_status=parse_status,
            content_hash=content_hash,
            raw_text=text,
            fields=fields,
        )


# --- Image ---

class ImageReader(BaseDocReader):
    def preprocess(self, file_path: str) -> str:
        import cv2
        import numpy as np

        img = cv2.imread(file_path)
        if img is None:
            return file_path

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) == 0:
            return file_path

        angle = cv2.minAreaRect(coords)[-1]
        # OpenCV ≥ 4.5 returns minAreaRect angles in [0, 90); map to [-45, 45)
        if angle > 45:
            angle = angle - 90

        if abs(angle) < 1.0:
            return file_path

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        suffix = Path(file_path).suffix
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_name = tmp.name
        tmp.close()  # close handle before cv2 writes to the path
        cv2.imwrite(tmp_name, rotated)
        return tmp_name

    def read(
        self,
        file_path: str,
        schemas: list[FieldSchema],
        client: BaseLLMClient,
    ) -> DocRecord:
        content_hash = _hash_file(file_path)
        processed_path = self.preprocess(file_path)

        prompt = _build_prompt(
            schemas,
            context_note="This is a scanned image — OCR may have errors. Confidence should be at most medium.",
        )
        try:
            response = self.call_vlm(prompt, files=[processed_path], client=client)
            parse_status = "complete"
        except ParseFailedError as exc:
            return self._failed_record(file_path, exc, content_hash)

        fields = _response_to_extracted_fields(response, "document", schemas)
        # cap confidence at medium for image sources
        for f in fields:
            if f.confidence == "high":
                f.confidence = "medium"
                f.confidence_note = (
                    (f.confidence_note or "") + " [Capped to medium: scanned image source]"
                ).strip()

        doc_type = response.get("doc_type", "unknown")
        return DocRecord(
            file_name=Path(file_path).name,
            doc_type=doc_type,
            doc_role=_DOC_ROLE.get(doc_type, "other"),
            source_trust="document",
            parse_status=parse_status,
            content_hash=content_hash,
            fields=fields,
        )


# --- Text ---

class TextReader(BaseDocReader):
    def read(
        self,
        file_path: str,
        schemas: list[FieldSchema],
        client: BaseLLMClient,
    ) -> DocRecord:
        content = Path(file_path).read_text(encoding="utf-8", errors="replace")
        content_hash = _hash_file(file_path)

        wrapped = f"<document>\n{content}\n</document>"
        prompt = _build_prompt(
            schemas,
            context_note="This is a plain-text customer communication wrapped in XML tags.",
        )
        full_prompt = f"{prompt}\n\nDocument content:\n{wrapped}"

        try:
            response = self.call_vlm(full_prompt, files=None, client=client)
            parse_status = "complete"
        except ParseFailedError as exc:
            return self._failed_record(file_path, exc, content_hash, content)

        fields = _response_to_extracted_fields(response, "document", schemas)
        doc_type = response.get("doc_type", "unknown")
        return DocRecord(
            file_name=Path(file_path).name,
            doc_type=doc_type,
            doc_role=_DOC_ROLE.get(doc_type, "other"),
            source_trust="document",
            parse_status=parse_status,
            content_hash=content_hash,
            raw_text=content,
            fields=fields,
        )


# D1: Module-level function replaces DocReaderFactory class (single static method is a Java-ism)
def get_doc_reader(file_path: str) -> BaseDocReader:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".pdf":
        return PDFReader()
    if suffix in {".png", ".jpg", ".jpeg"}:
        return ImageReader()
    if suffix == ".txt":
        return TextReader()
    raise ValueError(f"Unsupported file type: '{suffix}' ({file_path})")
