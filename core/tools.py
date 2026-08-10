from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

_VIN_RE = re.compile(r"^[A-Z0-9]{17}$", re.IGNORECASE)

_DOC_TYPE_KEYWORDS: dict[str, str] = {
    "police_report": "police_report",
    "finance_agreement": "finance_agreement",
    "settlement_breakdown": "settlement_breakdown",
    "customer_reply": "customer_reply",
}


def validate_vin(vin: str) -> dict:
    """Check that vin matches the standard 17-character alphanumeric VIN format."""
    valid = bool(_VIN_RE.fullmatch(vin))
    return {
        "tool": "validate_vin",
        "input": {"vin": vin},
        "result": {
            "valid": valid,
            "reason": None if valid else f"'{vin}' does not match the 17-character alphanumeric VIN format",
        },
    }


def check_field_consistency(field_name: str, values: dict[str, str]) -> dict:
    """Check whether all source values for a field are identical.

    values maps source_document to unified_value,
    e.g. {"police_report.pdf": "ABC123", "finance_agreement.pdf": "XYZ789"}.
    """
    unique = set(values.values())
    consistent = len(unique) <= 1
    return {
        "tool": "check_field_consistency",
        "input": {"field_name": field_name, "values": values},
        "result": {"consistent": consistent, "unique_values": sorted(unique)},
    }


def classify_document(file_name: str, actual_type: str | None = None) -> dict:
    """Record document classification for the audit trail.

    `actual_type` is the VLM-confirmed doc_type set after reading the file.
    If omitted (pre-read call), only the filename heuristic is recorded.
    """
    lower = file_name.lower().replace("-", "_")
    inferred = next((v for k, v in _DOC_TYPE_KEYWORDS.items() if k in lower), "unknown")
    result: dict = {"inferred_doc_type": inferred}
    if actual_type is not None:
        result["actual_doc_type"] = actual_type
        result["overridden"] = actual_type != inferred
    return {
        "tool": "classify_document",
        "input": {"file_name": file_name},
        "result": result,
    }


# --- LLM-driven cross-validation dispatcher ---

@dataclass
class ValidationReport:
    """Structured result returned by the LLM-driven validation agent."""
    issues_found: list[str] = field(default_factory=list)


try:
    from byllm.lib import by, Model as _ByllmModel

    _tool_llm = _ByllmModel(
        model_name="gemini/gemini-2.5-flash",
        config={"api_key": os.environ.get("GEMINI_API_KEY", "")},
    )

    @by(_tool_llm, tools=[validate_vin, check_field_consistency])
    def run_cross_validation(fields_by_source: dict, field_schemas: list) -> ValidationReport:
        """
        Validate extracted insurance claim fields using the available tools.

        fields_by_source maps each field_name to a dict of {source_document: unified_value}.
        For every VIN field (field name contains 'vin'), call validate_vin for each unique value.
        For any field with more than one source document, call check_field_consistency
        with the field_name and a values dict of {source_document: value}.
        List each detected issue as a human-readable string in issues_found.
        """
        ...

except Exception:
    def run_cross_validation(fields_by_source: dict, field_schemas: list) -> ValidationReport:  # type: ignore[misc]
        raise RuntimeError(
            "byllm is not available. Install with: pip install jaclang jac-byllm"
        )
