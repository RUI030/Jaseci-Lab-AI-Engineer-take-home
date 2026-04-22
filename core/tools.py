from __future__ import annotations

import re

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
    """Check whether all source values for a field are identical."""
    unique = set(values.values())
    consistent = len(unique) <= 1
    return {
        "tool": "check_field_consistency",
        "input": {"field_name": field_name, "values": values},
        "result": {"consistent": consistent, "unique_values": sorted(unique)},
    }


def classify_document(file_name: str) -> dict:
    """Infer the likely document type from the filename.

    Content-based classification is a planned improvement — see README for details.
    """
    lower = file_name.lower().replace("-", "_")
    inferred = next((v for k, v in _DOC_TYPE_KEYWORDS.items() if k in lower), "unknown")
    return {
        "tool": "classify_document",
        "input": {"file_name": file_name},
        "result": {"inferred_doc_type": inferred},
    }
