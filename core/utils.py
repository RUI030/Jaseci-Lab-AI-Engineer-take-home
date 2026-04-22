from __future__ import annotations

import json
from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_yaml(relative_path: str) -> dict:
    with open(Path(__file__).parent.parent / relative_path) as f:
        return yaml.safe_load(f)


def load_field_schemas():
    from core.models import FieldSchema
    data = json.loads((CONFIG_DIR / "field_schema.json").read_text())
    return [FieldSchema(**item) for item in data]
