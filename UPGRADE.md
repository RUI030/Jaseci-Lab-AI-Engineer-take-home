# UPGRADE.md — byLLM Integration Plan

This document describes how to integrate [byLLM](https://github.com/Jaseci-Labs/jaseci/tree/main/jac-byllm)
into this project so that tool calls become genuinely LLM-driven at runtime, closing the gap flagged in
`TODO.md` ("tools_used not LLM-driven — Python-driven, not ReAct loop").

---

## Background

The spec (Section 7) calls for the agent to **decide at runtime** which tools to invoke. The current
implementation calls `validate_vin`, `check_field_consistency`, and `classify_document` through
hard-coded Python `if`-branches inside `core/parser.py` and `core/agent.py`. Those branches make
`tools_used` an audit log, not evidence of actual agent reasoning.

byLLM fixes this: any typed Python function decorated with `@by(llm)` becomes LLM-driven. Pass a
list of tool functions as `tools=[...]` and the LLM runs a ReAct-style loop — it decides which tools
to call, in what order, until it can produce a typed return value.

---

## What Changes vs. What Stays

| Component | Change? | Notes |
|---|---|---|
| `core/tools.py` | **Yes** — add `@by(llm)` entry points | See Path 1 below |
| `core/parser.py` | **Yes** — call byLLM entry points instead of tools directly | Remove hardcoded `if vin:` branches |
| `core/llm_adapters.py` | **No** | VLM document extraction stays as-is |
| `core/doc_reader.py` | **No** | PDF/Image/Text reading stays as-is |
| `core/agent.py` | **Minimal** | No graph topology changes; node functions call parser as before |
| `core/models.py` | **No** | `tools_used` field already exists and matches byLLM output shape |
| `config/` | **No** | All config files stay |

---

## Installation

```bash
pip install jaclang jac-byllm
```

Or add to `requirements/basic.txt`:

```
jaclang
jac-byllm
```

byLLM runs on [LiteLLM](https://github.com/BerriAI/litellm) internally, which means it supports any
provider: OpenAI, Gemini, Anthropic, or local models via Ollama.

---

## Model Configuration

### Option A — Local Qwen via Ollama (recommended for open-source)

Pull the model first:

```bash
ollama pull qwen2.5:7b
```

Then configure the byLLM model instance in `core/tools.py`:

```python
from byllm.lib import Model

tool_llm = Model(
    model_name="ollama/qwen2.5:7b",
    config={
        "api_base": "http://localhost:11434",
        "api_key": "none",       # Ollama ignores this
    },
)
```

For a larger reasoning model:

```python
tool_llm = Model(
    model_name="ollama/qwen2.5:72b",
    config={"api_base": "http://localhost:11434", "api_key": "none"},
)
```

### Option B — Gemini (same key already in .env)

```python
import os
from byllm.lib import Model

tool_llm = Model(
    model_name="gemini/gemini-1.5-flash",
    config={"api_key": os.environ["GEMINI_API_KEY"]},
)
```

### Option C — jac.toml (project-level default, no code change per tool)

Create `jac.toml` at the repo root:

```toml
[plugins.byllm.model]
default_model = "ollama/qwen2.5:7b"
api_base = "http://localhost:11434"
api_key = "none"
```

With jac.toml, `@by(llm)` picks up the default without an explicit `Model` instance.

---

## Path 1 — byLLM Python API for Tool Dispatch (Recommended)

This is the minimal-change path. The LangGraph graph and all existing node logic stays unchanged.
Only `core/tools.py` and the call sites in `core/parser.py` change.

### Step 1 — Rewrite `core/tools.py`

Replace the current pure-Python tool functions with byLLM-decorated dispatcher functions.
Keep the original Python functions as helpers; add LLM-driven wrappers on top.

```python
# core/tools.py
from __future__ import annotations

import re
from byllm.lib import by, Model

tool_llm = Model(
    model_name="ollama/qwen2.5:7b",
    config={"api_base": "http://localhost:11434", "api_key": "none"},
)

# ── Raw tool helpers (unchanged logic, usable by LLM as tools) ──────────────

_VIN_RE = re.compile(r"^[A-Z0-9]{17}$", re.IGNORECASE)

def validate_vin(vin: str) -> dict:
    """Check that vin matches the standard 17-character alphanumeric VIN format."""
    valid = bool(_VIN_RE.fullmatch(vin))
    return {
        "tool": "validate_vin",
        "input": {"vin": vin},
        "result": {
            "valid": valid,
            "reason": None if valid else f"'{vin}' is not a valid 17-char VIN",
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

def classify_document(file_name: str, actual_type: str | None = None) -> dict:
    """Record document classification. actual_type is the VLM-confirmed type."""
    _KEYWORDS = {
        "police_report": "police_report",
        "finance_agreement": "finance_agreement",
        "settlement_breakdown": "settlement_breakdown",
        "customer_reply": "customer_reply",
    }
    lower = file_name.lower().replace("-", "_")
    inferred = next((v for k, v in _KEYWORDS.items() if k in lower), "unknown")
    result: dict = {"inferred_doc_type": inferred}
    if actual_type is not None:
        result["actual_doc_type"] = actual_type
        result["overridden"] = actual_type != inferred
    return {"tool": "classify_document", "input": {"file_name": file_name}, "result": result}


# ── LLM-driven dispatcher ────────────────────────────────────────────────────

class ValidationReport(BaseModel):  # import from pydantic
    """Structured result returned by the LLM-driven validation agent."""
    tools_called: list[dict]           # mirrors Claim.tools_used entries
    issues_found: list[str]            # human-readable issue descriptions
    recommendation: str                # "complete" | "incomplete" | "needs_review"

@by(tool_llm, tools=[validate_vin, check_field_consistency])
def run_cross_validation(
    extracted_fields: dict,
    field_schemas: list,
) -> ValidationReport:
    """
    Validate the extracted claim fields using available tools.
    Call validate_vin for any VIN field present.
    Call check_field_consistency for any field that appears in more than one document.
    Return a structured ValidationReport.
    """
    ...
```

The `...` body is intentional — byLLM replaces it with LLM inference. The docstring becomes the
system prompt. The LLM sees `validate_vin` and `check_field_consistency` as callable tools and
decides when to invoke them.

### Step 2 — Update `core/parser.py`

In `cross_validate`, replace the explicit `if vin: validate_vin(...)` and field-loop calls with a
single call to `run_cross_validation`:

```python
# core/parser.py  (inside cross_validate)
from core.tools import run_cross_validation

def cross_validate(self, claim: Claim, field_schemas: list[FieldSchema]) -> None:
    report = run_cross_validation(
        extracted_fields={k: v.model_dump() for k, v in claim.extracted_fields.items()},
        field_schemas=[s.model_dump() for s in field_schemas],
    )
    claim.tools_used.extend(report.tools_called)
    for issue_text in report.issues_found:
        claim.validation_issues.append(
            ValidationIssue(
                issue_type="inconsistency",
                severity="blocking",
                description=issue_text,
            )
        )
```

The LLM now decides the tool call sequence. `tools_used` is populated by actual LLM decisions, not
Python `if`-branches.

### Step 3 — Verify `tools_used` output

After the upgrade, each `tools_used` entry should look like:

```json
{
  "tool": "validate_vin",
  "input": {"vin": "1HGBH41JXMN109186"},
  "result": {"valid": true, "reason": null}
}
```

This matches `Claim.tools_used: list[dict]` — no model changes needed.

---

## Path 2 — Full Jac Graph Rewrite (Future / Optional)

Rewrite the entire orchestration layer in Jac (`.jac` files) using byLLM's native graph API.
This replaces LangGraph entirely with Jaseci walkers.

**Not recommended for the current sprint** — the LangGraph graph works and the byLLM Python API
(Path 1) already closes the spec gap. A full rewrite is worthwhile only if the team adopts Jac
as a primary language.

Example of what the cross-validation node would look like in Jac:

```jac
import from byllm.lib { by }

def validate_vin(vin: str) -> dict { ... }
def check_field_consistency(field_name: str, values: dict[str, str]) -> dict { ... }

obj ValidationReport {
    has tools_called: list,
        issues_found: list[str],
        recommendation: str;
}

"""Validate extracted claim fields using available tools."""
def run_cross_validation(extracted_fields: dict, field_schemas: list) -> ValidationReport
    by llm(tools=[validate_vin, check_field_consistency]);
```

---

## What This Does NOT Change

- **Document extraction** (`core/doc_reader.py`): PDF/image VLM extraction uses `BaseLLMClient`
  directly and must stay that way. byLLM's vision support (`from byllm.lib import Image`) is an
  alternative, but switching would require restructuring `DocReader` classes and is out of scope.

- **LangGraph graph topology** (`core/agent.py`): The four-node graph, conditional routing, and
  `ClaimState` TypedDict are unchanged. byLLM runs inside node functions, not as the graph itself.

- **Status determination** (`core/parser.py → determine_status`): Still Python-driven. Status logic
  must be deterministic per the compliance requirement documented in `CLAUDE.md`.

- **Claim persistence** (`.cache/claim_state.json`): Unchanged.

- **Config files** (`config/`): Unchanged. The byLLM model instance is configured in code
  (`core/tools.py`) or via `jac.toml`.

---

## Spec Gap Closed

Before this upgrade, `tools_used` recorded which Python branches executed.
After this upgrade, `tools_used` records which tools the **LLM** chose to call to reach its answer —
satisfying Section 7's requirement for conditional tool use decided by the agent at runtime.

The remaining TODO item ("LangGraph routing is deterministic") is intentional per the compliance
requirement and is not addressed here.
