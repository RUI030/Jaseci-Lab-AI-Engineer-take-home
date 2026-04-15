# Take-Home Assignment: AI Claims Processing Agent

## Overview

Build an AI agent that processes insurance claims — reading documents, pulling out key info, checking that everything adds up, and figuring out what to do next.

We're not looking for a perfect system. We want to see how you think about the problem: how you break it down, what tools you build, and how your agent handles the messy stuff.

---

## Business Context

A customer files a total-loss vehicle insurance claim and submits a bunch of documents. These come in as PDFs, scanned images, and sometimes plain text. The quality varies. Your agent needs to:

- Pull out the important fields from each document
- Check that everything is there and makes sense
- Figure out the next step
- Write a message to the customer if something's missing

You don't need to know anything about insurance. All the rules are below.

---

## Input Data

You get **5 claims** (`CLM-001` through `CLM-005`). Each claim folder has a mix of:

- **PDFs** (clean, machine-readable)
- **Scanned images** (noisy, slightly rotated — like something that went through an actual scanner)
- **Text files** (customer emails)
---

## Requirements

### 1. Document Intake

Your agent should:

- Take a claim folder as input
- Analyze and process each document. 

#### Required Document Types

A complete claim needs:

- **Police Report**
- **Finance Agreement**
- **Settlement Breakdown**

Anything else is extra. Your agent should still look at it, but shouldn't require it.

---

### 2. Field Extraction

Pull out these fields when they're available:

| Field | Validation Rule |
|-------|----------------|
| VIN | Exactly 17 alphanumeric characters |
| Date of Loss | Valid date |
| Insurance Payout | Numeric |
| Outstanding Loan Balance | Numeric |

For each field, report a **confidence level** (high / medium / low) and a short reason when it's not high.
---

### 3. Validation

#### Cross-Document Consistency

When the same field shows up in multiple documents, the values should match. If they don't:

- Flag it and report which documents disagree
- If possible, provide educated guess on the correct value.


#### Duplicate Documents

The agent should be able to handle duplicate documents.

---

### 4. Decision Making

Your agent should decide the claim status:

| Status | When |
|--------|------|
| `complete` | Everything's there, valid, and consistent |
| `incomplete` | Missing documents or fields |
| `needs_review` | Data conflicts, low-confidence extractions, or things that can't be resolved automatically |

---

### 5. Multi-Turn Processing

Some claims include a **customer reply** (a text file) that responds to a previous request for information. Your agent should:

- Process the original documents first
- Figure out what's missing
- Then read the customer reply
- Re-evaluate the claim with the new info

The reply might only partially answer the question. Don't assume it fixes everything.

---

### 6. Interactive Mode (Highly Encouraged)

The customer reply text files simulate what would really be a live conversation. Instead of processing static files, consider making your system interactive — a CLI chat, a simple web UI, whatever you prefer — where a user can play the role of the customer and talk to the agent in real time.

This is how a system like this would actually work in production, and building it will surface design problems that a batch processor won't.

---

### 7. Tool Design (Core Requirement)

Your agent needs to use tools, but **we're not telling you which ones to build**.

You decide:

- What tools make sense
- How they're wired up to the agent
- When the agent should call them vs. just handle things directly

The important thing is that tool usage is **conditional** — the agent decides at runtime, not a hardcoded sequence.

In your README, cover:

- What tools you built and why
- How the agent decides when to use them
- What you thought about building but didn't

---

### 8. Claim Prioritization

After processing all 5 claims, output a **recommended processing order** — which ones to finalize first and why.

There's no single right answer here. We just want to see your reasoning.

---

### 9. Output Format

For each claim, return structured output. An example can be as follows. Again, this is just an example. Feel free to create your own format as it fits your design

```json
{
  "claim_id": "CLM-001",
  "status": "complete | incomplete | needs_review",
  "extracted_fields": {
    "vin": {
      "value": "1HGCM82633A004352",
      "confidence": "high",
      "source": "police_report.pdf",
      "reason": null
    },
    "date_of_loss": { "..." : "..." },
    "insurance_payout": { "..." : "..." },
    "loan_balance": { "..." : "..." }
  },
  "documents": {
    "identified": [
      {"file": "police_report.pdf", "type": "police_report"},
      {"file": "adjuster_note.png", "type": "unknown — handwritten note"}
    ],
    "missing": ["finance_agreement"],
    "duplicates": []
  },
  "issues": [
    {
      "type": "inconsistency | missing | invalid | low_confidence",
      "description": "VIN mismatch between police report and finance agreement",
      "details": "police_report: 2T1BURHE5JC034127, finance_agreement: 2T1BURHE5JC034182"
    }
  ],
  "next_action": {
    "type": "finalize | message_customer | escalate",
    "message": "..."
  },
  "tools_used": [
    {"tool": "vin_validator", "input": "1HGCM82633A004352", "result": "valid"}
  ]
}
```

After all claims, include a prioritization:

```json
{
  "processing_order": [
    {"claim_id": "CLM-001", "reason": "All documents present and valid — ready to finalize"},
    {"claim_id": "CLM-002", "reason": "..."}
  ]
}
```

---

## Customer Communication

When something's missing or doesn't add up, your agent should write a message to the customer. Design the response style as you see fit.

---

## Technical Guidelines

- Python for backend (required)
- Any frameworks or libraries
- AI coding tools are fair game — ChatGPT, Claude, Copilot, Cursor, whatever you use. See below for what we'd like you to include.
- Any LLM provider for your agent (OpenAI, Anthropic, open-source, etc.). **Using open-source models will be a PLUS.**

### Show Your Work with AI Tools

We assume you'll use AI assistants for parts of this. That's fine and expected.

Include an `ai_usage/` folder with your AI chat logs. Most tools make this easy:

- **ChatGPT**: Hit the share button, drop the link in a `links.md` file
- **Claude.ai**: Export or copy the conversation
- **Claude Code**: Copy the session transcript
- **Cursor / Windsurf**: Copy your composer/chat history
- **Anything else**: Screenshots or copy-paste work, only if it makes sense and doesn't add a ton of overhead.

Don't worry about capturing every interaction with the AI. We mainly want to see your back-and-forth on the bigger decisions — how you broke down the problem, what you asked for help with, how you pushed back when something wasn't right.

We're not checking whether you used AI. We're looking at *how* you used it. Asking good questions, spotting bad suggestions, and knowing when to override the output — that's a real skill and one we care about for this role.

---

## Jaseci / Jac / byLLM (Optional Bonus)

If you're interested, try building this (or part of it) with the **Jac programming language**, **byLLM**, or the **Jaseci ecosystem**. Not required. See [jaseci.org](https://www.jaseci.org/) and [byllm.ai](https://byllm.jaseci.org/)

---

## Time

Aim for **2–4 hours**. Don't over-engineer it. A clean, simple system that makes good decisions beats a complicated one that tries to handle everything.

If you run out of time, write up what you'd do next in the README. How you think about the problem matters as much as what you ship.

---

## What to Submit

1. **Source code** (GitHub repo or zip)
2. **README** covering:
   - Your approach and architecture
   - Key decisions and tradeoffs
   - Tool design rationale
   - What you'd do with more time
3. **Example output** for at least 2 claims
4. **`ai_usage/`** folder with your AI chat logs

---

## How We Evaluate

**Baseline** — The system works. It processes the clean cases correctly, produces structured output, and uses tools in a way that makes sense.

**Strong** — The system handles the messy cases too, not just the clean ones. The code reads well, the tools have clear boundaries, and the README shows someone who thought about the problem before writing code.

**Exceptional** — The system feels like it could grow. It handles things it wasn't explicitly told to handle, the agent makes decisions we'd trust, and the candidate can clearly explain what they'd do differently at scale.

---

## What We Actually Care About

We want to see:

- **How you break down a problem** — what becomes a tool, what stays in the agent, what gets skipped
- **How you deal with messy input** — not everything is clean or complete
- **What you build vs. what you skip** — and whether you can explain why
- **How you write about your work** — the README matters

---

## One More Thing

We'll do a **deep-dive conversation** about your submission. Be ready to walk through your design, talk about what you'd change at scale, and work through scenarios you didn't implement.
