# AI Usage

This folder documents AI assistant usage during development of the claims processing agent.

## Claude Code (claude.ai/code)

Session transcript: available via Claude Code session export.

Key decisions discussed with AI:
- Architecture layering (Chatbot / ClaimAgent / ClaimParser / DocReader / LLMAdapters)
- Status semantics (`pending` vs `needs_review`)
- Duplicate routing strategy (same_filename → incomplete, same_content → pending)
- Confidence immutability after extraction
- Trust model for customer replies (user_input never resolves document inconsistencies)
- LangGraph conditional routing vs hardcoded pipeline
