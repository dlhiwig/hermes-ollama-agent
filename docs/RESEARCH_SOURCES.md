# Research Notes

## Core integration choices

1. Use Ollama through an OpenAI-compatible endpoint (`/v1`) so the runtime can switch between local and hosted providers with minimal code changes.
2. Use Microsoft Agent Framework as the primary runtime and tool-calling layer.
3. Implement Hermes-style local primitives first: markdown memory, markdown skills, and command-driven workflow.
4. Implement delegation as planner + parallel role workers + synthesis to mirror Hermes-style specialization.
5. Route lighter roles (planner/reviewer) to cheaper models and coder role to stronger coding models.

## Source mapping

- Ollama Hermes integration docs:
  - https://docs.ollama.com/integrations/hermes
  - Confirms direct Hermes + Ollama path and local-model focus.

- Hermes Agent references:
  - https://github.com/nousresearch/hermes-agent
  - https://hermes-agent.nousresearch.com/user-guide/features/memory/
  - https://hermes-agent.nousresearch.com/user-guide/features/skills/
  - https://hermes-agent.nousresearch.com/user-guide/features/delegation/
  - Captures the memory + skills + delegation pattern used in this prototype.

- Microsoft Agent Framework:
  - https://github.com/microsoft/agent-framework
  - https://learn.microsoft.com/en-us/agent-framework/integrations/openai-endpoints
  - Confirms `OpenAIChatClient` with `base_url` works for OpenAI-compatible servers including Ollama.

- LangChain:
  - https://github.com/langchain-ai/langchain
  - https://docs.langchain.com/oss/python/integrations/chat/ollama
  - Used as future extension guidance for richer graph/workflow integrations.

- ruflo:
  - https://github.com/ruvnet/ruflo
  - Treated as optional inspiration for orchestration/tooling expansion.

- ruvector:
  - https://github.com/ruvnet/ruvector
  - Treated as optional vector memory backend candidate.

- Deer-Flow:
  - https://github.com/bytedance/deer-flow
  - Used as architecture reference for planner/researcher/coder/reporter role split.
