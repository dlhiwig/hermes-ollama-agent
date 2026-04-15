# Architecture Plan

## Goal

Build a Hermes-style local agent that runs on Ollama and is extensible toward multi-agent workflows.

## Current implementation

- Runtime: Microsoft Agent Framework `OpenAIChatClient` with `base_url`
- Model serving: Ollama OpenAI-compatible endpoint
- Memory: local markdown files (`.hermes/MEMORY.md`, `.hermes/USER.md`)
- Skills: markdown skill docs as a lightweight skill system
- Delegation layer: planner -> parallel role workers -> synthesis
- Role-based model routing (planner/reviewer/coder/synthesizer/etc.)
- Per-role tool permission policy (read-only by default, coder write-restricted)
- UX: terminal REPL with `/delegate`

## Why this stack

- Agent Framework provides a clean tool-calling runtime with OpenAI-compatible endpoints.
- Ollama provides local model hosting and privacy.
- Hermes patterns (memory + skills + delegation-first design) map well to markdown-backed state and role-based expansion.

## Phase roadmap

1. Base runtime (done)
2. Delegation layer (done)
3. Role-based model routing and tool policy (done)
4. Workflow graph with persistent role state
5. Memory retrieval upgrade (vector backend)
6. Web UI and long-running job support

## Suggested future integrations

- `langchain` for graph/tool ecosystem breadth
- `ruvector` as optional vector memory backend
- `deer-flow`-style planner/researcher/reporter orchestration pattern
