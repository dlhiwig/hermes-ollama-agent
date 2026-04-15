# Hermes Ollama Agent (Local Prototype)

Hermes-inspired local agent runtime for this system, built with:

- Microsoft Agent Framework for the agent loop and tool calls
- Ollama via OpenAI-compatible endpoint (`http://localhost:11434/v1/`)
- Local markdown memory and skill files

## What this implements

- Persistent `MEMORY.md` and `USER.md` context
- Markdown skill registry (`skills/*.md`)
- Tool surface for listing/searching/reading skills and saving memory notes
- Interactive CLI shell
- Phase 2 delegation: planner + parallel child workers + synthesis

## Prerequisites

- Python 3.11+
- Ollama running locally
- A pulled model (example): `ollama pull llama3.2`

## Install

```bash
cd hermes-ollama-agent
python -m pip install -e .
```

## Run

```bash
hermes-ollama
```

Optional:

```bash
hermes-ollama --model llama3.2 --base-url http://localhost:11434/v1/
```

Delegation worker count:

```bash
hermes-ollama --delegate-workers 4
```

Role-specific model routing example:

```bash
hermes-ollama ^
  --model llama3.2 ^
  --model-planner llama3.2:1b ^
  --model-reviewer llama3.2:1b ^
  --model-coder qwen2.5-coder:7b ^
  --model-synthesizer llama3.2
```

Environment variable equivalents:

```bash
set OLLAMA_MODEL=llama3.2
set OLLAMA_MODEL_PLANNER=llama3.2:1b
set OLLAMA_MODEL_REVIEWER=llama3.2:1b
set OLLAMA_MODEL_CODER=qwen2.5-coder:7b
set OLLAMA_MODEL_SYNTHESIZER=llama3.2
```

## Commands

- `/help` show available commands
- `/routing` show model routing and per-role tool policy
- `/skills` list loaded skills
- `/skill <name>` print a skill
- `/search <query>` search skills
- `/delegate <objective>` decompose objective and run subtasks in parallel
- `/reload` reload skill markdown from disk
- `/memory` print memory context
- `/remember <note>` append durable memory note
- `/prefer <note>` append user preference note
- `/exit` quit

## Default role tool permissions

- `default`: list/read/search skills, read memory, add memory, add user preference
- `planner`: list/search skills, read memory
- `researcher`: list/read/search skills, read memory
- `coder`: list/read/search skills, read memory, add memory
- `reviewer`: list/read/search skills, read memory
- `synthesizer`: list skills, read memory
- `generalist`: list/read/search skills, read memory

## Project layout

```text
hermes-ollama-agent/
  src/hermes_ollama_agent/
    cli.py
    config.py
    memory.py
    prompts.py
    runtime.py
    skills.py
  skills/
  .hermes/              # created at runtime
```

## Next extensions

- Add vector search memory backend
- Add web research and code execution toolchains
- Add dynamic routing based on task complexity
