from __future__ import annotations

import argparse
import asyncio

from .config import HermesConfig
from .runtime import HermesRuntime

HELP_TEXT = """Commands:
  /help                  Show this help
  /exit                  Exit the session
  /routing               Show role -> model and role -> tool policy
  /skills                List loaded skills
  /skill <name>          Show one skill markdown
  /search <query>        Search skills
  /delegate <objective>  Plan + run parallel child agents
  /reload                Reload skill files from disk
  /memory                Show current memory context
  /remember <note>       Append note to MEMORY.md
  /prefer <note>         Append note to USER.md
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-ollama",
        description="Hermes-inspired local agent using Ollama",
    )
    parser.add_argument("--model", default=None, help="Ollama model id (e.g. llama3.2)")
    parser.add_argument("--model-planner", default=None, help="Model for planner agent role")
    parser.add_argument("--model-researcher", default=None, help="Model for researcher agent role")
    parser.add_argument("--model-coder", default=None, help="Model for coder agent role")
    parser.add_argument("--model-reviewer", default=None, help="Model for reviewer agent role")
    parser.add_argument("--model-synthesizer", default=None, help="Model for synthesizer agent role")
    parser.add_argument("--model-generalist", default=None, help="Model for generalist agent role")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="API key placeholder for compatible endpoints")
    parser.add_argument("--state-dir", default=None, help="Directory for MEMORY.md and USER.md")
    parser.add_argument("--skills-dir", default=None, help="Directory with markdown skills")
    parser.add_argument(
        "--delegate-workers",
        default=3,
        type=int,
        help="Max parallel child agents for /delegate (default: 3)",
    )
    return parser


def _apply_overrides(cfg: HermesConfig, args: argparse.Namespace) -> HermesConfig:
    if args.model:
        cfg.model_id = args.model
    if args.model_planner:
        cfg.planner_model_id = args.model_planner
    if args.model_researcher:
        cfg.researcher_model_id = args.model_researcher
    if args.model_coder:
        cfg.coder_model_id = args.model_coder
    if args.model_reviewer:
        cfg.reviewer_model_id = args.model_reviewer
    if args.model_synthesizer:
        cfg.synthesizer_model_id = args.model_synthesizer
    if args.model_generalist:
        cfg.generalist_model_id = args.model_generalist
    if args.base_url:
        cfg.base_url = args.base_url
    if args.api_key:
        cfg.api_key = args.api_key
    if args.state_dir:
        from pathlib import Path

        cfg.state_dir = Path(args.state_dir)
    if args.skills_dir:
        from pathlib import Path

        cfg.skills_dir = Path(args.skills_dir)
    return cfg


async def run_repl(args: argparse.Namespace) -> None:
    cfg = _apply_overrides(HermesConfig.from_env(), args)
    runtime = HermesRuntime(cfg)

    print(f"Model: {cfg.model_id}")
    print(f"Base URL: {cfg.base_url}")
    print(f"State dir: {cfg.state_dir}")
    print(f"Skills dir: {cfg.skills_dir}")
    print(runtime.describe_routing())
    print("Type /help for commands.\n")

    while True:
        try:
            user_input = input("you> ").strip()
        except EOFError:
            print()
            break

        if not user_input:
            continue
        if user_input in {"/exit", "/quit"}:
            break
        if user_input == "/help":
            print(HELP_TEXT)
            continue
        if user_input == "/routing":
            print(runtime.describe_routing())
            continue
        if user_input == "/skills":
            print(runtime.list_skills())
            continue
        if user_input.startswith("/skill "):
            print(runtime.read_skill(user_input[7:].strip()))
            continue
        if user_input.startswith("/search "):
            print(runtime.search_skills(user_input[8:].strip()))
            continue
        if user_input.startswith("/delegate "):
            objective = user_input[10:].strip()
            if not objective:
                print("Usage: /delegate <objective>")
                continue
            try:
                delegated = await runtime.delegate_parallel(
                    objective=objective,
                    max_workers=args.delegate_workers,
                )
            except Exception as exc:  # pragma: no cover - runtime integration path
                print(f"Delegation error: {exc}")
                continue
            print(f"hermes-delegate> {delegated}\n")
            continue
        if user_input == "/reload":
            runtime.reload_skills()
            print("Reloaded skills.")
            continue
        if user_input == "/memory":
            print(runtime.get_memory())
            continue
        if user_input.startswith("/remember "):
            runtime.add_memory(user_input[10:].strip())
            print("Saved to MEMORY.md")
            continue
        if user_input.startswith("/prefer "):
            runtime.add_user_pref(user_input[8:].strip())
            print("Saved to USER.md")
            continue

        try:
            response = await runtime.run_turn(user_input)
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency. Install with: pip install -e ."
            ) from exc
        except Exception as exc:  # pragma: no cover - runtime integration path
            print(f"Agent error: {exc}")
            continue

        print(f"hermes> {response}\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        asyncio.run(run_repl(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
