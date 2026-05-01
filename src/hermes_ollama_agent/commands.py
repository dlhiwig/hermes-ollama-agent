from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from .kernel import AgentKernel

CommandHandler = Callable[[str], Awaitable[str | None]]


@dataclass(slots=True)
class CommandContext:
    kernel: AgentKernel
    delegate_workers: int


class CommandRegistry:
    def __init__(self, context: CommandContext) -> None:
        self.ctx = context
        self._handlers: dict[str, CommandHandler] = {
            "/help": self._cmd_help,
            "/routing": self._cmd_routing,
            "/skills": self._cmd_skills,
            "/skill": self._cmd_skill,
            "/search": self._cmd_search,
            "/reload": self._cmd_reload,
            "/memory": self._cmd_memory,
            "/remember": self._cmd_remember,
            "/prefer": self._cmd_prefer,
            "/delegate": self._cmd_delegate,
            "/health": self._cmd_health,
            "/status": self._cmd_status,
            "/events": self._cmd_events,
            "/runs": self._cmd_runs,
            "/run": self._cmd_run,
            "/resume": self._cmd_resume,
            "/abort": self._cmd_abort,
            "/retry": self._cmd_retry,
        }

    def has_command(self, text: str) -> bool:
        if not text.startswith("/"):
            return False
        command = text.split(" ", 1)[0].strip()
        return command in self._handlers

    async def execute(self, text: str) -> str | None:
        command, _, rest = text.partition(" ")
        handler = self._handlers.get(command.strip())
        if handler is None:
            return f"Unknown command: {command}"
        return await handler(rest.strip())

    async def _cmd_help(self, _: str) -> str:
        return (
            "Commands:\n"
            "  /help\n  /exit\n  /routing\n  /skills\n  /skill <name>\n"
            "  /search <query>\n  /delegate <objective>\n  /reload\n"
            "  /memory\n  /remember <note>\n  /prefer <note>\n"
            "  /health\n  /status\n  /events"
            "\n  /runs\n  /run <id>\n  /resume <id>\n  /abort <id>\n  /retry <id>"
        )

    async def _cmd_routing(self, _: str) -> str:
        return self.ctx.kernel.runtime.describe_routing()

    async def _cmd_skills(self, _: str) -> str:
        return self.ctx.kernel.runtime.list_skills()

    async def _cmd_skill(self, arg: str) -> str:
        if not arg:
            return "Usage: /skill <name>"
        return self.ctx.kernel.runtime.read_skill(arg)

    async def _cmd_search(self, arg: str) -> str:
        if not arg:
            return "Usage: /search <query>"
        return self.ctx.kernel.runtime.search_skills(arg)

    async def _cmd_reload(self, _: str) -> str:
        self.ctx.kernel.runtime.reload_skills()
        return "Reloaded skills."

    async def _cmd_memory(self, _: str) -> str:
        return self.ctx.kernel.runtime.get_memory()

    async def _cmd_remember(self, arg: str) -> str:
        if not arg:
            return "Usage: /remember <note>"
        self.ctx.kernel.add_memory(arg)
        return "Saved to MEMORY.md"

    async def _cmd_prefer(self, arg: str) -> str:
        if not arg:
            return "Usage: /prefer <note>"
        self.ctx.kernel.add_user_pref(arg)
        return "Saved to USER.md"

    async def _cmd_delegate(self, arg: str) -> str:
        if not arg:
            return "Usage: /delegate <objective>"
        return await self.ctx.kernel.delegate(arg, max_workers=self.ctx.delegate_workers)

    async def _cmd_health(self, _: str) -> str:
        return await self.ctx.kernel.health()

    async def _cmd_status(self, _: str) -> str:
        status = self.ctx.kernel.status()
        return (
            "Kernel status:\n"
            f"- initialized: {status.initialized}\n"
            f"- skills_loaded: {status.skills_loaded}\n"
            f"- clients_warm: {status.clients_warm}\n"
        )

    async def _cmd_events(self, _: str) -> str:
        rows = self.ctx.kernel.recent_events(limit=10)
        if not rows:
            return "No events."
        formatted = ["Recent events:"]
        for item in rows:
            formatted.append(f"- {item['timestamp']} {item['name']} {item['payload']}")
        return "\n".join(formatted)

    async def _cmd_runs(self, _: str) -> str:
        rows = self.ctx.kernel.list_runs()
        if not rows:
            return "No runs."
        out = ["Recent runs:"]
        for row in rows:
            summary = self.ctx.kernel.summarize_run(row)
            out.append(
                f"- {summary.get('run_id')} status={summary.get('status')} "
                f"ok={summary.get('ok_count')} errors={summary.get('error_count')} "
                f"error_kinds={summary.get('error_counts')} objective={str(summary.get('objective', ''))[:60]}"
            )
        return "\n".join(out)

    async def _cmd_run(self, arg: str) -> str:
        if not arg:
            return "Usage: /run <id>"
        row = self.ctx.kernel.get_run(arg)
        if row is None:
            return f"Run not found: {arg}"
        summary = self.ctx.kernel.summarize_run(row)
        return f"Summary: {summary}\nPayload: {row}"

    async def _cmd_resume(self, arg: str) -> str:
        if not arg:
            return "Usage: /resume <id>"
        return await self.ctx.kernel.resume_run(arg, self.ctx.delegate_workers)

    async def _cmd_abort(self, arg: str) -> str:
        if not arg:
            return "Usage: /abort <id>"
        return self.ctx.kernel.abort_run(arg)

    async def _cmd_retry(self, arg: str) -> str:
        if not arg:
            return "Usage: /retry <id>"
        return await self.ctx.kernel.retry_run(arg, self.ctx.delegate_workers, failed_only=True)
