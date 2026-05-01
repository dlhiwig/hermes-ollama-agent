from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .events import EventBus
from .health import HealthChecker
from .runtime import HermesRuntime


@dataclass(slots=True)
class KernelStatus:
    initialized: bool
    skills_loaded: int
    clients_warm: int
    routing: str


class AgentKernel:
    def __init__(self, runtime: HermesRuntime) -> None:
        self.runtime = runtime
        self.events = EventBus()
        self.health_checker = HealthChecker(runtime)
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        self.runtime.memory.ensure()
        self.runtime.skills.load()
        self._initialized = True
        self.events.emit("kernel.initialized")

    async def shutdown(self) -> None:
        self.events.emit("kernel.shutdown")
        self._initialized = False

    async def health(self) -> str:
        results = await self.health_checker.run()
        self.events.emit(
            "kernel.health_checked",
            payload={
                "ok": all(item.ok for item in results),
                "checks": len(results),
            },
        )
        return self.health_checker.format(results)

    def status(self) -> KernelStatus:
        return KernelStatus(
            initialized=self._initialized,
            skills_loaded=len(self.runtime.skills.names()),
            clients_warm=len(self.runtime._clients),
            routing=self.runtime.describe_routing(),
        )

    async def chat_turn(self, user_input: str) -> str:
        self.events.emit("turn.started", {"input": user_input[:120]})
        text = await self.runtime.run_turn(user_input)
        self.events.emit("turn.completed", {"output_len": len(text)})
        return text

    async def delegate(self, objective: str, max_workers: int) -> str:
        self.events.emit("delegate.started", {"objective": objective[:120], "workers": max_workers})
        result = await self.runtime.delegate_parallel(objective=objective, max_workers=max_workers)
        self.events.emit("delegate.completed", {"output_len": len(result)})
        return result

    def add_memory(self, note: str) -> None:
        self.runtime.add_memory(note)
        self.events.emit("memory.updated", {"kind": "memory"})

    def add_user_pref(self, note: str) -> None:
        self.runtime.add_user_pref(note)
        self.events.emit("memory.updated", {"kind": "user"})

    def recent_events(self, limit: int = 10) -> list[dict[str, Any]]:
        return [
            {"name": event.name, "timestamp": event.timestamp, "payload": event.payload}
            for event in self.events.recent(limit)
        ]

    def list_runs(self) -> list[dict[str, Any]]:
        return self.runtime.list_runs()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self.runtime.get_run(run_id)

    def summarize_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.runtime.summarize_run(payload)

    async def resume_run(self, run_id: str, max_workers: int) -> str:
        self.events.emit("delegate.resumed", {"run_id": run_id, "workers": max_workers})
        return await self.runtime.resume_run(run_id=run_id, max_workers=max_workers)

    def abort_run(self, run_id: str) -> str:
        self.events.emit("delegate.aborted", {"run_id": run_id})
        return self.runtime.abort_run(run_id)

    async def retry_run(self, run_id: str, max_workers: int, failed_only: bool = True) -> str:
        self.events.emit("delegate.retried", {"run_id": run_id, "workers": max_workers, "failed_only": failed_only})
        return await self.runtime.retry_run(run_id=run_id, max_workers=max_workers, failed_only=failed_only)
