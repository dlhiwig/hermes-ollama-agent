from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable


class TaskExecutionError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(slots=True)
class TaskSpec:
    subtask_id: int
    role: str
    task: str
    run: Callable[[], Awaitable[str]]


@dataclass(slots=True)
class TaskResult:
    subtask_id: int
    role: str
    task: str
    status: str
    attempts: int
    error_kind: str | None
    output: str


class TaskEngine:
    def __init__(self, *, max_workers: int, timeout_s: float = 90.0, retries: int = 1) -> None:
        self.max_workers = max(1, min(max_workers, 6))
        self.timeout_s = timeout_s
        self.retries = max(0, retries)

    async def execute(self, specs: list[TaskSpec]) -> list[TaskResult]:
        semaphore = asyncio.Semaphore(self.max_workers)

        async def run_spec(spec: TaskSpec) -> TaskResult:
            async with semaphore:
                attempts = 0
                while True:
                    attempts += 1
                    try:
                        output = await asyncio.wait_for(spec.run(), timeout=self.timeout_s)
                        return TaskResult(spec.subtask_id, spec.role, spec.task, "ok", attempts, None, output)
                    except asyncio.TimeoutError:
                        if attempts > self.retries:
                            return TaskResult(spec.subtask_id, spec.role, spec.task, "error", attempts, "timeout", "Task timed out.")
                    except Exception as exc:
                        kind = "connection" if "connect" in str(exc).lower() else "runtime"
                        if attempts > self.retries:
                            return TaskResult(spec.subtask_id, spec.role, spec.task, "error", attempts, kind, f"Task failed: {exc}")

        return await asyncio.gather(*(run_spec(spec) for spec in specs))
