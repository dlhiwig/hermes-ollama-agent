from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(slots=True)
class ExecResult:
    exit_code: int
    stdout_tail: str
    stderr_tail: str
    duration_ms: int
    timed_out: bool


class SandboxedExecutor:
    def __init__(self, *, allowed_prefixes: list[str], cwd_allowlist: list[Path], timeout_s: int = 30) -> None:
        self.allowed_prefixes = allowed_prefixes
        self.cwd_allowlist = cwd_allowlist
        self.timeout_s = timeout_s

    def run(self, command: str, cwd: Path) -> dict[str, object]:
        if not any(str(cwd).lower().startswith(str(base).lower()) for base in self.cwd_allowlist):
            return asdict(ExecResult(126, "", "blocked cwd", 0, False))
        tokens = command.strip().split()
        if not tokens:
            return asdict(ExecResult(2, "", "empty command", 0, False))
        if tokens[0] not in self.allowed_prefixes:
            return asdict(ExecResult(126, "", f"blocked command: {tokens[0]}", 0, False))
        started = time.time()
        try:
            proc = subprocess.run(tokens, cwd=str(cwd), capture_output=True, text=True, timeout=self.timeout_s, shell=False)
            dur = int((time.time() - started) * 1000)
            return asdict(ExecResult(proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:], dur, False))
        except subprocess.TimeoutExpired as exc:
            dur = int((time.time() - started) * 1000)
            return asdict(ExecResult(124, (exc.stdout or "")[-2000:], (exc.stderr or "")[-2000:], dur, True))
