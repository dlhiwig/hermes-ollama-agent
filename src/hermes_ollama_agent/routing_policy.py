from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(slots=True)
class RoutingDecision:
    roles: list[str]
    worker_cap: int
    needs_web: bool
    needs_exec: bool
    score: int
    reason: str


class RoutingPolicyEngine:
    def decide(self, text: str, recent_runs: list[dict]) -> dict[str, object]:
        t = text.lower()
        score = len(text.split()) // 20
        needs_web = any(k in t for k in ["research", "latest", "web", "search"])
        needs_exec = any(k in t for k in ["implement", "debug", "benchmark", "test", "code"])
        failures = sum(1 for run in recent_runs[:10] if str(run.get("status")) in {"partial", "failed"})
        score += failures
        roles = ["generalist"]
        if needs_web:
            roles.append("researcher")
        if needs_exec:
            roles.append("coder")
        if score >= 3:
            roles.append("reviewer")
        worker_cap = 1 if score <= 1 else 2 if score <= 3 else 3
        return asdict(RoutingDecision(roles=list(dict.fromkeys(roles)), worker_cap=worker_cap, needs_web=needs_web, needs_exec=needs_exec, score=score, reason="keyword+history heuristic"))
