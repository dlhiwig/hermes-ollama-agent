from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .runtime import HermesRuntime


@dataclass(slots=True)
class HealthCheckResult:
    name: str
    ok: bool
    details: str


class HealthChecker:
    def __init__(self, runtime: HermesRuntime) -> None:
        self.runtime = runtime

    async def run(self) -> list[HealthCheckResult]:
        results = [
            self._check_skills(),
            self._check_memory(),
            self._check_ollama_version(),
            self._check_ollama_models(),
        ]
        return results

    def format(self, results: list[HealthCheckResult]) -> str:
        lines = ["Health check results:"]
        for result in results:
            marker = "OK" if result.ok else "FAIL"
            lines.append(f"- [{marker}] {result.name}: {result.details}")
        overall = all(item.ok for item in results)
        lines.append(f"Overall: {'HEALTHY' if overall else 'UNHEALTHY'}")
        return "\n".join(lines)

    def _check_skills(self) -> HealthCheckResult:
        names = self.runtime.skills.names()
        if names:
            return HealthCheckResult("skills", True, f"{len(names)} loaded ({', '.join(names[:5])})")
        return HealthCheckResult("skills", False, "no skills loaded")

    def _check_memory(self) -> HealthCheckResult:
        try:
            memory_text = self.runtime.memory.read_memory()
            user_text = self.runtime.memory.read_user()
            return HealthCheckResult(
                "memory",
                True,
                f"memory={len(memory_text)} chars, user={len(user_text)} chars",
            )
        except Exception as exc:
            return HealthCheckResult("memory", False, str(exc))

    def _check_ollama_version(self) -> HealthCheckResult:
        url = self._ollama_api_url("/api/version")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            version = payload.get("version", "unknown")
            return HealthCheckResult("ollama.version", True, f"version={version}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return HealthCheckResult("ollama.version", False, str(exc))

    def _check_ollama_models(self) -> HealthCheckResult:
        url = self._ollama_api_url("/api/tags")
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
            models = payload.get("models", [])
            if not models:
                return HealthCheckResult("ollama.models", False, "no models returned")
            names = [str(item.get("name", "unknown")) for item in models[:5]]
            return HealthCheckResult("ollama.models", True, f"{len(models)} model(s): {', '.join(names)}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return HealthCheckResult("ollama.models", False, str(exc))

    def _ollama_api_url(self, endpoint: str) -> str:
        base = self.runtime.config.base_url.rstrip("/")
        if base.endswith("/v1"):
            root = base[: -len("/v1")]
        else:
            root = base
        return f"{root}{endpoint}"
