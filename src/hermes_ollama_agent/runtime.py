from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import HermesConfig
from .code_exec import SandboxedExecutor
from .memory_provider import ChromaMemoryProvider, MarkdownMemoryProvider, MemoryRecord
from .memory import MemoryStore
from .prompts import build_system_prompt, build_user_turn
from .routing_policy import RoutingPolicyEngine
from .skills import SkillLibrary
from .task_engine import TaskEngine, TaskResult, TaskSpec
from .web_tools import WebResearchTools

ROLE_INSTRUCTIONS: dict[str, str] = {
    "researcher": (
        "You are a research specialist. Surface facts, assumptions, and unknowns. "
        "Prioritize constraints, dependencies, and risk flags."
    ),
    "coder": (
        "You are an implementation specialist. Convert the task into concrete code changes, "
        "interfaces, test cases, and validation steps."
    ),
    "reviewer": (
        "You are a quality and risk specialist. Challenge weak assumptions, identify regressions, "
        "and define verification criteria."
    ),
    "generalist": (
        "You are a pragmatic generalist agent. Deliver a focused execution response and explicit next actions."
    ),
}

ROLE_TOOL_POLICY: dict[str, set[str]] = {
    "default": {
        "list_skills",
        "read_skill",
        "search_skills",
        "read_memory_snapshot",
        "add_memory",
        "add_user_preference",
    },
    "planner": {"list_skills", "search_skills", "read_memory_snapshot"},
    "researcher": {"list_skills", "read_skill", "search_skills", "read_memory_snapshot"},
    "coder": {"list_skills", "read_skill", "search_skills", "read_memory_snapshot", "add_memory"},
    "reviewer": {"list_skills", "read_skill", "search_skills", "read_memory_snapshot"},
    "synthesizer": {"list_skills", "read_memory_snapshot"},
    "generalist": {"list_skills", "read_skill", "search_skills", "read_memory_snapshot"},
}

TOOL_ORDER: list[str] = [
    "list_skills",
    "read_skill",
    "search_skills",
    "read_memory_snapshot",
    "add_memory",
    "add_user_preference",
    "memory_search",
    "web_search",
    "web_fetch",
    "web_cite",
    "code_exec",
]


@dataclass(slots=True)
class DelegationSubtask:
    subtask_id: int
    role: str
    task: str


@dataclass(slots=True)
class DelegationPlan:
    objective: str
    subtasks: list[DelegationSubtask]


@dataclass(slots=True)
class DelegationResult:
    subtask: DelegationSubtask
    output: str
    status: str
    error_kind: str | None = None


class HermesRuntime:
    def __init__(self, config: HermesConfig) -> None:
        self.config = config
        self.memory = MemoryStore(
            root=config.state_dir,
            memory_char_budget=config.memory_char_budget,
            user_char_budget=config.user_char_budget,
            max_turn_chars=config.max_turn_chars,
        )
        self.skills = SkillLibrary(config.skills_dir)
        self._clients: dict[str, Any] = {}
        self._agent: Any = None
        self._runs_dir: Path = self.config.state_dir / "runs"
        self.memory.ensure()
        self.skills.load()
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._aborted_runs: set[str] = set()
        self.routing_policy = RoutingPolicyEngine()
        self.web_tools = WebResearchTools(max_requests=self.config.web_max_requests)
        self.exec_runner = SandboxedExecutor(
            allowed_prefixes=["python", "pytest", "pwsh", "powershell"],
            cwd_allowlist=[Path(".").resolve(), self.config.state_dir.resolve()],
            timeout_s=self.config.exec_timeout_s,
        )
        if self.config.memory_backend == "chroma":
            self.memory_provider = ChromaMemoryProvider(self.config.chroma_dir)
        else:
            self.memory_provider = MarkdownMemoryProvider(self.memory)

    def _build_tools(self, allowed_tool_names: set[str]) -> list[Any]:
        from agent_framework import tool

        @tool(approval_mode="never_require")
        def list_skills() -> str:
            """List available skill documents."""
            return self.skills.list_for_model()

        @tool(approval_mode="never_require")
        def read_skill(skill_name: str) -> str:
            """Read one skill markdown file by name."""
            doc = self.skills.get(skill_name)
            if doc is None:
                return f"Skill '{skill_name}' not found."
            return doc.body

        @tool(approval_mode="never_require")
        def search_skills(query: str) -> str:
            """Search skills by simple lexical matching."""
            docs = self.skills.search(query)
            if not docs:
                return "No matching skills found."
            return "\n".join(f"- {doc.name}: {doc.preview}" for doc in docs)

        @tool(approval_mode="never_require")
        def read_memory_snapshot() -> str:
            """Return current memory and user profile context."""
            return self.memory.context_block()

        @tool(approval_mode="never_require")
        def add_memory(note: str) -> str:
            """Persist a durable project memory note."""
            self.memory.add_memory_note(note)
            return "Saved note to MEMORY.md"

        @tool(approval_mode="never_require")
        def add_user_preference(note: str) -> str:
            """Persist a user preference note."""
            self.memory.add_user_note(note)
            return "Saved note to USER.md"
        @tool(approval_mode="never_require")
        def memory_search(query: str, top_k: int = 5) -> str:
            rows = self.memory_provider.search(query, max(1, min(top_k, 20)))
            if not rows:
                return "No memory matches."
            return "\n".join(f"- [{r.source}/{r.role}] {r.text[:200]}" for r in rows)
        @tool(approval_mode="never_require")
        def web_search(query: str, limit: int = 3) -> str:
            return self.web_tools.search(query, limit=limit)
        @tool(approval_mode="never_require")
        def web_fetch(url: str) -> str:
            return self.web_tools.fetch(url)
        @tool(approval_mode="never_require")
        def web_cite() -> str:
            return json.dumps(self.web_tools.cite())
        @tool(approval_mode="never_require")
        def code_exec(command: str, cwd: str = ".") -> str:
            result = self.exec_runner.run(command, Path(cwd).resolve())
            return json.dumps(result)

        registry: dict[str, Any] = {
            "list_skills": list_skills,
            "read_skill": read_skill,
            "search_skills": search_skills,
            "read_memory_snapshot": read_memory_snapshot,
            "add_memory": add_memory,
            "add_user_preference": add_user_preference,
            "memory_search": memory_search,
            "web_search": web_search,
            "web_fetch": web_fetch,
            "web_cite": web_cite,
            "code_exec": code_exec,
        }
        return [registry[name] for name in TOOL_ORDER if name in allowed_tool_names]

    @staticmethod
    def _normalize_role(role: str) -> str:
        candidate = role.strip().lower()
        if candidate in ROLE_TOOL_POLICY:
            return candidate
        return "generalist"

    def _tool_policy_for_role(self, role: str) -> set[str]:
        normalized = self._normalize_role(role)
        return set(ROLE_TOOL_POLICY.get(normalized, ROLE_TOOL_POLICY["generalist"]))

    async def _get_client(self, model_id: str) -> Any:
        if model_id in self._clients:
            return self._clients[model_id]
        from agent_framework.openai import OpenAIChatClient

        client = OpenAIChatClient(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model=model_id,
        )
        self._clients[model_id] = client
        return client

    async def _make_agent(
        self,
        *,
        name: str,
        instructions: str,
        model_role: str,
        policy_role: str,
    ) -> Any:
        model_id = self.config.model_for_role(model_role)
        client = await self._get_client(model_id)
        tools = self._build_tools(self._tool_policy_for_role(policy_role))
        return client.as_agent(name=name, instructions=instructions, tools=tools)

    async def _ensure_agent(self) -> None:
        if self._agent is not None:
            return
        self._agent = await self._make_agent(
            name="HermesLocal",
            instructions=build_system_prompt(self.skills.list_for_model()),
            model_role="default",
            policy_role="default",
        )

    def reload_skills(self) -> None:
        self.skills.load()
        self._agent = None

    def describe_routing(self) -> str:
        models = self.config.model_routing_table()
        lines = ["Model routing and tool policy:"]
        for role in [
            "default",
            "planner",
            "researcher",
            "coder",
            "reviewer",
            "synthesizer",
            "generalist",
        ]:
            allowed = ", ".join(sorted(self._tool_policy_for_role(role)))
            lines.append(f"- {role}: model={models[role]} | tools={allowed}")
        return "\n".join(lines)

    async def run_turn(self, user_input: str) -> str:
        await self._ensure_agent()
        prompt = build_user_turn(user_input=user_input, memory_block=self.memory.context_block())
        response = await self._agent.run(prompt)
        text = self._extract_text(response)
        self.memory.append_turn(user_text=user_input, assistant_text=text)
        self.memory_provider.add(MemoryRecord(id=uuid.uuid4().hex[:12], timestamp=datetime.now(timezone.utc).isoformat(), source="turn", role="assistant", text=text, tags=["chat"]))
        return text

    async def delegate_parallel(self, objective: str, max_workers: int = 3, run_id: str | None = None) -> str:
        bounded_workers = max(1, min(max_workers, 6))
        active_run_id = run_id or uuid.uuid4().hex[:12]

        routing_decision = self.routing_policy.decide(objective, self.list_runs(limit=10))
        bounded_workers = min(bounded_workers, int(routing_decision.get("worker_cap", bounded_workers)))
        plan = await self._build_delegation_plan(objective=objective, max_workers=bounded_workers)
        self._save_run(active_run_id, {"run_id": active_run_id, "objective": objective, "status": "running", "plan": [asdict(s) for s in plan.subtasks], "results": [], "attempts": [], "sources": [], "exec_artifacts": [], "routing_decision": routing_decision})
        if active_run_id in self._aborted_runs:
            self._aborted_runs.remove(active_run_id)
        results = await self._execute_plan(plan=plan, max_workers=bounded_workers)
        has_errors = any(item.status != "ok" for item in results)
        has_success = any(item.status == "ok" for item in results)
        status = "aborted" if active_run_id in self._aborted_runs else ("failed" if has_errors and not has_success else ("partial" if has_errors else "completed"))
        payload = self.get_run(active_run_id) or {"run_id": active_run_id, "objective": objective, "attempts": []}
        payload.update({"status": status, "plan": [asdict(s) for s in plan.subtasks], "results": [asdict(r) for r in results], "sources": self.web_tools.cite()})
        payload["attempts"] = payload.get("attempts", []) + [{"status": status, "results": [asdict(r) for r in results]}]
        self._save_run(active_run_id, payload)
        synthesis = await self._synthesize_results(plan=plan, results=results)

        worker_summary = "\n".join(
            f"{item.subtask_id}. [{item.role}] {item.task}"
            for item in plan.subtasks
        )
        composed = (
            "Delegation plan:\n"
            f"{worker_summary}\n\n"
            "Integrated response:\n"
            f"{synthesis}"
        )
        self.memory.append_turn(
            user_text=f"[delegated] {objective}",
            assistant_text=composed,
        )
        return f"Run ID: {active_run_id}\n{composed}"

    async def _build_delegation_plan(self, objective: str, max_workers: int) -> DelegationPlan:
        planner_prompt = (
            "Create a delegation plan in strict JSON.\n"
            "Return only JSON with this shape:\n"
            '{\n  "subtasks": [\n    {"role": "researcher|coder|reviewer|generalist", "task": "..." }\n  ]\n}\n\n'
            f"Constraints:\n- Max subtasks: {max_workers}\n"
            "- Use concise tasks.\n"
            "- Keep roles valid.\n\n"
            f"Objective:\n{objective}\n\n"
            "Context snapshot:\n"
            f"{self.memory.context_block()}\n\n"
            "Skills snapshot:\n"
            f"{self.skills.list_for_model()}\n"
        )

        planner_agent = await self._make_agent(
            name="DelegationPlanner",
            instructions=(
                "You are a planning agent that decomposes objectives into parallelizable subtasks. "
                "Return strict JSON only."
            ),
            model_role="planner",
            policy_role="planner",
        )
        raw_plan = self._extract_text(await planner_agent.run(planner_prompt))
        parsed = self._parse_plan(raw_plan, objective=objective, max_workers=max_workers)
        if parsed.subtasks and self._is_valid_plan_schema(parsed, max_workers=max_workers):
            return parsed
        return self._fallback_plan(objective=objective, max_workers=max_workers)

    async def _execute_plan(self, plan: DelegationPlan, max_workers: int) -> list[DelegationResult]:
        engine = TaskEngine(max_workers=max_workers, timeout_s=90.0, retries=1)
        specs: list[TaskSpec] = []
        for subtask in plan.subtasks:
            async def run_for_subtask(s: DelegationSubtask = subtask) -> str:
                role_instructions = ROLE_INSTRUCTIONS.get(s.role, ROLE_INSTRUCTIONS["generalist"])
                worker_agent = await self._make_agent(
                    name=f"{s.role.title()}Worker{s.subtask_id}",
                    instructions=(
                        f"{role_instructions}\n"
                        "You are one worker in a parallel delegation pipeline. "
                        "Be explicit about assumptions and hand-off details."
                    ),
                    model_role=s.role,
                    policy_role=s.role,
                )
                worker_prompt = (
                    f"Top-level objective:\n{plan.objective}\n\n"
                    f"Assigned subtask ({s.role}):\n{s.task}\n\n"
                    "Deliver output with headings:\n"
                    "1) Findings\n2) Proposed Actions\n3) Hand-off Notes\n"
                )
                output = self._extract_text(await worker_agent.run(worker_prompt))
                if self._is_low_signal_output(output):
                    raise ValueError("low-signal worker output")
                return output

            specs.append(TaskSpec(subtask.subtask_id, subtask.role, subtask.task, run_for_subtask))

        task_results: list[TaskResult] = await engine.execute(specs)
        by_id = {s.subtask_id: s for s in plan.subtasks}
        return [
            DelegationResult(
                subtask=by_id[result.subtask_id],
                output=result.output if result.error_kind is None else f"[{result.error_kind}] {result.output}",
                status=result.status,
                error_kind=result.error_kind,
            )
            for result in task_results
        ]

    def _save_run(self, run_id: str, payload: dict[str, Any]) -> None:
        target = self._runs_dir / f"{run_id}.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self._runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            try:
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        return rows

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        path = self._runs_dir / f"{run_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    async def resume_run(self, run_id: str, max_workers: int = 3) -> str:
        payload = self.get_run(run_id)
        if payload is None:
            return f"Run not found: {run_id}"
        if payload.get("status") == "completed":
            return f"Run {run_id} is already completed."
        objective = str(payload.get("objective", "")).strip()
        if not objective:
            return f"Run {run_id} has no objective."
        return await self.delegate_parallel(objective=objective, max_workers=max_workers, run_id=run_id)

    def abort_run(self, run_id: str) -> str:
        payload = self.get_run(run_id)
        if payload is None:
            return f"Run not found: {run_id}"
        current_status = str(payload.get("status", ""))
        if current_status in {"completed", "failed", "aborted"}:
            return f"Run {run_id} is already terminal: {current_status}"
        self._aborted_runs.add(run_id)
        payload["status"] = "aborted"
        self._save_run(run_id, payload)
        return f"Abort requested for run {run_id}"

    async def retry_run(self, run_id: str, max_workers: int = 3, failed_only: bool = True) -> str:
        payload = self.get_run(run_id)
        if payload is None:
            return f"Run not found: {run_id}"
        objective = str(payload.get("objective", "")).strip()
        if not objective:
            return f"Run {run_id} has no objective."
        if not failed_only:
            return await self.delegate_parallel(objective=objective, max_workers=max_workers, run_id=run_id)
        plan_items = payload.get("plan", [])
        result_items = payload.get("results", [])
        failed_ids = {int(item.get("subtask", {}).get("subtask_id")) for item in result_items if str(item.get("status")) != "ok"}
        if not failed_ids:
            return f"Run {run_id} has no failed subtasks to retry."
        subtasks: list[DelegationSubtask] = []
        for item in plan_items:
            subtask_id = int(item.get("subtask_id", 0))
            if subtask_id in failed_ids:
                subtasks.append(DelegationSubtask(subtask_id=subtask_id, role=str(item.get("role", "generalist")), task=str(item.get("task", "")).strip()))
        if not subtasks:
            return f"Run {run_id} has no retryable failed subtasks."
        plan = DelegationPlan(objective=objective, subtasks=subtasks)
        self._save_run(run_id, {"run_id": run_id, "objective": objective, "status": "running", "plan": [asdict(s) for s in plan.subtasks], "results": []})
        results = await self._execute_plan(plan=plan, max_workers=max_workers)
        has_errors = any(item.status != "ok" for item in results)
        has_success = any(item.status == "ok" for item in results)
        status = "failed" if has_errors and not has_success else ("partial" if has_errors else "completed")
        self._save_run(run_id, {"run_id": run_id, "objective": objective, "status": status, "plan": [asdict(s) for s in plan.subtasks], "results": [asdict(r) for r in results]})
        synthesis = await self._synthesize_results(plan=plan, results=results)
        return f"Run ID: {run_id}\nRetried failed subtasks only.\n{synthesis}"

    @staticmethod
    def summarize_run(payload: dict[str, Any]) -> dict[str, Any]:
        result_items = payload.get("results", [])
        error_counts: dict[str, int] = {}
        ok_count = 0
        error_count = 0
        for item in result_items:
            status = str(item.get("status", ""))
            if status == "ok":
                ok_count += 1
                continue
            error_count += 1
            kind = str(item.get("error_kind") or "unknown")
            error_counts[kind] = error_counts.get(kind, 0) + 1
        return {
            "run_id": payload.get("run_id"),
            "status": payload.get("status"),
            "objective": payload.get("objective"),
            "ok_count": ok_count,
            "error_count": error_count,
            "error_counts": error_counts,
            "subtasks": len(payload.get("plan", [])),
        }

    async def _synthesize_results(self, plan: DelegationPlan, results: list[DelegationResult]) -> str:
        synth_agent = await self._make_agent(
            name="DelegationSynthesizer",
            instructions=(
                "You synthesize outputs from parallel workers into one actionable response. "
                "Resolve conflicts, call out risks, and provide ordered next steps."
            ),
            model_role="synthesizer",
            policy_role="synthesizer",
        )
        worker_blocks: list[str] = []
        for result in results:
            worker_blocks.append(
                f"Worker {result.subtask.subtask_id} ({result.subtask.role}, {result.status})\n"
                f"Task: {result.subtask.task}\n"
                f"Output:\n{result.output}\n"
            )
        prompt = (
            f"Objective:\n{plan.objective}\n\n"
            "Worker outputs:\n"
            f"{'\n'.join(worker_blocks)}\n"
            "Return one integrated response with these sections:\n"
            "Summary\nExecution Plan\nRisks\nVerification\n"
        )
        return self._extract_text(await synth_agent.run(prompt))

    def _parse_plan(self, raw_plan: str, objective: str, max_workers: int) -> DelegationPlan:
        payload = self._extract_json(raw_plan)
        if payload is None:
            return DelegationPlan(objective=objective, subtasks=[])
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return DelegationPlan(objective=objective, subtasks=[])

        if not isinstance(data, dict):
            return DelegationPlan(objective=objective, subtasks=[])
        raw_subtasks = data.get("subtasks")
        if not isinstance(raw_subtasks, list):
            return DelegationPlan(objective=objective, subtasks=[])

        subtasks: list[DelegationSubtask] = []
        for raw_item in raw_subtasks[:max_workers]:
            if not isinstance(raw_item, dict):
                continue
            task = str(raw_item.get("task", "")).strip()
            if not task:
                continue
            role = self._normalize_role(str(raw_item.get("role", "generalist")))
            if role not in ROLE_INSTRUCTIONS:
                role = "generalist"
            subtasks.append(
                DelegationSubtask(
                    subtask_id=len(subtasks) + 1,
                    role=role,
                    task=task,
                )
            )

        return DelegationPlan(objective=objective, subtasks=subtasks)

    def _is_valid_plan_schema(self, plan: DelegationPlan, max_workers: int) -> bool:
        if not plan.subtasks:
            return False
        if len(plan.subtasks) > max_workers:
            return False
        seen_ids: set[int] = set()
        for item in plan.subtasks:
            if item.subtask_id in seen_ids:
                return False
            seen_ids.add(item.subtask_id)
            if item.role not in ROLE_INSTRUCTIONS:
                return False
            if not item.task.strip():
                return False
        return True

    @staticmethod
    def _is_low_signal_output(text: str) -> bool:
        stripped = text.strip()
        if len(stripped) < 60:
            return True
        lowered = stripped.lower()
        weak_markers = [
            "i can't",
            "i cannot",
            "not enough information",
            "as an ai",
            "unable to",
            "no skills loaded",
        ]
        return any(marker in lowered for marker in weak_markers)

    @staticmethod
    def _extract_json(text: str) -> str | None:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped

        start = stripped.find("{")
        while start != -1:
            depth = 0
            for index in range(start, len(stripped)):
                char = stripped[index]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return stripped[start : index + 1]
            start = stripped.find("{", start + 1)
        return None

    def _fallback_plan(self, objective: str, max_workers: int) -> DelegationPlan:
        templates = [
            ("researcher", "Identify constraints, external dependencies, and unknowns for the objective."),
            ("coder", "Draft the implementation approach and concrete change list."),
            ("reviewer", "Assess risk, regressions, and verification strategy."),
        ]
        subtasks: list[DelegationSubtask] = []
        for role, task in templates[:max_workers]:
            subtasks.append(
                DelegationSubtask(
                    subtask_id=len(subtasks) + 1,
                    role=role,
                    task=f"{task} Objective: {objective}",
                )
            )
        if not subtasks:
            subtasks.append(
                DelegationSubtask(
                    subtask_id=1,
                    role="generalist",
                    task=objective,
                )
            )
        return DelegationPlan(objective=objective, subtasks=subtasks)

    def list_skills(self) -> str:
        return self.skills.list_for_model()

    def read_skill(self, name: str) -> str:
        doc = self.skills.get(name)
        if doc is None:
            return f"Skill '{name}' not found."
        return doc.body

    def search_skills(self, query: str) -> str:
        docs = self.skills.search(query)
        if not docs:
            return "No matching skills found."
        return "\n".join(f"- {doc.name}: {doc.preview}" for doc in docs)

    def get_memory(self) -> str:
        return self.memory_provider.snapshot()

    def memory_search(self, query: str, top_k: int = 5) -> str:
        rows = self.memory_provider.search(query, top_k)
        if not rows:
            return "No memory matches."
        return "\n".join(f"- [{r.source}/{r.role}] {r.text[:200]}" for r in rows)

    def routing_explain(self, text: str) -> str:
        return str(self.routing_policy.decide(text, self.list_runs(limit=10)))

    def add_memory(self, note: str) -> None:
        self.memory.add_memory_note(note)

    def add_user_pref(self, note: str) -> None:
        self.memory.add_user_note(note)

    @staticmethod
    def _extract_text(response: Any) -> str:
        if isinstance(response, str):
            return response
        if hasattr(response, "text"):
            text = getattr(response, "text")
            if isinstance(text, str) and text.strip():
                return text
        if hasattr(response, "output_text"):
            output_text = getattr(response, "output_text")
            if isinstance(output_text, str) and output_text.strip():
                return output_text
        return str(response)
