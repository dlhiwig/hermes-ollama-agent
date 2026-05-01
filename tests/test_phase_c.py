from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from hermes_ollama_agent.config import HermesConfig
from hermes_ollama_agent.memory_provider import ChromaMemoryProvider, MemoryRecord
from hermes_ollama_agent.runtime import DelegationPlan, DelegationResult, DelegationSubtask, HermesRuntime
from hermes_ollama_agent.task_engine import TaskEngine, TaskSpec


class TaskEngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_classification(self) -> None:
        async def slow() -> str:
            await asyncio.sleep(0.05)
            return "ok"

        engine = TaskEngine(max_workers=1, timeout_s=0.001, retries=0)
        out = await engine.execute([TaskSpec(1, "t", "slow", slow)])
        self.assertEqual(out[0].status, "error")
        self.assertEqual(out[0].error_kind, "timeout")

    async def test_retry_then_success(self) -> None:
        state = {"count": 0}

        async def flaky() -> str:
            state["count"] += 1
            if state["count"] == 1:
                raise RuntimeError("connect failed")
            return "ok"

        engine = TaskEngine(max_workers=1, timeout_s=1.0, retries=1)
        out = await engine.execute([TaskSpec(1, "t", "flaky", flaky)])
        self.assertEqual(out[0].status, "ok")
        self.assertEqual(out[0].attempts, 2)


class RuntimeStateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cfg = HermesConfig(
            model_id="mistral",
            base_url="http://127.0.0.1:11434/v1/",
            api_key="ollama",
            state_dir=root / ".hermes",
            skills_dir=root / "skills",
        )
        (self.cfg.skills_dir).mkdir(parents=True, exist_ok=True)
        (self.cfg.skills_dir / "sample.md").write_text("# sample\nx\n", encoding="utf-8")
        self.runtime = HermesRuntime(self.cfg)

        async def fake_plan(objective: str, max_workers: int) -> DelegationPlan:
            return DelegationPlan(objective, [DelegationSubtask(1, "generalist", f"do {objective}")])

        async def fake_exec(plan: DelegationPlan, max_workers: int) -> list[DelegationResult]:
            return [DelegationResult(plan.subtasks[0], "done", "ok")]

        async def fake_synth(plan: DelegationPlan, results: list[DelegationResult]) -> str:
            return "synth"

        self.runtime._build_delegation_plan = fake_plan  # type: ignore[method-assign]
        self.runtime._execute_plan = fake_exec  # type: ignore[method-assign]
        self.runtime._synthesize_results = fake_synth  # type: ignore[method-assign]

    async def asyncTearDown(self) -> None:
        self.tmp.cleanup()

    async def test_delegate_completed_status(self) -> None:
        text = await self.runtime.delegate_parallel("obj", max_workers=1, run_id="r1")
        self.assertIn("Run ID: r1", text)
        run = self.runtime.get_run("r1")
        assert run is not None
        self.assertEqual(run["status"], "completed")

    async def test_abort_marks_terminal(self) -> None:
        self.runtime._save_run("r2", {"run_id": "r2", "objective": "x", "status": "running", "plan": [], "results": []})
        msg = self.runtime.abort_run("r2")
        self.assertIn("Abort requested", msg)
        run = self.runtime.get_run("r2")
        assert run is not None
        self.assertEqual(run["status"], "aborted")

    async def test_retry_failed_only(self) -> None:
        self.runtime._save_run(
            "r3",
            {
                "run_id": "r3",
                "objective": "x",
                "status": "partial",
                "plan": [{"subtask_id": 1, "role": "generalist", "task": "a"}],
                "results": [{"subtask": {"subtask_id": 1}, "status": "error", "output": "bad"}],
            },
        )
        out = await self.runtime.retry_run("r3", max_workers=1, failed_only=True)
        self.assertIn("Retried failed subtasks only.", out)
        run = self.runtime.get_run("r3")
        assert run is not None
        self.assertEqual(run["status"], "completed")

    async def test_failed_terminal_state_when_all_fail(self) -> None:
        async def fake_exec_fail(plan: DelegationPlan, max_workers: int) -> list[DelegationResult]:
            return [DelegationResult(plan.subtasks[0], "[timeout] fail", "error", error_kind="timeout")]

        self.runtime._execute_plan = fake_exec_fail  # type: ignore[method-assign]
        await self.runtime.delegate_parallel("all fail", max_workers=1, run_id="r-fail")
        run = self.runtime.get_run("r-fail")
        assert run is not None
        self.assertEqual(run["status"], "failed")
        summary = self.runtime.summarize_run(run)
        self.assertEqual(summary["error_count"], 1)

    async def test_invalid_planner_schema_falls_back(self) -> None:
        plan = self.runtime._parse_plan('{"subtasks":[{"role":"bad","task":""}]}', "obj", 2)
        self.assertFalse(self.runtime._is_valid_plan_schema(plan, 2))
        fallback = self.runtime._fallback_plan("obj", 2)
        self.assertTrue(self.runtime._is_valid_plan_schema(fallback, 2))

    async def test_low_signal_detection(self) -> None:
        self.assertTrue(self.runtime._is_low_signal_output("ok"))
        self.assertTrue(self.runtime._is_low_signal_output("I cannot proceed without more info."))
        self.assertFalse(
            self.runtime._is_low_signal_output(
                "Findings: constraints captured. Proposed actions: implement, test, and validate with clear handoff notes."
            )
        )


if __name__ == "__main__":
    unittest.main()


class ChromaProviderTests(unittest.TestCase):
    def test_chroma_add_and_search(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        try:
            provider = ChromaMemoryProvider(Path(tmp.name) / "chroma")
            provider.add(
                MemoryRecord(
                    id="r1",
                    timestamp="2026-01-01T00:00:00Z",
                    source="memory_note",
                    role="user",
                    text="vector memory retrieval test phrase alpha",
                    tags=["test"],
                )
            )
            out = provider.search("alpha retrieval", top_k=3)
            self.assertGreaterEqual(len(out), 1)
        finally:
            tmp.cleanup()
