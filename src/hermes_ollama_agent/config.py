from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(slots=True)
class HermesConfig:
    model_id: str = "llama3.2"
    planner_model_id: str | None = None
    researcher_model_id: str | None = None
    coder_model_id: str | None = None
    reviewer_model_id: str | None = None
    synthesizer_model_id: str | None = None
    generalist_model_id: str | None = None
    base_url: str = "http://localhost:11434/v1/"
    api_key: str = "not-needed"
    state_dir: Path = Path(".hermes")
    skills_dir: Path = Path("skills")
    memory_char_budget: int = 6000
    user_char_budget: int = 3000
    max_turn_chars: int = 1200

    @classmethod
    def from_env(cls) -> "HermesConfig":
        def optional_env(name: str) -> str | None:
            value = os.getenv(name)
            if value is None:
                return None
            stripped = value.strip()
            return stripped or None

        return cls(
            model_id=os.getenv("OLLAMA_MODEL", "llama3.2"),
            planner_model_id=optional_env("OLLAMA_MODEL_PLANNER"),
            researcher_model_id=optional_env("OLLAMA_MODEL_RESEARCHER"),
            coder_model_id=optional_env("OLLAMA_MODEL_CODER"),
            reviewer_model_id=optional_env("OLLAMA_MODEL_REVIEWER"),
            synthesizer_model_id=optional_env("OLLAMA_MODEL_SYNTHESIZER"),
            generalist_model_id=optional_env("OLLAMA_MODEL_GENERALIST"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1/"),
            api_key=os.getenv("OLLAMA_API_KEY", "not-needed"),
            state_dir=Path(os.getenv("HERMES_STATE_DIR", ".hermes")),
            skills_dir=Path(os.getenv("HERMES_SKILLS_DIR", "skills")),
            memory_char_budget=int(os.getenv("HERMES_MEMORY_BUDGET", "6000")),
            user_char_budget=int(os.getenv("HERMES_USER_BUDGET", "3000")),
            max_turn_chars=int(os.getenv("HERMES_TURN_BUDGET", "1200")),
        )

    def model_for_role(self, role: str) -> str:
        role_key = role.strip().lower()
        role_map: Mapping[str, str | None] = {
            "planner": self.planner_model_id,
            "researcher": self.researcher_model_id,
            "coder": self.coder_model_id,
            "reviewer": self.reviewer_model_id,
            "synthesizer": self.synthesizer_model_id,
            "generalist": self.generalist_model_id,
            "default": self.model_id,
        }
        return role_map.get(role_key) or self.model_id

    def model_routing_table(self) -> dict[str, str]:
        return {
            "default": self.model_for_role("default"),
            "planner": self.model_for_role("planner"),
            "researcher": self.model_for_role("researcher"),
            "coder": self.model_for_role("coder"),
            "reviewer": self.model_for_role("reviewer"),
            "synthesizer": self.model_for_role("synthesizer"),
            "generalist": self.model_for_role("generalist"),
        }
