from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


@dataclass(slots=True)
class MemoryStore:
    root: Path
    memory_char_budget: int
    user_char_budget: int
    max_turn_chars: int

    @property
    def memory_file(self) -> Path:
        return self.root / "MEMORY.md"

    @property
    def user_file(self) -> Path:
        return self.root / "USER.md"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.memory_file.exists():
            self.memory_file.write_text(
                "# MEMORY\n\n"
                "Keep durable project-level notes here.\n",
                encoding="utf-8",
            )
        if not self.user_file.exists():
            self.user_file.write_text(
                "# USER\n\n"
                "Add user preferences and constraints here.\n",
                encoding="utf-8",
            )

    def read_memory(self) -> str:
        self.ensure()
        return self.memory_file.read_text(encoding="utf-8")

    def read_user(self) -> str:
        self.ensure()
        return self.user_file.read_text(encoding="utf-8")

    def add_memory_note(self, note: str) -> None:
        self._append_with_budget(
            self.memory_file,
            f"\n## {_utc_now()}\n{note.strip()}\n",
            self.memory_char_budget,
        )

    def add_user_note(self, note: str) -> None:
        self._append_with_budget(
            self.user_file,
            f"\n## {_utc_now()}\n{note.strip()}\n",
            self.user_char_budget,
        )

    def append_turn(self, user_text: str, assistant_text: str) -> None:
        user_text = user_text.strip()
        assistant_text = assistant_text.strip()
        compact_user = user_text[: self.max_turn_chars]
        compact_assistant = assistant_text[: self.max_turn_chars]
        self._append_with_budget(
            self.memory_file,
            (
                f"\n### Turn {_utc_now()}\n"
                f"- User: {compact_user}\n"
                f"- Assistant: {compact_assistant}\n"
            ),
            self.memory_char_budget,
        )

    def context_block(self) -> str:
        memory = self.read_memory().strip()
        user = self.read_user().strip()
        return (
            "Persistent context (authoritative):\n"
            f"{memory}\n\n"
            "User profile and preferences:\n"
            f"{user}\n"
        )

    @staticmethod
    def _append_with_budget(path: Path, text: str, budget: int) -> None:
        existing = path.read_text(encoding="utf-8")
        merged = f"{existing.rstrip()}\n{text}".strip() + "\n"
        if len(merged) > budget:
            merged = merged[-budget:]
            first_newline = merged.find("\n")
            if first_newline != -1:
                merged = merged[first_newline + 1 :]
        path.write_text(merged, encoding="utf-8")
