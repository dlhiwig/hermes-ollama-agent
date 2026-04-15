from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass(frozen=True, slots=True)
class SkillDoc:
    name: str
    path: Path
    title: str
    body: str

    @property
    def preview(self) -> str:
        lines = [line.strip() for line in self.body.splitlines() if line.strip()]
        if not lines:
            return "(empty skill)"
        return lines[0][:200]


class SkillLibrary:
    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = skills_dir
        self._docs: dict[str, SkillDoc] = {}

    def load(self) -> None:
        self._docs.clear()
        if not self.skills_dir.exists():
            return
        for path in sorted(self.skills_dir.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            title = self._extract_title(text) or path.stem.replace("-", " ")
            name = path.stem.lower()
            self._docs[name] = SkillDoc(name=name, path=path, title=title, body=text)

    @staticmethod
    def _extract_title(text: str) -> str | None:
        for line in text.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return None

    def names(self) -> list[str]:
        return sorted(self._docs.keys())

    def list_for_model(self) -> str:
        if not self._docs:
            return "No skills loaded."
        chunks: list[str] = []
        for doc in self._docs.values():
            chunks.append(f"- {doc.name}: {doc.preview}")
        return "\n".join(chunks)

    def get(self, name: str) -> SkillDoc | None:
        key = name.strip().lower()
        return self._docs.get(key)

    def search(self, query: str, limit: int = 5) -> list[SkillDoc]:
        norm = _normalize(query)
        if not norm:
            return []
        scored: list[tuple[int, SkillDoc]] = []
        for doc in self._docs.values():
            corpus = _normalize(f"{doc.title}\n{doc.body}")
            score = sum(1 for token in norm.split() if token in corpus)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:limit]]

    def __iter__(self) -> Iterable[SkillDoc]:
        return iter(self._docs.values())
