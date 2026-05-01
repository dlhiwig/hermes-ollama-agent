from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class MemoryRecord:
    id: str
    timestamp: str
    source: str
    role: str
    text: str
    tags: list[str]


class MemoryProvider:
    def add(self, record: MemoryRecord) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int) -> list[MemoryRecord]:
        raise NotImplementedError

    def snapshot(self) -> str:
        raise NotImplementedError


class MarkdownMemoryProvider(MemoryProvider):
    def __init__(self, memory_store: Any) -> None:
        self.memory_store = memory_store

    def add(self, record: MemoryRecord) -> None:
        if record.source == "user_pref":
            self.memory_store.add_user_note(record.text)
        else:
            self.memory_store.add_memory_note(f"[{record.source}/{record.role}] {record.text}")

    def search(self, query: str, top_k: int) -> list[MemoryRecord]:
        corpus = self.memory_store.context_block()
        if query.lower() not in corpus.lower():
            return []
        return [
            MemoryRecord(
                id=uuid.uuid4().hex[:12],
                timestamp=_ts(),
                source="memory_snapshot",
                role="system",
                text=corpus[:2000],
                tags=["snapshot"],
            )
        ][:top_k]

    def snapshot(self) -> str:
        return self.memory_store.context_block()


class ChromaMemoryProvider(MemoryProvider):
    def __init__(self, chroma_dir: Path) -> None:
        self.chroma_dir = chroma_dir
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.records_file = self.chroma_dir / "records.jsonl"

    def add(self, record: MemoryRecord) -> None:
        line = json.dumps(asdict(record), ensure_ascii=True)
        with self.records_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def search(self, query: str, top_k: int) -> list[MemoryRecord]:
        q = query.lower().strip()
        if not self.records_file.exists() or not q:
            return []
        scored: list[tuple[int, MemoryRecord]] = []
        for line in self.records_file.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                rec = MemoryRecord(**obj)
            except Exception:
                continue
            text = rec.text.lower()
            score = sum(1 for token in q.split() if token in text)
            if score > 0:
                scored.append((score, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [rec for _, rec in scored[: max(1, top_k)]]

    def snapshot(self) -> str:
        if not self.records_file.exists():
            return "Persistent context (authoritative):\n# MEMORY\n\n(no vector records yet)\n\nUser profile and preferences:\n# USER\n"
        rows = self.records_file.read_text(encoding="utf-8").splitlines()[-20:]
        merged = []
        for line in rows:
            try:
                obj = json.loads(line)
                merged.append(f"- [{obj.get('source')}/{obj.get('role')}] {obj.get('text')}")
            except Exception:
                continue
        return "Persistent context (authoritative):\n# MEMORY\n\n" + "\n".join(merged) + "\n\nUser profile and preferences:\n# USER\n"
