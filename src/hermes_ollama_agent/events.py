from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Event:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_now)


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[Event], None]]] = defaultdict(list)
        self._history: list[Event] = []

    def on(self, name: str, handler: Callable[[Event], None]) -> None:
        self._handlers[name].append(handler)

    def emit(self, name: str, payload: dict[str, Any] | None = None) -> None:
        event = Event(name=name, payload=payload or {})
        self._history.append(event)
        for handler in self._handlers.get(name, []):
            handler(event)

    def recent(self, limit: int = 25) -> list[Event]:
        return self._history[-max(1, limit) :]
