from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict


@dataclass(slots=True)
class WebSource:
    url: str
    title: str
    snippet: str
    content_hash: str


class WebResearchTools:
    def __init__(self, *, timeout_s: float = 8.0, max_bytes: int = 100_000, max_requests: int = 5) -> None:
        self.timeout_s = timeout_s
        self.max_bytes = max_bytes
        self.max_requests = max_requests
        self._used = 0
        self._sources: list[WebSource] = []

    def search(self, query: str, limit: int = 3) -> str:
        if self._used >= self.max_requests:
            return "web_search blocked: request budget exceeded"
        self._used += 1
        q = urllib.parse.quote_plus(query)
        return json.dumps([{"title": f"Search result {i+1}", "url": f"https://example.com/search?q={q}&i={i+1}"} for i in range(max(1, limit))])

    def fetch(self, url: str) -> str:
        if self._used >= self.max_requests:
            return "web_fetch blocked: request budget exceeded"
        self._used += 1
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_s) as resp:
                data = resp.read(self.max_bytes).decode("utf-8", errors="ignore")
        except Exception as exc:
            return f"web_fetch failed: {exc}"
        snippet = data[:500]
        title = url
        source = WebSource(url=url, title=title, snippet=snippet[:200], content_hash=hashlib.sha256(data.encode("utf-8")).hexdigest()[:16])
        self._sources.append(source)
        return snippet

    def cite(self) -> list[dict[str, str]]:
        return [asdict(item) for item in self._sources]
