from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


_TOKEN = re.compile(r"\w+")
_HEADING = re.compile(r"^(#{1,6})\s+\S")


def _tokenize(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


def _chunk_markdown(text: str) -> list[str]:
    """Split on `## ` (and deeper) headings; each chunk keeps its heading.

    Falls back to paragraph splits if the document has no headings.
    """
    if not text.strip():
        return []
    if not any(_HEADING.match(line) for line in text.splitlines()):
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if _HEADING.match(line) and current:
            chunks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append(current)
    return ["\n".join(c).strip() for c in chunks if "".join(c).strip()]


class MemoryStore:
    """Three persistent markdown files the model owns.

    - identity.md: the model's self-description, always injected into the system prompt.
    - notes.md:    short scratchpad, always injected (small, full content).
    - memory.md:   long-term store; only top-K BM25 hits per user turn are injected,
                   and the model can search/edit it via tool calls.
    """

    def __init__(self, base_dir: Path):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.identity_path = self.base / "identity.md"
        self.notes_path = self.base / "notes.md"
        self.memory_path = self.base / "memory.md"
        for p in (self.identity_path, self.notes_path, self.memory_path):
            if not p.exists():
                p.write_text("")
        self._index_cache: tuple[float, BM25Okapi, list[str]] | None = None

    # ---- file ops ---------------------------------------------------------

    def read_identity(self) -> str:
        return self.identity_path.read_text()

    def write_identity(self, content: str) -> None:
        self.identity_path.write_text(content.strip() + "\n")

    def read_notes(self) -> str:
        return self.notes_path.read_text()

    def append_note(self, text: str) -> None:
        with self.notes_path.open("a") as f:
            f.write(text.rstrip() + "\n")

    def overwrite_notes(self, content: str) -> None:
        self.notes_path.write_text(content.rstrip() + "\n")

    def append_memory(self, text: str) -> None:
        with self.memory_path.open("a") as f:
            sep = "" if not self.memory_path.read_text().endswith("\n\n") else ""
            f.write(sep + text.rstrip() + "\n\n")
        self._index_cache = None

    def edit_memory(self, old: str, new: str) -> int:
        """Exact-string replace. Errors if 0 or >1 matches; returns 1 on success."""
        body = self.memory_path.read_text()
        count = body.count(old)
        if count == 0:
            raise ValueError("edit_memory: old string not found")
        if count > 1:
            raise ValueError(f"edit_memory: old string matches {count} times; make it unique")
        self.memory_path.write_text(body.replace(old, new, 1))
        self._index_cache = None
        return 1

    def overwrite_memory(self, content: str) -> None:
        self.memory_path.write_text(content.rstrip() + "\n")
        self._index_cache = None

    # ---- search -----------------------------------------------------------

    def _index(self) -> tuple[BM25Okapi, list[str]]:
        mtime = self.memory_path.stat().st_mtime
        if self._index_cache and self._index_cache[0] == mtime:
            return self._index_cache[1], self._index_cache[2]
        chunks = _chunk_markdown(self.memory_path.read_text())
        if not chunks:
            bm25 = BM25Okapi([[""]])
            self._index_cache = (mtime, bm25, [])
            return bm25, []
        bm25 = BM25Okapi([_tokenize(c) for c in chunks])
        self._index_cache = (mtime, bm25, chunks)
        return bm25, chunks

    def search_memory(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        bm25, chunks = self._index()
        if not chunks:
            return []
        scores = bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [(c, float(s)) for c, s in ranked[:k] if s > 0]

    # ---- tool dispatch ----------------------------------------------------

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            _tool("identity_read", "Read your current identity document.", {}),
            _tool(
                "identity_write",
                "Replace your identity document with new content. Use sparingly.",
                {"content": {"type": "string", "description": "Full new identity in markdown."}},
                required=["content"],
            ),
            _tool(
                "notes_append",
                "Append a short note to your scratchpad (notes.md).",
                {"text": {"type": "string", "description": "Note text. Keep it brief."}},
                required=["text"],
            ),
            _tool(
                "notes_overwrite",
                "Replace the entire scratchpad. Use to prune obsolete notes.",
                {"content": {"type": "string", "description": "New full notes content."}},
                required=["content"],
            ),
            _tool(
                "memory_search",
                "Search long-term memory and return the top-K matching chunks.",
                {
                    "query": {"type": "string", "description": "Search query."},
                    "k": {"type": "integer", "description": "Max results (default 5)."},
                },
                required=["query"],
            ),
            _tool(
                "memory_append",
                "Append a new entry to long-term memory. Prefer a `## Heading` first line so it indexes well.",
                {"text": {"type": "string", "description": "Entry to append."}},
                required=["text"],
            ),
            _tool(
                "memory_edit",
                "Replace an exact substring inside long-term memory. The substring must be unique.",
                {
                    "old": {"type": "string", "description": "Exact text currently in memory.md."},
                    "new": {"type": "string", "description": "Replacement text."},
                },
                required=["old", "new"],
            ),
        ]

    def dispatch(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "identity_read":
                return self.read_identity() or "(empty)"
            if name == "identity_write":
                self.write_identity(args["content"])
                return "ok"
            if name == "notes_append":
                self.append_note(args["text"])
                return "ok"
            if name == "notes_overwrite":
                self.overwrite_notes(args["content"])
                return "ok"
            if name == "memory_search":
                hits = self.search_memory(args["query"], int(args.get("k", 5)))
                return json.dumps(
                    [{"score": round(s, 3), "chunk": c} for c, s in hits],
                    ensure_ascii=False,
                )
            if name == "memory_append":
                self.append_memory(args["text"])
                return "ok"
            if name == "memory_edit":
                self.edit_memory(args["old"], args["new"])
                return "ok"
            return f"error: unknown tool {name}"
        except Exception as e:
            return f"error: {e}"


def _tool(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }
