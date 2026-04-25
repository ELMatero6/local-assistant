from __future__ import annotations

import base64
import json
import logging
from collections import deque
from typing import AsyncIterator

import httpx

from .config import LLMCfg
from .memory import MemoryStore


MEMORY_HITS_PER_TURN = 4
MAX_TOOL_HOPS = 6

chat = logging.getLogger("assistant.chat")


class OllamaClient:
    """Streams chat completions from a local Ollama server.

    - Always injects identity.md and notes.md into the system prompt.
    - Pre-searches memory.md for the user's turn and injects top hits.
    - Exposes notes/memory tools to the model and handles the call/respond loop.
    """

    def __init__(self, cfg: LLMCfg, memory: MemoryStore):
        self.cfg = cfg
        self.memory = memory
        self.history: deque[dict] = deque(maxlen=cfg.context_messages)
        self._client = httpx.AsyncClient(base_url=cfg.host, timeout=httpx.Timeout(120.0, connect=5.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    def needs_vision(self, text: str) -> bool:
        low = text.lower()
        return any(kw in low for kw in self.cfg.vision_keywords)

    def _build_system_prompt(self, user_text: str) -> str:
        parts: list[str] = []
        if self.cfg.system_prompt:
            parts.append(self.cfg.system_prompt.strip())

        identity = self.memory.read_identity().strip()
        if identity:
            parts.append("# Your identity\n" + identity)
        else:
            parts.append(
                "# Your identity\n"
                "You have no identity yet. On your first meaningful interaction, "
                "use the `identity_write` tool to set one."
            )

        notes = self.memory.read_notes().strip()
        if notes:
            parts.append("# Your scratchpad (notes.md)\n" + notes)

        hits = self.memory.search_memory(user_text, k=MEMORY_HITS_PER_TURN)
        if hits:
            joined = "\n\n---\n\n".join(c for c, _ in hits)
            parts.append("# Relevant long-term memory\n" + joined)

        parts.append(
            "You can call tools to read and edit your identity, notes, and long-term memory. "
            "Save anything you'll want to recall later (preferences, facts about the user, "
            "ongoing context) to memory.md via `memory_append`."
        )
        return "\n\n".join(parts)

    async def stream_reply(
        self,
        user_text: str,
        image_jpeg: bytes | None = None,
    ) -> AsyncIterator[str]:
        user_msg: dict = {"role": "user", "content": user_text}
        if image_jpeg is not None:
            user_msg["images"] = [base64.b64encode(image_jpeg).decode("ascii")]

        messages: list[dict] = [{"role": "system", "content": self._build_system_prompt(user_text)}]
        messages.extend(self.history)
        messages.append(user_msg)

        tools = self.memory.tool_definitions()
        spoken: list[str] = []

        for _ in range(MAX_TOOL_HOPS):
            assistant_content = ""
            tool_calls: list[dict] = []

            payload = {
                "model": self.cfg.model,
                "messages": messages,
                "stream": True,
                "tools": tools,
            }

            async with self._client.stream("POST", "/api/chat", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    msg = obj.get("message") or {}
                    token = msg.get("content", "")
                    if token:
                        assistant_content += token
                        spoken.append(token)
                        yield token
                    if msg.get("tool_calls"):
                        tool_calls.extend(msg["tool_calls"])
                    if obj.get("done"):
                        break

            if not tool_calls:
                break

            messages.append({
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                result = self.memory.dispatch(name, args)
                ok = not str(result).startswith("error:")
                chat.info(
                    "tool: %s(%s) %s %s",
                    name,
                    _summarize_args(args),
                    "->" if ok else "x",
                    _summarize_result(result, ok),
                )
                messages.append({"role": "tool", "content": result})

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": "".join(spoken)})


def _summarize_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v).replace("\n", " ")
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(f"{k}={s!r}" if isinstance(v, str) else f"{k}={s}")
    return ", ".join(parts)


def _summarize_result(result: str, ok: bool) -> str:
    s = str(result).replace("\n", " ")
    if not ok:
        return s if len(s) <= 80 else s[:77] + "..."
    if s == "ok":
        return "ok"
    return f"{len(s)} chars"
