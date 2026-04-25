from __future__ import annotations

import base64
import json
from collections import deque
from typing import AsyncIterator

import httpx

from .config import LLMCfg


class OllamaClient:
    """Streams chat completions from a local Ollama server.

    Keeps a short rolling history so follow-up questions have context.
    Supports attaching a JPEG-encoded webcam frame on a single turn for vision.
    """

    def __init__(self, cfg: LLMCfg):
        self.cfg = cfg
        self.history: deque[dict] = deque(maxlen=cfg.context_messages)
        self._client = httpx.AsyncClient(base_url=cfg.host, timeout=httpx.Timeout(120.0, connect=5.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    def needs_vision(self, text: str) -> bool:
        low = text.lower()
        return any(kw in low for kw in self.cfg.vision_keywords)

    async def stream_reply(
        self,
        user_text: str,
        image_jpeg: bytes | None = None,
    ) -> AsyncIterator[str]:
        user_msg: dict = {"role": "user", "content": user_text}
        if image_jpeg is not None:
            user_msg["images"] = [base64.b64encode(image_jpeg).decode("ascii")]

        messages = []
        if self.cfg.system_prompt:
            messages.append({"role": "system", "content": self.cfg.system_prompt})
        messages.extend(self.history)
        messages.append(user_msg)

        payload = {"model": self.cfg.model, "messages": messages, "stream": True}

        full_reply: list[str] = []
        async with self._client.stream("POST", "/api/chat", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                msg = obj.get("message") or {}
                token = msg.get("content", "")
                if token:
                    full_reply.append(token)
                    yield token
                if obj.get("done"):
                    break

        # Persist text-only history (drop the image to keep payloads small).
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": "".join(full_reply)})
