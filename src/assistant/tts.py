from __future__ import annotations

import re

import numpy as np
from kokoro import KPipeline

from .config import TTSCfg

_SENTENCE_END = re.compile(r"([\.!\?])\s+")


class TTS:
    """Kokoro text-to-speech, 24 kHz float32 mono."""

    SAMPLE_RATE = 24000

    def __init__(self, cfg: TTSCfg, lang_code: str = "a"):
        self.cfg = cfg
        self.pipeline = KPipeline(lang_code=lang_code)

    def synth(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        chunks = []
        for _gs, _ps, audio in self.pipeline(text, voice=self.cfg.voice, speed=self.cfg.speed):
            chunks.append(np.asarray(audio, dtype=np.float32))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def split_sentences_streaming(buffer: str) -> tuple[list[str], str]:
    """Pull complete sentences out of a growing buffer; return (sentences, remainder)."""
    out: list[str] = []
    last = 0
    for m in _SENTENCE_END.finditer(buffer):
        out.append(buffer[last:m.end()].strip())
        last = m.end()
    return out, buffer[last:]
