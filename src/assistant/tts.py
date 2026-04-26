from __future__ import annotations

import atexit
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import TTSCfg

log = logging.getLogger("assistant.tts")

_SENTENCE_END = re.compile(r"([\.!\?])\s+")


def _prepare_ref(audio_path: str, ref_text: str, max_sec: float) -> tuple[str, str]:
    """Load ref_audio; trim to max_sec if longer and proportionally truncate the transcript."""
    data, sr = sf.read(audio_path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    duration = len(data) / sr
    log.info("ref_audio: %.1fs @ %d Hz — %s", duration, sr, audio_path)

    if duration <= max_sec:
        return audio_path, ref_text.strip()

    log.warning(
        "ref_audio is %.1fs (> %.0fs max); trimming. "
        "For best quality use a %.0fs clip with an exact matching transcript.",
        duration, max_sec, max_sec,
    )
    trimmed = data[: int(max_sec * sr)]
    words = ref_text.split()
    keep = max(1, int(len(words) * max_sec / duration))
    trimmed_text = " ".join(words[:keep])

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, trimmed, sr)
    tmp.close()
    atexit.register(os.unlink, tmp.name)
    log.info("Trimmed ref text (%d→%d words): %s…", len(words), keep, trimmed_text[:60])
    return tmp.name, trimmed_text


class TTS:
    """Qwen3-TTS voice cloning via qwen_tts.Qwen3TTSModel."""

    SAMPLE_RATE = 22050  # updated after first synth from returned sr

    def __init__(self, cfg: TTSCfg):
        import torch
        from qwen_tts import Qwen3TTSModel

        self.cfg = cfg
        if not cfg.ref_text.strip():
            raise ValueError("tts.ref_text is empty. It must be the transcript of ref_audio.")
        if not Path(cfg.ref_audio).is_file():
            raise FileNotFoundError(
                f"tts.ref_audio not found: {cfg.ref_audio}. "
                "Put the reference voice clip there or change the path in config.yaml."
            )

        self._ref_audio, self._ref_text = _prepare_ref(
            cfg.ref_audio, cfg.ref_text, max_sec=cfg.ref_audio_max_sec
        )

        # "cuda" → "cuda:0"; explicit "cuda:N" or "cpu" passed through unchanged.
        device_map = (cfg.device + ":0") if cfg.device == "cuda" else cfg.device

        log.info("Loading Qwen3-TTS (%s)...", cfg.model)
        self.model = Qwen3TTSModel.from_pretrained(
            cfg.model,
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # Serializes synth calls so a cancelled-but-still-running thread can't
        # overlap with the next turn's synthesis (which would segfault the GPU model).
        self._lock = threading.Lock()

        if cfg.prewarm:
            log.info("Pre-warming TTS (first synth is always slowest)...")
            t0 = time.perf_counter()
            _ = self.synth("Initializing.")
            log.info("Pre-warm took %.2fs", time.perf_counter() - t0)

    def synth(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        with self._lock:
            return self._synth_locked(text)

    def _synth_locked(self, text: str) -> np.ndarray:
        t0 = time.perf_counter()
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=self.cfg.language,
            ref_audio=self._ref_audio,
            ref_text=self._ref_text,
        )
        self.SAMPLE_RATE = int(sr)
        wav = wavs[0]
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        audio = np.asarray(wav, dtype=np.float32).reshape(-1)
        elapsed = time.perf_counter() - t0
        rt = audio.size / self.SAMPLE_RATE if self.SAMPLE_RATE else 0
        log.debug(
            "synth %.2fs -> %.2fs audio (%.2fx realtime) for %d chars",
            elapsed, rt, rt / elapsed if elapsed else 0, len(text),
        )
        return np.clip(audio, -1.0, 1.0)


def split_sentences_streaming(buffer: str) -> tuple[list[str], str]:
    """Pull complete sentences out of a growing buffer; return (sentences, remainder)."""
    out: list[str] = []
    last = 0
    for m in _SENTENCE_END.finditer(buffer):
        out.append(buffer[last:m.end()].strip())
        last = m.end()
    return out, buffer[last:]
