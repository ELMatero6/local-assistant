from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import numpy as np
import torch

from .config import TTSCfg

log = logging.getLogger("assistant.tts")

_SENTENCE_END = re.compile(r"([\.!\?])\s+")
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class TTS:
    """Qwen3-TTS voice cloning. Each call clones the configured ref voice for new text."""

    SAMPLE_RATE = 24000  # updated to the actual rate after the first call

    def __init__(self, cfg: TTSCfg):
        import transformers
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        from qwen_tts import Qwen3TTSModel

        self.cfg = cfg
        if not cfg.ref_text.strip():
            raise ValueError("tts.ref_text is empty. It must be the transcript of ref_audio.")
        if not Path(cfg.ref_audio).is_file():
            raise FileNotFoundError(
                f"tts.ref_audio not found: {cfg.ref_audio}. "
                "Put the reference voice clip there or change the path in config.yaml."
            )

        # Ampere+ matmul win: tradeoff a bit of fp32 precision for ~10-20% speed.
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        log.info("Loading Qwen3-TTS (%s)...", cfg.model_id)
        self.model = Qwen3TTSModel.from_pretrained(
            cfg.model_id,
            device_map=cfg.device,
            dtype=_DTYPES[cfg.dtype],
            attn_implementation=cfg.attn_implementation,
        )

        if cfg.prewarm:
            log.info("Pre-warming TTS (first synth is always slowest)...")
            t0 = time.perf_counter()
            _ = self.synth("Initializing.")
            log.info("Pre-warm took %.2fs", time.perf_counter() - t0)

    def synth(self, text: str) -> np.ndarray:
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        t0 = time.perf_counter()
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=self.cfg.language,
            ref_audio=self.cfg.ref_audio,
            ref_text=self.cfg.ref_text,
        )
        self.SAMPLE_RATE = int(sr)
        audio = np.asarray(wavs[0], dtype=np.float32)
        elapsed = time.perf_counter() - t0
        rt = audio.size / self.SAMPLE_RATE if self.SAMPLE_RATE else 0
        log.debug(
            "synth %.2fs -> %.2fs audio (%.2fx realtime) for %d chars",
            elapsed, rt, rt / elapsed if elapsed else 0, len(text),
        )
        # Defensive clip in case the model outputs something out-of-range.
        return np.clip(audio, -1.0, 1.0)


def split_sentences_streaming(buffer: str) -> tuple[list[str], str]:
    """Pull complete sentences out of a growing buffer; return (sentences, remainder)."""
    out: list[str] = []
    last = 0
    for m in _SENTENCE_END.finditer(buffer):
        out.append(buffer[last:m.end()].strip())
        last = m.end()
    return out, buffer[last:]
