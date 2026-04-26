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


def _silent_progress(iterable, **_kwargs):
    """Pass-through replacement for tqdm so F5-TTS doesn't print progress bars."""
    return iterable


def _silent_info(*_args, **_kwargs) -> None:
    pass


class TTS:
    """F5-TTS voice cloning. Each call clones the configured ref voice for new text."""

    SAMPLE_RATE = 24000  # F5-TTS Base outputs 24 kHz; updated after first synth

    def __init__(self, cfg: TTSCfg):
        from f5_tts.api import F5TTS

        self.cfg = cfg
        if not cfg.ref_text.strip():
            raise ValueError("tts.ref_text is empty. It must be the transcript of ref_audio.")
        if not Path(cfg.ref_audio).is_file():
            raise FileNotFoundError(
                f"tts.ref_audio not found: {cfg.ref_audio}. "
                "Put the reference voice clip there or change the path in config.yaml."
            )

        # Ampere+ matmul win: trade a touch of fp32 precision for ~10-20% speed.
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        log.info("Loading F5-TTS (%s)...", cfg.model)
        self.model = F5TTS(model=cfg.model, device=cfg.device)

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
        wav, sr, _ = self.model.infer(
            ref_file=self.cfg.ref_audio,
            ref_text=self.cfg.ref_text,
            gen_text=text,
            nfe_step=self.cfg.nfe_step,
            cfg_strength=self.cfg.cfg_strength,
            speed=self.cfg.speed,
            cross_fade_duration=self.cfg.cross_fade_duration,
            seed=self.cfg.seed,
            show_info=_silent_info,
            progress=_silent_progress,
        )
        self.SAMPLE_RATE = int(sr)
        if hasattr(wav, "detach"):  # torch.Tensor
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
