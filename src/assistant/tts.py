from __future__ import annotations

import atexit
import logging
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .config import TTSCfg

log = logging.getLogger("assistant.tts")

_SENTENCE_END = re.compile(r"([\.!\?])\s+")


def _silent_info(*_args, **_kwargs) -> None:
    pass


def _prepare_ref(audio_path: str, ref_text: str, max_sec: float) -> tuple[str, str]:
    """Load ref_audio; if longer than max_sec, trim and proportionally truncate the text.

    Returns (path, text). If trimmed, path is a temp wav file registered for
    deletion at exit; otherwise path is the original.
    """
    data, sr = sf.read(audio_path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo → mono
    duration = len(data) / sr

    log.info("ref_audio: %.1fs @ %d Hz — %s", duration, sr, audio_path)

    if duration <= max_sec:
        return audio_path, ref_text.strip()

    log.warning(
        "ref_audio is %.1fs which is longer than ref_audio_max_sec=%.0fs. "
        "Trimming to %.0fs for better F5-TTS boundary detection. "
        "For best quality use a %.0fs clip with an exact matching transcript.",
        duration, max_sec, max_sec, max_sec,
    )

    trimmed_data = data[: int(max_sec * sr)]

    # Proportionally truncate the transcript by word count.
    words = ref_text.split()
    keep = max(1, int(len(words) * max_sec / duration))
    trimmed_text = " ".join(words[:keep])

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, trimmed_data, sr)
    tmp.close()
    atexit.register(os.unlink, tmp.name)

    log.info("Trimmed ref text (%d→%d words): %s…", len(words), keep, trimmed_text[:60])
    return tmp.name, trimmed_text


class TTS:
    """F5-TTS voice cloning. Each call clones the configured ref voice for new text."""

    SAMPLE_RATE = 24000  # F5-TTS Base outputs 24 kHz; updated after first synth

    def __init__(self, cfg: TTSCfg):
        # SWivid's F5-TTS exposes F5TTS in f5_tts.api; older/fallback builds may skip the submodule.
        try:
            from f5_tts.api import F5TTS
        except ImportError:
            from f5_tts import F5TTS  # type: ignore[no-redef]

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
            ref_file=self._ref_audio,
            ref_text=self._ref_text,
            gen_text=text,
            nfe_step=self.cfg.nfe_step,
            cfg_strength=self.cfg.cfg_strength,
            speed=self.cfg.speed,
            cross_fade_duration=self.cfg.cross_fade_duration,
            seed=self.cfg.seed if self.cfg.seed is not None else -1,
            show_info=_silent_info,
            progress=None,
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
