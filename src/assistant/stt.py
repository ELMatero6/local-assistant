from __future__ import annotations

import asyncio

import numpy as np
import torch
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad

from .audio import to_float32
from .config import STTCfg


class Recorder:
    """Records from the mic queue after wake word, gated by Silero VAD.

    Returns a float32 mono PCM array at 16 kHz containing the spoken utterance.
    """

    FRAME_SAMPLES = 1280  # match wake-detector frame size; 80 ms @ 16 kHz
    VAD_WINDOW = 512      # silero-vad expects 512-sample windows at 16 kHz

    def __init__(self, cfg: STTCfg, sample_rate: int = 16000):
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.vad = load_silero_vad()

    async def record_utterance(self, mic_queue: asyncio.Queue[np.ndarray]) -> np.ndarray:
        max_frames = int(self.cfg.max_utterance_sec * self.sample_rate / self.FRAME_SAMPLES)
        silence_frames_needed = int(self.cfg.silence_timeout_sec * self.sample_rate / self.FRAME_SAMPLES)

        collected: list[np.ndarray] = []
        silent_run = 0
        heard_speech = False

        for _ in range(max_frames):
            frame = await mic_queue.get()
            collected.append(frame)
            if self._frame_has_speech(frame):
                heard_speech = True
                silent_run = 0
            else:
                silent_run += 1
            if heard_speech and silent_run >= silence_frames_needed:
                break

        pcm = np.concatenate(collected) if collected else np.zeros(0, dtype=np.int16)
        return to_float32(pcm)

    def _frame_has_speech(self, frame_int16: np.ndarray) -> bool:
        # Silero needs 512-sample windows. Take max probability across windows.
        f32 = to_float32(frame_int16)
        probs = []
        for start in range(0, len(f32) - self.VAD_WINDOW + 1, self.VAD_WINDOW):
            window = torch.from_numpy(f32[start:start + self.VAD_WINDOW])
            probs.append(self.vad(window, self.sample_rate).item())
        if not probs:
            return False
        return max(probs) > self.cfg.vad_threshold


class STT:
    def __init__(self, cfg: STTCfg):
        self.cfg = cfg
        self.model = WhisperModel(
            cfg.model,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )

    def transcribe(self, pcm_f32_16k: np.ndarray) -> str:
        if pcm_f32_16k.size == 0:
            return ""
        segments, _ = self.model.transcribe(
            pcm_f32_16k,
            language=self.cfg.language,
            vad_filter=self.cfg.whisper_vad_filter,
            beam_size=1,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()
