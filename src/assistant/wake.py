from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
from openwakeword.model import Model
from openwakeword.utils import download_models

from .config import WakeCfg


class WakeDetector:
    """Wraps openWakeWord to fire whenever the configured wake word is detected.

    Consumes 16k mono int16 frames of length 1280 (80 ms) from a shared queue;
    that is the chunk size openWakeWord expects.
    """

    FRAME_SAMPLES = 1280  # 80 ms at 16 kHz

    def __init__(self, cfg: WakeCfg):
        self.cfg = cfg
        # openWakeWord wheels don't include the model files (wake-word .onnx
        # plus the shared melspec / embedding helpers). Download is idempotent.
        if not Path(cfg.model).is_file():
            download_models()
        self.model = Model(wakeword_models=[cfg.model], inference_framework="onnx")
        self._last_fire = 0.0

    async def wait_for_wake(self, mic_queue: asyncio.Queue[np.ndarray]) -> None:
        """Drain mic frames until the wake word fires."""
        while True:
            frame = await mic_queue.get()
            scores = self.model.predict(frame)
            score = max(scores.values()) if scores else 0.0
            now = time.monotonic()
            if score >= self.cfg.threshold and (now - self._last_fire) > self.cfg.cooldown_sec:
                self._last_fire = now
                # Reset internal buffers so we don't immediately re-fire.
                self.model.reset()
                return
