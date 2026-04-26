from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np
from openwakeword.model import Model
from openwakeword.utils import download_models

from .config import WakeCfg

log = logging.getLogger("assistant.wake")


class WakeDetector:
    """Wraps openWakeWord to fire whenever the configured wake word is detected.

    Consumes 16k mono int16 frames of length 1280 (80 ms) from a shared queue;
    that is the chunk size openWakeWord expects.
    """

    FRAME_SAMPLES = 1280  # 80 ms at 16 kHz

    BUNDLED = {"alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy", "weasley", "timer"}

    def __init__(self, cfg: WakeCfg):
        self.cfg = cfg
        # cfg.model is either a project-local .onnx path (custom-trained, e.g.
        # wake/hey_dave.onnx) or one of the bundled names. The shared melspec /
        # embedding helpers always need to be downloaded the first time.
        download_models()
        path = Path(cfg.model)
        if path.is_file():
            wake_models = [str(path.resolve())]
        elif cfg.model in self.BUNDLED:
            wake_models = [cfg.model]
        else:
            log.warning(
                "Wake model %r is neither a bundled name nor an existing file; "
                "falling back to 'hey_jarvis'. Train a custom model and drop it "
                "at the configured path to use your own.",
                cfg.model,
            )
            wake_models = ["hey_jarvis"]
        log.info("Wake model: %s", wake_models[0])
        self.model = Model(wakeword_models=wake_models, inference_framework="onnx")
        self._last_fire = 0.0

    async def wait_for_wake(self, mic_queue: asyncio.Queue[np.ndarray]) -> None:
        """Drain mic frames until the wake word fires.

        With debug logging on, prints peak mic RMS + max wake score every
        second so you can see whether the mic is being heard at all.
        """
        debug = log.isEnabledFor(logging.DEBUG)
        last_log = time.monotonic()
        peak_rms = 0.0
        peak_score = 0.0

        while True:
            frame = await mic_queue.get()
            scores = self.model.predict(frame)
            score = max(scores.values()) if scores else 0.0

            if debug:
                rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
                peak_rms = max(peak_rms, rms)
                peak_score = max(peak_score, score)
                now = time.monotonic()
                if now - last_log > 1.0:
                    log.debug(
                        "mic rms_peak=%5d / 32767  wake_score_peak=%.3f  threshold=%.2f",
                        int(peak_rms), peak_score, self.cfg.threshold,
                    )
                    last_log = now
                    peak_rms = 0.0
                    peak_score = 0.0

            now = time.monotonic()
            if score >= self.cfg.threshold and (now - self._last_fire) > self.cfg.cooldown_sec:
                self._last_fire = now
                # Reset internal buffers so we don't immediately re-fire.
                self.model.reset()
                return
