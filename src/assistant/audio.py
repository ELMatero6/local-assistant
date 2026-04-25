from __future__ import annotations

import asyncio

import numpy as np
import sounddevice as sd


class MicStream:
    """Continuous 16-bit mono mic capture in fixed-size frames.

    Pushes int16 numpy arrays of length `frame_samples` onto an asyncio.Queue
    so wake-word and recorder coroutines can consume the same stream.
    """

    def __init__(
        self,
        sample_rate: int,
        frame_samples: int,
        device: int | None,
        loop: asyncio.AbstractEventLoop,
    ):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device = device
        self.loop = loop
        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            # Underflows are common on first start; ignore.
            pass
        # indata shape: (frames, 1) int16
        chunk = indata[:, 0].copy()
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, chunk)
        except asyncio.QueueFull:
            pass

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def play_pcm(pcm: np.ndarray, sample_rate: int, device: int | None) -> None:
    """Blocking playback of float32 mono PCM in [-1, 1]."""
    sd.play(pcm, samplerate=sample_rate, device=device, blocking=True)


def to_float32(pcm_int16: np.ndarray) -> np.ndarray:
    return (pcm_int16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
