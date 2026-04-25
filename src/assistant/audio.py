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
        gain: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device = device
        self.loop = loop
        self.gain = float(gain)
        self.queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)
        self._stream: sd.InputStream | None = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            # Underflows are common on first start; ignore.
            pass
        # indata shape: (frames, 1) int16
        chunk = indata[:, 0].copy()
        if self.gain != 1.0:
            scaled = chunk.astype(np.int32) * self.gain
            chunk = np.clip(scaled, -32768, 32767).astype(np.int16)
        # The actual put runs on the loop thread, so swallow QueueFull there
        # rather than in this callback (where it would never fire).
        self.loop.call_soon_threadsafe(self._enqueue, chunk)

    def _enqueue(self, chunk: np.ndarray) -> None:
        try:
            self.queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass

    def drain(self) -> None:
        """Drop any audio frames currently buffered. Call after TTS playback so
        the model doesn't hear its own voice (or echo) on the next turn."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

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
