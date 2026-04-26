from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np

import sounddevice as sd

from .audio import MicStream, play_pcm
from .config import Config, load_config
from .fx import apply_fx, build_hl1_chain
from .llm import OllamaClient
from .memory import MemoryStore
from .stt import STT, Recorder
from .tts import TTS, split_sentences_streaming
from .vision import Webcam
from .wake import WakeDetector

log = logging.getLogger("assistant")
chat = logging.getLogger("assistant.chat")


def _silence_startup_noise() -> None:
    """Hide third-party warnings that aren't actionable for our users."""
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="kokoro")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    sys.stderr = _StderrLineFilter(sys.stderr, _NOISY_LINES)


_NOISY_LINES = (
    # Qwen3-TTS / CosyVoice2 init chatter
    "code_predictor_config is None",
    "Initializing code_predictor model",
    "talker_config is None",
    "Initializing talker model",
    "speaker_encoder_config is None",
    "encoder_config is None",
    "decoder_config is None",
    "Setting `pad_token_id`",
    "Setting pad_token_id",
    # F5-TTS optional-dependency warnings
    "Warning: flash-attn is not installed",
    "SoX could not be found",
    "sox: not found",
    "If you do not have SoX",
    "http://sox.sourceforge.net",
    "If you do (or think",
    "path variables",
)


class _StderrLineFilter:
    """Stream wrapper that drops lines matching any of the configured substrings.

    Per-line: any unmatched content (real errors, tracebacks, etc.) still passes
    through. Used only in default mode; -v keeps the raw stream.
    """

    def __init__(self, base, patterns: tuple[str, ...]):
        self._base = base
        self._patterns = patterns
        self._buf = ""

    def write(self, data: str) -> int:
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not any(p in line for p in self._patterns):
                self._base.write(line + "\n")
        return len(data)

    def flush(self) -> None:
        if self._buf:
            self._base.write(self._buf)
            self._buf = ""
        self._base.flush()

    def __getattr__(self, name: str):
        return getattr(self._base, name)


async def speak_streaming(
    token_iter,
    tts: TTS,
    fx_chain,
    cfg: Config,
) -> str:
    """Buffer streamed tokens by sentence; synthesize + play each one as it lands.

    Returns the full accumulated reply text.
    """
    buffer = ""
    full = ""
    async for token in token_iter:
        buffer += token
        full += token
        sentences, buffer = split_sentences_streaming(buffer)
        for sentence in sentences:
            await _say(sentence, tts, fx_chain, cfg)
    if buffer.strip():
        await _say(buffer.strip(), tts, fx_chain, cfg)
    return full.strip()


async def _say(text: str, tts: TTS, fx_chain, cfg: Config) -> None:
    loop = asyncio.get_running_loop()
    pcm = await loop.run_in_executor(None, tts.synth, text)
    if pcm.size == 0:
        return
    if fx_chain is not None:
        pcm = apply_fx(fx_chain, pcm, tts.SAMPLE_RATE)
    try:
        await loop.run_in_executor(
            None,
            play_pcm,
            pcm,
            tts.SAMPLE_RATE,
            cfg.audio.output_device,
        )
    except asyncio.CancelledError:
        # Interrupted by user; cut playback immediately.
        sd.stop()
        raise


async def run(cfg: Config) -> None:
    loop = asyncio.get_running_loop()

    log.debug("Loading models...")
    memory = MemoryStore(Path(cfg.memory.dir))
    wake = WakeDetector(cfg.wake)
    recorder = Recorder(cfg.stt, sample_rate=cfg.audio.sample_rate)
    stt = STT(cfg.stt)
    tts = TTS(cfg.tts)
    fx_chain = build_hl1_chain(cfg.fx) if cfg.fx.enabled else None
    llm = OllamaClient(cfg.llm, memory)
    webcam = Webcam(cfg.vision)

    mic = MicStream(
        sample_rate=cfg.audio.sample_rate,
        frame_samples=WakeDetector.FRAME_SAMPLES,
        device=cfg.audio.input_device,
        loop=loop,
        gain=cfg.audio.input_gain,
    )
    mic.start()
    chat.info("listening for '%s'...", cfg.wake.model)

    try:
        skip_wake = False
        while True:
            if not skip_wake:
                await wake.wait_for_wake(mic.queue)
            skip_wake = False

            chat.info("(listening...)")
            pcm = await recorder.record_utterance(mic.queue)
            log.debug("Recorded %.1fs; transcribing.", pcm.size / cfg.audio.sample_rate)

            text = await loop.run_in_executor(None, stt.transcribe, pcm)
            text = text.strip()
            if not text:
                chat.info("(didn't catch that)")
                continue
            chat.info("you: %s", text)

            image = None
            if llm.needs_vision(text):
                log.debug("Vision keyword detected; grabbing webcam frame.")
                image = await loop.run_in_executor(None, webcam.grab_jpeg)

            try:
                token_iter = llm.stream_reply(text, image)
                speak_task = asyncio.create_task(
                    speak_streaming(token_iter, tts, fx_chain, cfg)
                )
                wake_task = asyncio.create_task(wake.wait_for_wake(mic.queue))
                done, _ = await asyncio.wait(
                    {speak_task, wake_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if wake_task in done and not speak_task.done():
                    sd.stop()
                    speak_task.cancel()
                    try:
                        await speak_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    chat.info("(interrupted)")
                    skip_wake = True
                else:
                    wake_task.cancel()
                    try:
                        await wake_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    reply = speak_task.result() if speak_task.done() and not speak_task.cancelled() else ""
                    if reply:
                        chat.info("assistant: %s", reply)
            except Exception:
                log.exception("LLM/TTS error")
            finally:
                # Drop frames captured while we were speaking so the next wake
                # cycle doesn't fire on our own voice or stale echo.
                mic.drain()
    finally:
        mic.stop()
        await llm.aclose()


def mic_test(cfg: Config, seconds: float) -> None:
    """List devices, then capture from the configured input and print levels.

    Use this to confirm the mic is actually being heard. Fix `audio.input_device`
    in config.yaml until you see the rms_peak rise when you talk.
    """
    import sounddevice as sd

    print(sd.query_devices())
    print(f"\nUsing audio.input_device = {cfg.audio.input_device} (None = system default)")
    print(f"Recording {seconds:.0f}s at {cfg.audio.sample_rate} Hz... talk now.")

    n = int(seconds * cfg.audio.sample_rate)
    rec = sd.rec(n, samplerate=cfg.audio.sample_rate, channels=1, dtype="int16",
                 device=cfg.audio.input_device)
    sd.wait()
    rec = rec[:, 0]

    raw_peak = int(np.abs(rec).max())
    if cfg.audio.input_gain != 1.0:
        scaled = rec.astype(np.int32) * cfg.audio.input_gain
        rec = np.clip(scaled, -32768, 32767).astype(np.int16)
    peak = int(np.abs(rec).max())
    rms = float(np.sqrt(np.mean(rec.astype(np.float32) ** 2)))
    print(f"raw peak (pre-gain):  {raw_peak} / 32767")
    print(f"peak (after gain x{cfg.audio.input_gain}): {peak} / 32767  (>5000 = healthy speech)")
    print(f"RMS (after gain):     {rms:.0f}            (>500 when talking)")
    if raw_peak < 200:
        print("  -> mic is silent. Wrong device or muted.")
    elif peak < 2000:
        print("  -> still quiet after gain. Raise audio.input_gain in config, or boost ALSA gain.")
    else:
        print("  -> mic looks fine.")


def cli() -> None:
    parser = argparse.ArgumentParser(prog="local-assistant")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd")
    mt = sub.add_parser("mic-test", help="Capture from the configured mic and print levels.")
    mt.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        _silence_startup_noise()
        # Conversation-only output: terse, no timestamps, no third-party noise.
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        for noisy in (
            "httpx", "httpcore", "urllib3",
            "faster_whisper", "openwakeword", "silero_vad",
            "kokoro", "phonemizer", "asyncio",
        ):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        # Suppress our own scaffolding logs; only the chat sub-logger speaks.
        logging.getLogger("assistant").setLevel(logging.WARNING)
        logging.getLogger("assistant.chat").setLevel(logging.INFO)

    cfg = load_config(args.config)
    try:
        if args.cmd == "mic-test":
            mic_test(cfg, args.seconds)
        else:
            asyncio.run(run(cfg))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    cli()
