from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np

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
    pcm = apply_fx(fx_chain, pcm, tts.SAMPLE_RATE)
    await loop.run_in_executor(
        None,
        play_pcm,
        pcm,
        tts.SAMPLE_RATE,
        cfg.audio.output_device,
    )


async def run(cfg: Config) -> None:
    loop = asyncio.get_running_loop()

    log.debug("Loading models...")
    memory = MemoryStore(Path(cfg.memory.dir))
    wake = WakeDetector(cfg.wake)
    recorder = Recorder(cfg.stt, sample_rate=cfg.audio.sample_rate)
    stt = STT(cfg.stt)
    tts = TTS(cfg.tts)
    fx_chain = build_hl1_chain(cfg.fx)
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
        while True:
            await wake.wait_for_wake(mic.queue)
            log.debug("Wake fired; recording utterance.")
            pcm = await recorder.record_utterance(mic.queue)
            log.debug("Recorded %.1fs; transcribing.", pcm.size / cfg.audio.sample_rate)

            text = await loop.run_in_executor(None, stt.transcribe, pcm)
            text = text.strip()
            if not text:
                log.debug("Empty transcript, back to listening.")
                continue
            chat.info("you: %s", text)

            image = None
            if llm.needs_vision(text):
                log.debug("Vision keyword detected; grabbing webcam frame.")
                image = await loop.run_in_executor(None, webcam.grab_jpeg)

            try:
                reply = await speak_streaming(llm.stream_reply(text, image), tts, fx_chain, cfg)
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
