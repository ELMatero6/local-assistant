from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import numpy as np

from .audio import MicStream, play_pcm
from .config import Config, load_config
from .fx import apply_fx, build_hl1_chain
from .llm import OllamaClient
from .stt import STT, Recorder
from .tts import TTS, split_sentences_streaming
from .vision import Webcam
from .wake import WakeDetector

log = logging.getLogger("assistant")


async def speak_streaming(
    token_iter,
    tts: TTS,
    fx_chain,
    cfg: Config,
) -> None:
    """Buffer streamed tokens by sentence; synthesize + play each one as it lands."""
    buffer = ""
    async for token in token_iter:
        buffer += token
        sentences, buffer = split_sentences_streaming(buffer)
        for sentence in sentences:
            await _say(sentence, tts, fx_chain, cfg)
    if buffer.strip():
        await _say(buffer.strip(), tts, fx_chain, cfg)


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

    log.info("Loading models...")
    wake = WakeDetector(cfg.wake)
    recorder = Recorder(cfg.stt, sample_rate=cfg.audio.sample_rate)
    stt = STT(cfg.stt)
    tts = TTS(cfg.tts)
    fx_chain = build_hl1_chain(cfg.fx)
    llm = OllamaClient(cfg.llm)
    webcam = Webcam(cfg.vision)

    mic = MicStream(
        sample_rate=cfg.audio.sample_rate,
        frame_samples=WakeDetector.FRAME_SAMPLES,
        device=cfg.audio.input_device,
        loop=loop,
    )
    mic.start()
    log.info("Listening for wake word '%s'...", cfg.wake.model)

    try:
        while True:
            await wake.wait_for_wake(mic.queue)
            log.info("Wake. Recording utterance...")
            pcm = await recorder.record_utterance(mic.queue)
            log.info("Recorded %.1fs; transcribing...", pcm.size / cfg.audio.sample_rate)

            text = await loop.run_in_executor(None, stt.transcribe, pcm)
            text = text.strip()
            if not text:
                log.info("Empty transcript, back to listening.")
                continue
            log.info("User: %s", text)

            image = None
            if llm.needs_vision(text):
                log.info("Vision keyword detected; grabbing webcam frame.")
                image = await loop.run_in_executor(None, webcam.grab_jpeg)

            try:
                await speak_streaming(llm.stream_reply(text, image), tts, fx_chain, cfg)
            except Exception:
                log.exception("LLM/TTS error")
    finally:
        mic.stop()
        await llm.aclose()


def cli() -> None:
    parser = argparse.ArgumentParser(prog="local-assistant")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    try:
        asyncio.run(run(cfg))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    cli()
