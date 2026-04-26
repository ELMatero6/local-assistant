from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AudioCfg:
    input_device: int | None = None
    output_device: int | None = None
    sample_rate: int = 16000
    playback_rate: int = 24000
    input_gain: float = 1.0           # software gain for quiet mics; 1.0 = none


@dataclass
class WakeCfg:
    model: str = "hey_jarvis"
    threshold: float = 0.5
    cooldown_sec: float = 1.5


@dataclass
class STTCfg:
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    language: str = "en"
    silence_timeout_sec: float = 1.2
    max_utterance_sec: float = 20.0
    vad_threshold: float = 0.3        # Silero VAD speech probability cutoff
    whisper_vad_filter: bool = False  # second VAD pass inside Whisper; off by default
    initial_prompt: str = ""          # vocabulary hint to bias Whisper (names, jargon, etc.)


@dataclass
class LLMCfg:
    host: str = "http://localhost:11434"
    model: str = "gemma4:latest"
    system_prompt: str = ""
    vision_keywords: list[str] = field(default_factory=list)
    context_messages: int = 8


@dataclass
class TTSCfg:
    model: str = "F5TTS_v1_Base"          # F5-TTS checkpoint (F5TTS_v1_Base | F5TTS_Base | E2TTS_Base)
    device: str = "cuda"                  # "cuda" or "cpu"
    ref_audio: str = "dave.mp3"           # file path to the reference voice clip
    ref_text: str = ""                    # transcript of ref_audio (required for cloning)
    prewarm: bool = True                  # synthesize a throwaway phrase at startup to JIT kernels
    nfe_step: int = 32                    # diffusion steps; lower = faster / less detailed (try 16)
    cfg_strength: float = 2.0             # classifier-free guidance; higher = closer to ref voice
    speed: float = 1.0                    # 1.0 = same tempo as ref; <1 slower, >1 faster
    cross_fade_duration: float = 0.15     # crossfade between F5's internal text chunks
    seed: int | None = None               # set int for reproducible voice; null = random


@dataclass
class FXCfg:
    enabled: bool = False                 # default off: clean cloned voice. Flip on for HL1 PA tone.
    highpass_hz: float = 300
    lowpass_hz: float = 3200
    bitcrush_bits: int = 10
    drive_db: float = 8
    reverb_room_size: float = 0.18
    reverb_damping: float = 0.7
    reverb_wet: float = 0.14
    reverb_dry: float = 0.85
    output_gain_db: float = 2


@dataclass
class VisionCfg:
    device_index: int = 0
    width: int = 1280
    height: int = 720
    jpeg_quality: int = 85


@dataclass
class MemoryCfg:
    dir: str = "memory"


@dataclass
class Config:
    audio: AudioCfg = field(default_factory=AudioCfg)
    wake: WakeCfg = field(default_factory=WakeCfg)
    stt: STTCfg = field(default_factory=STTCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)
    tts: TTSCfg = field(default_factory=TTSCfg)
    fx: FXCfg = field(default_factory=FXCfg)
    vision: VisionCfg = field(default_factory=VisionCfg)
    memory: MemoryCfg = field(default_factory=MemoryCfg)


def _build(section: dict[str, Any] | None, cls):
    import dataclasses
    if not section:
        return cls()
    valid = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in section.items() if k in valid})


def load_config(path: str | Path = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        audio=_build(raw.get("audio"), AudioCfg),
        wake=_build(raw.get("wake"), WakeCfg),
        stt=_build(raw.get("stt"), STTCfg),
        llm=_build(raw.get("llm"), LLMCfg),
        tts=_build(raw.get("tts"), TTSCfg),
        fx=_build(raw.get("fx"), FXCfg),
        vision=_build(raw.get("vision"), VisionCfg),
        memory=_build(raw.get("memory"), MemoryCfg),
    )
