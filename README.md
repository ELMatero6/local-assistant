# local-assistant

Local voice assistant that pipes:

```
mic -> openWakeWord -> faster-whisper STT -> Ollama (gemma4) -> Kokoro TTS -> HL1 tram FX -> speaker
```

Optionally grabs a webcam frame and passes it to the model when you say things
like "look at this", "describe what you see", etc. Everything runs locally.

## Requirements

- Linux + an NVIDIA GPU (3090 tested). CUDA 12.x with the matching cuDNN runtime
  (faster-whisper needs `libcudnn_ops_infer` on PATH).
- An Ollama server running locally with `gemma4:latest` pulled.
- A working ALSA/PulseAudio mic (the webcam mic is fine) and speaker.
- ffmpeg + libsndfile (`apt install ffmpeg libsndfile1`).
- For Kokoro: espeak-ng (`apt install espeak-ng`).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If `faster-whisper` complains about cuDNN, install `nvidia-cudnn-cu12` matching
your CUDA version, or fall back to `compute_type: int8_float16` in `config.yaml`.

## Run

```bash
ollama serve &           # if not already running
ollama pull gemma4:latest
local-assistant -v
```

Say "hey jarvis", wait for the chime-free silence, then ask your question.

## Picking audio devices

```bash
python -m sounddevice
```

Set the `input_device` and `output_device` indices in `config.yaml`.

## Custom wake word

`hey_jarvis` ships with openWakeWord. To get a true HL1 "hey gordon",
follow the openWakeWord training notebook
(<https://github.com/dscripka/openWakeWord#training-new-models>), drop the
resulting `.onnx` into the project, and point `wake.model` at its path.

## VRAM footprint

Rough split on a single 3090 (24 GB):

| Component                         | VRAM    |
|-----------------------------------|---------|
| Ollama: gemma4:latest             | ~10 GB  |
| faster-whisper large-v3 (fp16)    | ~3 GB   |
| Kokoro                            | <1 GB   |
| openWakeWord                      | CPU     |

Plenty of headroom for KV cache and the webcam frame on the model.

## Layout

```
src/assistant/
  audio.py    mic capture + playback helpers
  wake.py     openWakeWord listener
  stt.py      Silero-VAD-gated recorder + faster-whisper transcriber
  llm.py      Ollama streaming client (text + vision)
  vision.py   on-demand webcam JPEG grab
  tts.py      Kokoro pipeline + sentence splitter for streaming
  fx.py       Pedalboard HL1 tram-PA chain
  main.py     async orchestrator + CLI entrypoint
```

## What's deferred

- Tool calling / function calling on the model
- Web search backend
- Barge-in (interrupting TTS when you start talking)
- A custom "hey gordon" wake-word model
