# local-assistant

Local voice assistant that pipes:

```
mic -> openWakeWord -> faster-whisper STT -> Ollama (gemma4) -> Kokoro TTS -> HL1 tram FX -> speaker
```

The model has persistent memory across sessions (`identity.md` + `notes.md` +
`memory.md`) and can edit/search those files via tool calls. It can also grab
a webcam frame on demand when you say things like "look at this", "describe
what you see", etc. Everything runs locally.

## Ubuntu setup

Tested on Ubuntu 22.04 / 24.04 with a single RTX 3090.

```bash
# system deps
sudo apt update
sudo apt install -y \
    python3-venv python3-dev \
    ffmpeg libsndfile1 \
    espeak-ng \
    portaudio19-dev \
    v4l-utils

# (one-time) install the NVIDIA driver + CUDA 12.x if you don't already have it,
# then verify:
nvidia-smi

# (one-time) Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull gemma4:latest
```

Then the project:

```bash
git clone <this repo>
cd local-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If `faster-whisper` errors with a `libcudnn_ops_infer.so` not-found, either
`pip install nvidia-cudnn-cu12` or drop `stt.compute_type` to `int8_float16`
in `config.yaml`.

Make sure your user is in the `audio` and `video` groups so PortAudio and
V4L2 can see the webcam mic + camera:

```bash
sudo usermod -aG audio,video $USER
# log out + back in
```

## Run

```bash
ollama serve &           # skip if it's already running as a systemd unit
local-assistant -v
```

Say "hey jarvis", pause briefly, then ask your question.

## Picking audio + video devices

Mic / speaker:

```bash
python -m sounddevice
```

Set the index numbers in `config.yaml` under `audio.input_device` and
`audio.output_device`.

Webcam:

```bash
v4l2-ctl --list-devices
```

Set `vision.device_index` (typically `0` for `/dev/video0`).

## Memory layer

The model has three persistent markdown files in `memory/`:

| File          | Purpose                                                 | Always in prompt? |
|---------------|---------------------------------------------------------|-------------------|
| `identity.md` | The model's self-description. It can rewrite this.      | Yes (full)        |
| `notes.md`    | Short scratchpad for in-flight context.                 | Yes (full)        |
| `memory.md`   | Long-term store. Can grow large; only top hits injected.| Top-K BM25 hits   |

Tools exposed to the model (handled in `src/assistant/memory.py`):

- `identity_read`, `identity_write`
- `notes_append`, `notes_overwrite`
- `memory_search(query, k)` — BM25 over `## `-delimited chunks
- `memory_append(text)` — start chunks with a `## Heading` for best recall
- `memory_edit(old, new)` — exact-string patch; `old` must be unique

Every user turn, the orchestrator pre-searches `memory.md` for the user's
transcript and injects the top 4 chunks alongside the full identity + notes.
The model can then call `memory_search` itself if it wants more.

### Why BM25?

For a personal note store of <10 MB, BM25 hits the sweet spot:

- **Zero VRAM** — leaves the 3090 for Gemma + Whisper + Kokoro.
- **<10 ms latency** at our sizes; index is rebuilt on file change (mtime cache).
- **Keyword recall is a feature** for memory ("when did I last say *X*?").
- The library is `rank_bm25`, ~200 lines of pure Python, no native deps.

If you ever want semantic / paraphrase recall, the upgrade is straightforward:

1. Add `sentence-transformers` (e.g. `all-MiniLM-L6-v2`, ~80 MB on CPU).
2. Persist embeddings in `sqlite-vec` keyed by chunk hash.
3. Combine BM25 + cosine via Reciprocal Rank Fusion in `MemoryStore.search_memory`.

The interface (`search_memory(query, k) -> list[(chunk, score)]`) doesn't
change, so nothing else needs to know.

## Custom wake word

`hey_jarvis` ships with openWakeWord. For a real HL1 "hey gordon", follow the
openWakeWord training notebook
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
| BM25 memory index                 | CPU     |

Plenty of headroom for KV cache and the webcam frame on the model.

## Layout

```
src/assistant/
  audio.py    mic capture + playback helpers
  wake.py     openWakeWord listener
  stt.py      Silero-VAD-gated recorder + faster-whisper transcriber
  llm.py      Ollama streaming client + tool-call loop (text + vision)
  memory.py   identity/notes/memory store, BM25 search, tool dispatch
  vision.py   on-demand webcam JPEG grab
  tts.py      Kokoro pipeline + sentence splitter for streaming
  fx.py       Pedalboard HL1 tram-PA chain
  main.py     async orchestrator + CLI entrypoint
memory/
  identity.md notes.md memory.md
```

## What's deferred

- Web search backend (next; will be exposed as another tool)
- Barge-in (interrupting TTS when you start talking)
- A custom "hey gordon" wake-word model
- Semantic search upgrade for `memory.md` (see above)
