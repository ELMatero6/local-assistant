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
git clone <this repo>
cd local-assistant
git checkout claude/local-ai-assistant-vision-XWqE4
./setup.sh         # apt deps + Ollama + .venv + pip install + cuDNN probe
```

`setup.sh` is idempotent — re-run any time. It will:

1. `apt install` ffmpeg, libsndfile, espeak-ng, portaudio, v4l-utils, build tools.
2. Add you to the `audio` and `video` groups (log out + back in once after first run).
3. Install Ollama and pull the model named in `config.yaml` (`llm.model`).
4. Create `.venv/` and `pip install -e .` into it.
5. Smoke-test faster-whisper + CUDA; if cuDNN is missing it tries
   `nvidia-cudnn-cu12` automatically and falls back to advising
   `int8_float16` if that also fails.

You'll need an NVIDIA driver + CUDA 12.x already installed
(`nvidia-smi` should work). On a fresh box:
`sudo ubuntu-drivers autoinstall && sudo reboot`.

## Run

```bash
./run.sh             # forwards args to local-assistant
./run.sh -v          # verbose logs
```

`run.sh` shells straight into `.venv/bin/local-assistant` — no need to
`source .venv/bin/activate` first.

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
