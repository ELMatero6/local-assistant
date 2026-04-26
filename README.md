# local-assistant

Local voice assistant that pipes:

```
mic -> openWakeWord -> faster-whisper STT -> Ollama (gemma4) -> F5-TTS (voice clone) -> speaker
```

The TTS clones any reference voice you point it at (default: `dave.mp3` —
Dave Mustaine). The model has persistent memory across sessions
(`identity.md` + `notes.md` + `memory.md`) and can edit/search those files
via tool calls. It can also grab a webcam frame on demand when you say
things like "look at this", "describe what you see", etc. Everything runs
locally.

The HL1 tram-PA effects chain is still in the codebase but disabled by
default — flip `fx.enabled: true` in `config.yaml` to layer it on top of
the cloned voice.

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

## Mic gain (webcam mics are quiet)

If `./run.sh mic-test` shows a peak under ~5000 when you talk normally, the
mic is too quiet for reliable wake-word + VAD. Fix in this order:

1. **Hardware gain (best — no noise penalty).**

   ```bash
   alsamixer            # F4 to switch to capture, F6 to pick the mic, arrow up
   # or with PulseAudio:
   pactl list sources short
   pactl set-source-volume <name> 150%
   ```

2. **Software gain (fallback).** In `config.yaml`:

   ```yaml
   audio:
     input_gain: 6.0    # 4.0-8.0 typical for webcam mics
   ```

   Re-run `./run.sh mic-test` and aim for peak > 5000 when speaking.

3. **Loosen the VAD** if speech is detected but trimmed:

   ```yaml
   stt:
     vad_threshold: 0.2
   ```

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

The default config expects `wake/hey_dave.onnx` and falls back to bundled
`hey_jarvis` (with a warning) until you train your own. See
[`wake/README.md`](wake/README.md) for the openWakeWord training pipeline
— ~30 min on a free Colab T4, fully local once trained.

## TTS speed

F5-TTS uses 32 diffusion steps per call (`nfe_step`). On a 3090 expect
roughly 1.5-3× realtime per sentence with the default settings. The
fastest dial is dropping `nfe_step`:

```yaml
tts:
  nfe_step: 16          # ~2x faster, still solid quality
  # nfe_step: 8          # ~4x faster, audibly rougher
```

Quality / speed tradeoff is purely runtime — no reload needed.

Other levers already on by default:
- `tts.prewarm: true` — synthesize a throwaway phrase at startup so the
  first real reply isn't slow due to kernel JIT + cuDNN autotune.
- `torch.set_float32_matmul_precision("high")` — Ampere TF32 matmuls.
- `cudnn.benchmark = True` — picks the fastest conv algorithm per input.

To see actual per-synth times run with `-v`:

```
synth 0.42s -> 1.85s audio (4.40x realtime) for 38 chars
```

## Barge-in (interruption)

Start talking while the assistant is speaking; it stops mid-word, drops
the in-flight LLM stream, and waits for "hey dave" again to start a new
turn. Tunable in `config.yaml`:

```yaml
wake:
  interrupt_vad_threshold: 0.7    # stricter than recorder's so we don't trip on our own voice
  interrupt_min_frames: 4         # ~320 ms of consistent speech needed
```

If the model interrupts itself when you have speakers (echo back into
the webcam mic), raise `interrupt_vad_threshold` toward 0.85, lower mic
gain, or use headphones.

## Voice cloning reference audio

`tts.ref_audio` (default: `dave.mp3`) is the file the cloned voice is
copied from. `tts.ref_text` is the **exact transcript** of that file —
the model uses it to align the speaker embedding. If you replace the
voice, also replace the transcript.

3-30 s of clean speech is the sweet spot. Mp3, wav, or any libsndfile
format works.

## VRAM footprint

Rough split on a single 3090 (24 GB):

| Component                         | VRAM    |
|-----------------------------------|---------|
| Ollama: gemma4:latest             | ~10 GB  |
| faster-whisper large-v3 (fp16)    | ~3 GB   |
| F5-TTS_v1_Base + Vocos vocoder    | ~2-3 GB |
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
  tts.py      F5-TTS voice-clone pipeline + sentence splitter for streaming
  fx.py       Pedalboard HL1 tram-PA chain (disabled by default)
  main.py     async orchestrator + CLI entrypoint
memory/
  identity.md notes.md memory.md
```

## What's deferred

- Web search backend (next; will be exposed as another tool)
- Barge-in (interrupting TTS when you start talking)
- A custom "hey gordon" wake-word model
- Semantic search upgrade for `memory.md` (see above)
