#!/usr/bin/env bash
# Bootstraps the local-assistant on Ubuntu.
# Idempotent: safe to re-run.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

log() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!! \033[0m %s\n' "$*" >&2; }

# ---- 1. system packages -----------------------------------------------------
log "Installing apt packages (sudo)"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    build-essential \
    ffmpeg libsndfile1 \
    espeak-ng \
    portaudio19-dev \
    v4l-utils \
    git curl ca-certificates

# ---- 2. audio/video group membership ----------------------------------------
if ! id -nG "$USER" | tr ' ' '\n' | grep -qx audio || \
   ! id -nG "$USER" | tr ' ' '\n' | grep -qx video; then
    log "Adding $USER to audio,video groups (log out + back in for this to take effect)"
    sudo usermod -aG audio,video "$USER"
fi

# ---- 3. Ollama --------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
    log "Installing Ollama"
    curl -fsSL https://ollama.com/install.sh | sh
else
    log "Ollama already installed: $(ollama --version 2>/dev/null || echo unknown)"
fi

if command -v systemctl >/dev/null 2>&1; then
    sudo systemctl enable --now ollama || warn "Could not enable ollama service; start it manually."
fi

# Pull the model name out of config.yaml using a quick grep so we don't need
# pyyaml installed yet (and so this works regardless of which python is default).
MODEL="$(awk '/^llm:/{f=1;next} f && /^[^ ]/ {f=0} f && /^[[:space:]]+model:/ {gsub(/^[[:space:]]+model:[[:space:]]*/,""); gsub(/[\"'\'']/,""); print; exit}' config.yaml 2>/dev/null)"
MODEL="${MODEL:-gemma4:latest}"
log "Pulling Ollama model: $MODEL"
if ! ollama pull "$MODEL"; then
    warn "ollama pull '$MODEL' failed. Check the tag with 'ollama list' and update config.yaml."
fi

# ---- 4. Python venv + project install ---------------------------------------
# Pick a compatible interpreter (3.10-3.12). Avoid 3.13+ because Kokoro's
# `misaki` dep doesn't yet ship wheels for them.
PY_BIN=""
for v in 3.12 3.11 3.10; do
    if command -v "python$v" >/dev/null 2>&1; then
        PY_BIN="python$v"
        break
    fi
done
if [ -z "$PY_BIN" ]; then
    warn "No python3.10-3.12 found on PATH. Install one (e.g. apt install python3.12 python3.12-venv) and re-run."
    exit 1
fi
log "Using interpreter: $PY_BIN ($($PY_BIN --version))"

# If an existing .venv was created with a different Python, blow it away.
if [ -d .venv ]; then
    EXISTING="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")"
    EXPECTED="$($PY_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [ "$EXISTING" != "$EXPECTED" ]; then
        warn "Existing .venv uses Python $EXISTING but we want $EXPECTED. Recreating."
        rm -rf .venv
    fi
fi

if [ ! -d .venv ]; then
    log "Creating virtualenv at .venv with $PY_BIN"
    "$PY_BIN" -m venv .venv
else
    log "Reusing existing .venv"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

log "Upgrading pip + installing project (editable)"
pip install --upgrade pip wheel
pip install -e .

# ---- 5. cuDNN sanity --------------------------------------------------------
log "Probing faster-whisper / cuDNN"
if ! python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cuda', compute_type='float16')" 2>/dev/null; then
    warn "faster-whisper float16 on CUDA failed. Trying nvidia-cudnn-cu12..."
    pip install nvidia-cudnn-cu12 || true
    if ! python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', device='cuda', compute_type='float16')" 2>/dev/null; then
        warn "Still failing. Edit config.yaml: set stt.compute_type to int8_float16."
    fi
fi

log "Done. Launch with: ./run.sh"
