#!/usr/bin/env bash
# Runs local-assistant from the project venv. Forwards all args.
# No need to `source .venv/bin/activate` first.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$REPO_DIR/.venv/bin/local-assistant"

if [ ! -x "$BIN" ]; then
    echo "venv missing or project not installed. Run ./setup.sh first." >&2
    exit 1
fi

# Make CUDA libs from pip wheels (nvidia-cublas-cu12, nvidia-cudnn-cu12, etc.)
# discoverable to CTranslate2 / faster-whisper without needing a system CUDA toolkit.
EXTRA_LIBS=""
while IFS= read -r d; do
    EXTRA_LIBS="${EXTRA_LIBS:+$EXTRA_LIBS:}$d"
done < <(find "$REPO_DIR/.venv" -type d -path '*/nvidia/*/lib' 2>/dev/null)
if [ -n "$EXTRA_LIBS" ]; then
    export LD_LIBRARY_PATH="${EXTRA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec "$BIN" "$@"
