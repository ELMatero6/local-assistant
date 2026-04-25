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

exec "$BIN" "$@"
