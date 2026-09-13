#!/bin/sh
# One-shot setup on a fresh Mac (Apple Silicon or Intel):
#   git clone ... && cd remove-bg && ./setup.sh
# Installs uv if missing, builds .venv with Python 3.12, installs the pinned
# packages and downloads the model weights so the tool works offline after.
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv (python package manager) ..."
  if command -v brew >/dev/null 2>&1; then
    brew install uv
  else
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

if [ ! -x .venv/bin/python ]; then
  echo "creating .venv with Python 3.12 ..."
  uv venv --python 3.12 .venv
fi

echo "installing packages (torch is ~200 MB, first time takes a few minutes) ..."
VIRTUAL_ENV=.venv uv pip install -r requirements.lock.txt

MODEL="${1:-hr-matting}"
echo "downloading weights for $MODEL (~900 MB, once) ..."
.venv/bin/python - "$MODEL" <<'PY'
import sys, rmbg
from huggingface_hub import snapshot_download
repo = rmbg.MODELS[sys.argv[1]][0]
snapshot_download(repo)
print("weights cached in ~/.cache/huggingface")
PY

.venv/bin/python - <<'PY'
import torch, rmbg
dev = rmbg.pick_device()
print(f"\nready: device={dev.type}, RAM={rmbg.total_ram_gb():.0f} GB, "
      f"default resolution={rmbg.pick_size('hr-matting')}px")
print("run  ./run-ui.sh  and open http://127.0.0.1:8777")
PY
