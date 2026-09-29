#!/usr/bin/env bash
# One-command setup inside the AMD hackathon JupyterLab session (notebooks.amd.com/hackathon).
#
#   bash scripts/notebook_setup.sh [train]
#
# pip installs inside a session vanish when the daily quota resets, so dependencies live in a
# virtualenv on persistent storage and are created once.
#
# The venv is created with --system-site-packages on purpose. The session image ships a ROCm build of
# torch; `pip install --target ...` would re-download torch (a CUDA build, several GB) into the target
# directory and put it first on PYTHONPATH, silently shadowing the ROCm one -- the trap the official
# challenge document warns about. In a system-site-packages venv pip sees the ROCm torch as already
# installed, and PIP_CONSTRAINT additionally refuses any attempt to change it.
set -euo pipefail

# Persistent storage is /persistent on "jupyter-hack-*" pods and /workspace on "rgapi-hackathon-*" pods.
if [ -d /persistent ] && [ -w /persistent ]; then PERSIST=/persistent; else PERSIST=/workspace; fi
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PERSIST/venv-gamemaster"
CONSTRAINTS="$PERSIST/torch-constraints.txt"
mkdir -p "$PERSIST/hf-cache"

# Pin the whole torch stack of the *base* interpreter before touching pip.
python3 -m pip freeze 2>/dev/null | grep -iE '^(torch|torchvision|torchaudio|triton|pytorch-triton-rocm)==' > "$CONSTRAINTS" || true
if [ ! -s "$CONSTRAINTS" ]; then
  echo "WARNING: no torch found in the base environment; the ROCm check at the end may fail." >&2
fi
export PIP_CONSTRAINT="$CONSTRAINTS"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv --system-site-packages "$VENV" || {
    echo "python3 -m venv failed; falling back to virtualenv" >&2
    python3 -m pip install --user -q virtualenv
    python3 -m virtualenv -q --system-site-packages "$VENV"
  }
fi

if [ ! -f "$VENV/.installed" ]; then
  "$VENV/bin/python" -m pip install -q -r "$REPO/requirements.txt" pytest pytest-asyncio
  touch "$VENV/.installed"
fi
if [ "${1:-}" = "train" ] && [ ! -f "$VENV/.train-installed" ]; then
  "$VENV/bin/python" -m pip install -q -r "$REPO/requirements-train.txt"
  touch "$VENV/.train-installed"
fi

cat > "$PERSIST/env-gamemaster.sh" <<EOF
export PIP_CONSTRAINT="$CONSTRAINTS"
export HF_HOME="$PERSIST/hf-cache"
export PERSIST="$PERSIST"
export PYTHONPATH="$REPO/academy-core/src:$REPO/src:\${PYTHONPATH:-}"
source "$VENV/bin/activate"
EOF

# Fails the script (set -e) if anything replaced the ROCm torch with a CUDA build.
"$VENV/bin/python" - <<'PY'
import sys

import torch

hip = torch.version.hip
print("torch", torch.__version__, "| hip", hip)
if not hip:
    print("ERROR: torch is no longer a ROCm build (a pip install replaced it). "
          "Delete the persistent venv and re-run this script.", file=sys.stderr)
    sys.exit(1)
ok = torch.cuda.is_available()
print("ROCm GPU available:", ok)
if ok:
    free, total = torch.cuda.mem_get_info(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  VRAM total {total/2**30:.1f} GiB, free {free/2**30:.1f} GiB")
PY
echo "Done. In new terminals run: source $PERSIST/env-gamemaster.sh"
echo "Remember: the 3 h/day quota counts idle time - press 'Turn-off Session' when you stop."
