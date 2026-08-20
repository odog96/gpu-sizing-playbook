#!/usr/bin/env bash
# Run the full GPU-memory sweep on a fresh remote GPU box and pull results back.
#
# Usage:
#   ./run_sweep_remote.sh <user@host> [run-label]
#
# Example:
#   ./run_sweep_remote.sh ozarate@89.169.96.15
#
# Assumes: Ubuntu-ish box with an NVIDIA GPU + driver (nvidia-smi works),
# python3 + venv available, and your SSH key already authorized on the box.
# Results land locally in runs/<run-label>/ (default label = current date-time),
# so repeated runs never overwrite each other or benchmark/results.csv.

set -euo pipefail

HOST="${1:?usage: ./run_sweep_remote.sh <user@host> [run-label]}"
LABEL="${2:-$(date +%Y%m%d-%H%M)}"
REPO_URL="https://github.com/odog96/gpu-sizing-playbook.git"

echo "=== [1/4] Bootstrap + run sweep on ${HOST} (this is the long step, ~20-25 min) ==="
ssh "${HOST}" 'bash -s' <<'REMOTE'
set -euo pipefail

# --- clone or update the repo ---
if [ ! -d gpu-sizing-playbook ]; then
  git clone https://github.com/odog96/gpu-sizing-playbook.git
fi
cd gpu-sizing-playbook
git pull --ff-only

# --- venv + dependencies ---
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
# bitsandbytes enables the adam8bit configs; benchmark.py skips them gracefully
# if this install fails, so warn-and-continue rather than abort.
pip install -q -r requirements-optional.txt \
  || echo "WARN: bitsandbytes install failed -- adam8bit rows will be skipped"

# --- preflight: CUDA torch actually sees the GPU ---
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False -- CPU-only torch or no GPU; see RUNBOOK.md step 2"
print("GPU OK:", torch.__version__, torch.cuda.get_device_name(0))
try:
    import bitsandbytes
    print("bitsandbytes OK:", bitsandbytes.__version__)
except Exception as e:
    print("bitsandbytes NOT usable -> adam8bit rows will be skipped:", e)
PY

# --- smoke test, then the real sweep ---
cd benchmark
python benchmark.py --smoke-test --output /tmp/smoke_results.csv
python benchmark.py --output results.csv 2>&1 | tee sweep_log.txt

# --- charts + validation (validate exits nonzero on acceptance-bar misses;
#     that's expected signal, not a script failure -- don't abort on it) ---
python plot_results.py --input results.csv --outdir charts/
python validate_results.py --input results.csv || true
REMOTE

echo "=== [2/4] Pulling results back to runs/${LABEL}/ ==="
mkdir -p "runs/${LABEL}"
scp "${HOST}:gpu-sizing-playbook/benchmark/results.csv"   "runs/${LABEL}/"
scp "${HOST}:gpu-sizing-playbook/benchmark/sweep_log.txt" "runs/${LABEL}/"
scp -r "${HOST}:gpu-sizing-playbook/benchmark/charts"     "runs/${LABEL}/"

echo "=== [3/4] Local copy of validation summary ==="
tail -n 40 "runs/${LABEL}/sweep_log.txt" || true

echo "=== [4/4] Done. Results in runs/${LABEL}/ ==="
echo "REMINDER: delete the VM in the Nebius console -- billing stops only on delete."
