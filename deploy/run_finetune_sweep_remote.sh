#!/usr/bin/env bash
# Run the full LoRA fine-tune sweep on a fresh remote GPU box and pull results back.
#
# FALLBACK PATH ONLY. The primary Article 2 runner is a Cloudera AI (CAI) session --
# see benchmark/RUNBOOK.md § "Fine-tuning sweep" (F1-F9). Use this script only when CAI
# isn't available.
#
# Usage:
#   ./run_finetune_sweep_remote.sh <user@host> [run-label]
#
# Example:
#   ./run_finetune_sweep_remote.sh ozarate@89.169.96.15
#
# Assumes: Ubuntu-ish box with an NVIDIA GPU + driver (nvidia-smi works),
# python3 + venv available, and your SSH key already authorized on the box.
# Results land locally in runs/<run-label>-finetune/ so repeated runs don't overwrite
# each other, and Article 1's sweep results stay untouched.

set -euo pipefail

HOST="${1:?usage: ./run_finetune_sweep_remote.sh <user@host> [run-label]}"
LABEL="${2:-$(date +%Y%m%d-%H%M)}-finetune"
REPO_URL="https://github.com/odog96/gpu-sizing-playbook.git"

echo "=== [1/4] Bootstrap + run fine-tune sweep on ${HOST} (~25-40 min on A100 80GB) ==="
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
pip install -q -r requirements-finetune.txt
# bitsandbytes enables adam8bit + INT8/INT4 base_precision rows; the benchmark skips
# them gracefully if this install fails, so warn-and-continue rather than abort.
pip install -q -r requirements-optional.txt \
  || echo "WARN: bitsandbytes install failed -- adam8bit and INT8/INT4 base rows will be skipped"

# --- preflight: CUDA torch actually sees the GPU, HF stack is importable, base model
#     pullable ---
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False -- see RUNBOOK.md F1"
print("GPU OK:", torch.__version__, torch.cuda.get_device_name(0))
import transformers, peft
print("transformers OK:", transformers.__version__)
print("peft OK:", peft.__version__)
from transformers import AutoConfig
c = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
print("Base model reachable:", c.num_hidden_layers, "layers,", c.hidden_size, "hidden,",
      c.num_key_value_heads, "kv_heads,", c.intermediate_size, "intermediate")
try:
    import bitsandbytes
    print("bitsandbytes OK:", bitsandbytes.__version__)
except Exception as e:
    print("bitsandbytes NOT usable -> adam8bit/INT8/INT4 rows will be skipped:", e)
PY

# --- smoke test, then the real fine-tune sweep ---
cd benchmark
python benchmark_finetune.py --smoke-test --output /tmp/smoke_finetune.csv
python benchmark_finetune.py --output results_finetune.csv 2>&1 | tee sweep_finetune_log.txt

# --- charts + validation ---
python plot_finetune_results.py --input results_finetune.csv --outdir charts_finetune/
python validate_finetune_results.py --input results_finetune.csv || true

# --- optional: allocator-history diagnostic (probes 1-3 in debug_finetune.py) ---
python debug_finetune.py > debug_finetune_out.json || true
REMOTE

echo "=== [2/4] Pulling results back to runs/${LABEL}/ ==="
mkdir -p "runs/${LABEL}"
scp "${HOST}:gpu-sizing-playbook/benchmark/results_finetune.csv"       "runs/${LABEL}/"
scp "${HOST}:gpu-sizing-playbook/benchmark/sweep_finetune_log.txt"     "runs/${LABEL}/"
scp -r "${HOST}:gpu-sizing-playbook/benchmark/charts_finetune"          "runs/${LABEL}/"
scp "${HOST}:gpu-sizing-playbook/benchmark/debug_finetune_out.json"    "runs/${LABEL}/" || true

echo "=== [3/4] Local copy of sweep tail ==="
tail -n 50 "runs/${LABEL}/sweep_finetune_log.txt" || true

echo "=== [4/4] Done. Results in runs/${LABEL}/ ==="
echo "REMINDER: delete the VM in your cloud console -- billing stops only on delete."
