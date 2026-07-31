# RUNBOOK — GPU steps

Everything in this file touches the GPU or installs packages behind the proxy. None of
it has been run yet — the no-GPU-required unit test suite is the only thing that's been
executed so far (see the handoff message for its output). There is no CPU training path
in this codebase; every config in the sweep runs on CUDA. Run these steps in order.

All commands assume you're in `benchmark/` inside a JupyterLab terminal on the CML
session with the A100.

---

## 1. Confirm the GPU is visible

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Expect:** `nvidia-smi` shows one A100 80GB with ~0 MB used (unless something else is
running on it). The Python line prints a torch version, `True`, and `NVIDIA A100`.

**If it fails:** `torch.cuda.is_available()` returning `False` usually means the CML
session wasn't given a GPU, or the installed torch is a CPU-only build (see step 2).
Check the session's resource profile before going further — nothing below will work
without a visible GPU.

---

## 2. Install dependencies

Check what's already there first — CML runtimes often ship a working CUDA-enabled
torch, and reinstalling it from behind the proxy is the step most likely to fail or to
silently swap in a CPU-only build:

```bash
python -c "import torch; print(torch.__version__)" 2>&1
```

If that printed a version (ideally with `+cu*` in it, e.g. `2.4.0+cu121`), skip
straight to installing the rest:

```bash
pip install pandas matplotlib
```

If torch is missing or the printed version has no CUDA build, install everything:

```bash
pip install -r ../requirements.txt
```

**Expect:** installs complete without error. Re-run the check from step 1 afterward —
if `torch.cuda.is_available()` is still `False` after installing, the proxy likely
served a CPU-only wheel. Force a CUDA build from the public PyTorch index first:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

`cu121` is broadly compatible with a driver reporting CUDA 12.9 (drivers are backward
compatible with older CUDA runtimes). If that 404s, retry with `cu124`. Re-run the
step-1 check afterward — expect a version string containing `+cu121` (or `+cu124`).

If the proxy blocks `download.pytorch.org` entirely, that means an internal index URL
is required for the CUDA build, which is environment-specific and outside this
runbook's scope — ask your infra team for the internal mirror URL.

**Optional: 8-bit Adam.** This enables the `adam8bit` and `both` configs (3 of the 24
config rows in the default sweep). It's proxy-sensitive, so it's kept out of the base
install:

```bash
pip install -r ../requirements-optional.txt
```

**If it fails:** that's fine — `benchmark.py` detects a missing `bitsandbytes` at
startup, prints `bitsandbytes not installed -- skipping N adam8bit config(s)`, and
runs the rest of the sweep without them. You'll get a smaller CSV (no 8-bit Adam rows)
rather than a crash. Come back to this step later if you get proxy access sorted out,
and just re-run the sweep — nothing else changes.

---

## 3. GPU smoke test

One tiny config (d=256, 2 layers, batch 32) to prove the whole pipeline — model
build, training loop, subprocess spawn, CUDA memory measurement, CSV write — works on
this machine before committing to the full sweep:

```bash
python benchmark.py --smoke-test --output /tmp/smoke_results.csv
```

**Expect:** finishes in well under a minute. Output ends with:
```
Wrote 1 rows to /tmp/smoke_results.csv
=== Summary (grouped by lever) ===
[smoke_test]
  smoke_test: allocated=0.0X GB  reserved=0.0X GB  predicted=0.0X GB  ratio=...
```
A few hundred MB allocated is normal for this tiny config.

**If it fails:** an OOM here would be surprising (this config is a few MB of
weights) — a stack trace instead means something structural is broken (import error,
CUDA driver mismatch). Paste the traceback rather than proceeding to step 4.

---

## 4. Full sweep

```bash
python benchmark.py --output results.csv 2>&1 | tee sweep_log.txt
```

This runs 21 configs by default, or 24 if bitsandbytes is installed (step 2) — the
default `--batch-values`/`--seq-values`/etc. sweeps sequentially: baseline, the
batch/sequence/optimizer/checkpointing sweeps, two checkpointing crosses (checkpointing
=True repeated across every batch and every seq value already in the sweep, not just
baseline), the combined-levers set at the most demanding batch×sequence combination, and
the FP32 reference. Each config runs in its own subprocess, so a crash or OOM in one
config doesn't take down the rest.

Each config now runs one extra, untimed profiling step internally (used to isolate
forward-pass-only peak memory) on top of the usual `--steps`, so expect wall time per
config to run a little longer than earlier versions of this benchmark — a few percent,
not a step change. `results.csv` also carries more columns now: direct measurements
(`measured_param_bytes`, `alloc_after_optimizer_gb`, `peak_forward_gb`, ...) alongside
the predicted line items, and `predicted_allocated_gb`/`predicted_reserved_gb`/
`predicted_oom` replace the old `predicted_total_gb`. See `MANUAL.md`'s column table
for the full list.

**Expected runtime:** roughly 15–25 minutes total (up from the previous 17-config
sweep's 10–20 minutes, since this now runs 21-24 configs). Most of that is CUDA context
init/teardown per subprocess (a few seconds each) plus the actual training steps;
the largest configs (batch 4096, seq 2048) take longer per step and may also fail
fast on OOM rather than running long.

**On the two new checkpointing crosses:** these are genuinely unverified — the formula
predicts `checkpointing_batch` OOMs around batch=4096 and `checkpointing_seq` OOMs around
seq=2048 (both fit at smaller values), but neither has ever run on real hardware. Check
each row's `predicted_oom` column against what actually happened rather than assuming
the formula called it right; that comparison is the whole point of adding these rows.

**Expected OOMs — don't treat these as failures:**
- `seq_len=2048` (in the sequence sweep) is a real candidate for OOM even at the
  baseline batch of 256 — attention memory can scale worse than the article's linear
  floor formula predicts, and this is exactly the boundary this benchmark exists to
  find.
- In the combined-levers set (batch 4096, seq 2048), the `baseline` (no levers) and
  `adam8bit`-only variants are the most likely to OOM, since neither reduces the
  activation floor. `checkpointing` and `both` are the ones designed to fit.

Each OOM prints `[child] OOM at <lever>=<value>: ...` to stderr and appears in
`results.csv` as `oom=True` with `max_allocated_gb`/`max_reserved_gb` set to `OOM` —
the sweep keeps going.

**Good final output** ends with:
```
Wrote N rows to results.csv

=== Summary (grouped by lever) ===
[baseline]
  baseline: allocated=... GB  reserved=... GB  predicted=... GB  ratio=...
...
[activation multiplier vs. sequence length]  (measured activations / predicted floor)
  seq_len=  100: measured~=... GB  predicted_floor=... GB  multiplier=...x
  ...
```

**If a config hangs:** each subprocess has a 900s timeout (`--timeout` to change it);
a hang past that raises `subprocess.TimeoutExpired` and stops the whole sweep. Kill it
(Ctrl-C), check `nvidia-smi` for a stuck process, and consider lowering
`--batch-values`/`--seq-values` for a first pass.

**If a config dies with `exit -9` and no JSON result (SIGKILL, not a caught OOM):** this
is a host-side kill, not a GPU/CUDA OOM (those raise a catchable `RuntimeError` and show
up as `oom=True` instead). Check `dmesg -T | egrep -i 'kill|oom'` — if you see
`Memory cgroup out of memory: Killed process ... (python)`, it's the CML session's pod
memory limit (RAM, separate from its GPU allocation), not host RAM (`free -h` can show
plenty of host memory free while the pod's own memcg limit is far smaller). This bites
specifically on the ~1B-param baseline: the model is built on CPU then `.to(device)`'d
(`train.py`), so there's a brief CPU-side copy needing ~4GB+ just for fp32 weights before
anything moves to the GPU. If `--smoke-test` (tiny model) passes but the full sweep dies
immediately on `baseline`, this is almost certainly it — bump the session's memory
profile (16GB+ is a safe target) in the CML UI, restart the session, and rerun.

**If you want a smaller/custom sweep:**
```bash
python benchmark.py --batch-values 256,1024 --seq-values 100,512,1024 --output results_small.csv
```
`--d` and `--layers` override the baseline model size if you want to test a
differently-sized model instead of the 1B-parameter baseline.

---

## 5. Generate the real charts

```bash
python plot_results.py --input results.csv --outdir charts/
```

**Expect:** finishes in a few seconds (no GPU involved — this is the same code path
already exercised against `fixtures/results_sample.csv` in the test suite). Prints
`Charts written to <absolute path to charts/>`.

---

## 6. Validate the predictions

```bash
python validate_results.py --input results.csv
```

**Expect:** finishes instantly (no GPU involved — reads the CSV's own
`predicted_allocated_gb`/`predicted_oom` columns and compares them to
`max_allocated_gb`/`oom`). Prints a per-row table (predicted, measured, absolute error,
percent error, whether the OOM call was right) plus a pass/fail summary against the
acceptance bar: ≤10% error on `max_allocated_gb` for every non-OOM row, and a correct
OOM call on every row.

**If a row fails:** that's real signal, not a script bug — it means the formula in
`predictions.py` is off for that config. As of the 2026-07-31 H100 sweep every row
validates (worst non-OOM error 2.2%, every OOM call correct), including the checkpointed
rows the earlier formula missed by 10–14%. If a checkpointed row starts drifting again,
suspect the fragmentation constant first (its measured reserved-minus-allocated gap
spans 3.58–6.12 GB) — but don't retune it to force a pass; it and the CUDA-context term
are deliberately kept as named, documented empirical terms (see `predictions.py`) rather
than fit to make numbers match.

**Where output lands:**
- `benchmark/results.csv` — the full sweep data
- `benchmark/sweep_log.txt` — the console log from step 4 (if you used `tee`)
- `benchmark/charts/memory_vs_batch.{png,svg}`
- `benchmark/charts/memory_vs_seqlen.{png,svg}`
- `benchmark/charts/lever_impact.{png,svg}`
- `benchmark/charts/allocated_vs_reserved.{png,svg}`

Pull `results.csv` and the `charts/` directory back for the article; everything else
in `benchmark/` is source.
