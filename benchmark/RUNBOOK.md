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

---

# Fine-tuning sweep

Everything above sizes **training** (Article 1). This section sizes **LoRA fine-tuning**
on a frozen bf16 base (Article 2). Same benchmark discipline — subprocess per config,
direct tensor measurement, sweeps one lever at a time from a fixed baseline — but a
different orchestrator (`benchmark_finetune.py`), different sweep (`sweep_config_finetune.py`),
different formulas (`predictions_finetune.py`), and a different CSV column set.
Target hardware for the first-pass sweep is **A100 80 GB**; H100 80 GB is a possible
second run.

The primary runner is a Cloudera AI (CAI) session. The SSH-based
`run_finetune_sweep_remote.sh` is retained as an appendix fallback but is not the
recommended path.

Run these steps in order, in a JupyterLab terminal on the CAI session with the A100 80GB.

## F1. Confirm the GPU is visible

Same check as step 1 above:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Expect:** one A100 80GB with ~0 MB used. If `torch.cuda.is_available()` is `False`,
resolve that before continuing — no step below will work.

## F2. Clone (or `git pull`) and install fine-tune dependencies

```bash
# From a fresh terminal, outside benchmark/:
cd ~
git clone https://github.com/odog96/gpu-sizing-playbook.git
# Or: cd ~/gpu-sizing-playbook && git pull --ff-only

cd ~/gpu-sizing-playbook
pip install -r requirements.txt
pip install -r requirements-finetune.txt
```

**Expect:** installs complete without error. `requirements-finetune.txt` adds
`transformers`, `peft`, `huggingface_hub`, `accelerate`, and `sentencepiece` — none
proxy-sensitive.

**If you also want `bitsandbytes`** (enables `adam8bit` optimizer rows and INT8/INT4
`base_precision` rows — proxy-sensitive, kept out of the base install):

```bash
pip install -r requirements-optional.txt
```

`benchmark_finetune.py` detects a missing `bitsandbytes` at startup and skips those
rows automatically — a failed install here just gives you a shorter sweep, not a crash.

## F3. Verify the base model can be pulled

TinyLlama-1.1B is ungated (no license click-through) but the first `from_pretrained`
call still needs network access to Hugging Face. Verify that works before committing
to the full sweep:

```bash
python -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'); print(c.num_hidden_layers, c.hidden_size, c.num_key_value_heads, c.intermediate_size)"
```

**Expect:** `22 2048 4 5632`. If it fails, you have a Hugging Face connectivity
problem, not a benchmark problem — resolve that before F4.

## F4. Fine-tune smoke test

One tiny config (batch 2, seq 128) that still loads TinyLlama-1.1B — proves the whole
pipeline works end-to-end before committing to the full sweep:

```bash
cd benchmark
python benchmark_finetune.py --smoke-test --output /tmp/smoke_finetune.csv
```

**Expect:** finishes in under 2 minutes (most of that is the base model download and
CUDA context init on first run). Output ends with a one-row summary.

**If it fails on OOM** at 2/128 with 80 GB free: something is wrong with the load path,
not the memory arithmetic — this config is a few GB total.

## F5. Full fine-tune sweep

```bash
python benchmark_finetune.py --output results_finetune.csv 2>&1 | tee sweep_finetune_log.txt
```

**Configs run by default:** 25 rows total (23 without `bitsandbytes` — the two
adam8bit rows in `combined` are auto-skipped). One baseline, 4 batch, 4 seq, 4 rank,
2 checkpointing, 4 adapter placement, 2 base precision, 4 combined.

**Expected runtime on A100 80 GB:** roughly 25–40 minutes. The two checkpointed
combined rows are the slowest (checkpointing adds ~30% step time; the demanding-corner
batch × seq is the largest).

**Expected OOMs — don't treat these as failures:**

- `combined=baseline` and `combined=adam8bit` at batch 32 × seq 2,048 with no
  checkpointing — the activation term predicts ~120 GB, well past the card.
- `seq_len=2048` on its own at batch 8 fits (predicted ~32 GB); `combined` OOMs
  because it also raises the batch to 32.

Every OOM prints `[child] OOM at <lever>=<value>: ...` to stderr, writes
`oom=True` in `results_finetune.csv` with memory columns set to `OOM`, and the sweep
keeps going — same behavior as `benchmark.py`.

## F6. Charts

```bash
python plot_finetune_results.py --input results_finetune.csv --outdir charts_finetune/
```

**Expect:** finishes in a few seconds (no GPU). Produces five fine-tune-specific
charts as PNG + SVG:

- `adapter_placement.{png,svg}` — the backward-reach story.
- `rank_flatness.{png,svg}` — two-panel; static columns grow linearly, activations
  are flat.
- `predicted_vs_measured.{png,svg}` — scatter colored by lever, ±10% band shaded.
- `autocast_cache_residual.{png,svg}` — the "bf16 base collapses Article 1's cache"
  story.
- `component_stack.{png,svg}` — which line item moves under each lever.

## F7. Validate

```bash
python validate_finetune_results.py --input results_finetune.csv
```

**Acceptance bar:** ≤10% error on `max_allocated_gb` for every non-OOM row, and a
correct OOM call on every row. Same rules as Article 1's validator; different columns.

**If a row fails:** the formula in `predictions_finetune.py` is off for that config —
diagnostic, not a script bug. First-pass sweeps often find one or two rows that miss,
and those become the second-pass fixture for the next formula iteration.

## F8. Optional allocator-history diagnostic

Answers three empirical questions the article's claims rest on:

```bash
python debug_finetune.py > debug_finetune_out.json
```

**Probes:**
1. Per-layer working-set decomposition at baseline (confirms the SwiGLU/GQA term
   structure).
2. Autocast-cache residual under bf16 base (compares to Article 1's ~2 GB).
3. Adapter-placement backward reach (asserts frozen lower layers get no gradient).

Commit the resulting `debug_finetune_out.json` alongside `results_finetune.csv` in
`benchmark/reference-run-finetune/` if the sweep validates; that becomes the tracked
evidence for Article 2's claims, mirroring `benchmark/reference-run/` for Article 1.

## F9. Where output lands

- `benchmark/results_finetune.csv` — full sweep data
- `benchmark/sweep_finetune_log.txt` — console log from F5
- `benchmark/charts_finetune/*.{png,svg}` — the five charts
- `benchmark/debug_finetune_out.json` — allocator-history diagnostic (if you ran F8)

Pull `results_finetune.csv` and `charts_finetune/` back for Phase B; everything else
in `benchmark/` is source.

## Appendix: SSH-based fallback runner

Not the recommended path — kept only for environments without CAI. From a workstation
with SSH access to a GPU host:

```bash
./run_finetune_sweep_remote.sh <user@host> <run-label>
```

That script mirrors `run_sweep_remote.sh`: 4-phase SSH-based flow that clones, installs,
runs the smoke test + full sweep + charts + validate, then `scp`s
`results_finetune.csv`, `sweep_finetune_log.txt`, and `charts_finetune/` back into
`runs/<label>-finetune/` locally. Prefer F1–F9 above in a CAI session.

