# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A GPU-memory benchmark (`benchmark/`) that validates the memory-accounting formulas used
in a two-part article series (`article-0-draft.md`, `article-1-draft.md` at repo root).
`article-1-draft.md` ("Training: The Four Line Items of GPU Memory") makes specific
numeric claims about how much VRAM a ~1B-parameter transformer's training run needs; the
benchmark exists to measure a real training loop's peak CUDA memory across a sweep of
configs and check the article's formulas against it. `memory-formulas-for-article.md` is
the working reference doc translating the corrected formulas back into article/spreadsheet
language — treat it as the source of truth for what the article's numbers *should* say,
since the formulas have already gone through one full correction pass (see "History" below).

There is no CPU training path anywhere in this codebase — training requires CUDA. Formula
code, sweep enumeration, and chart generation are CPU-only and unit-testable without a GPU.

## Commands

All commands run from `benchmark/`.

```bash
# CPU-only: formulas, sweep enumeration, param-count math, chart generation
python -m unittest discover -s tests              # full suite
python -m unittest tests.test_predictions          # single module
python -m unittest tests.test_predictions.TestActivationFormula.test_baseline_config_matches_hand_derived_per_tensor_sum  # single test
python plot_results.py --input fixtures/results_sample.csv --outdir /tmp/charts
python validate_results.py --input fixtures/results_sample.csv

# GPU required (see benchmark/RUNBOOK.md for the full walkthrough)
nvidia-smi                                                    # confirm GPU visible first
python benchmark.py --smoke-test                              # <1 min sanity check
python benchmark.py --output results.csv                      # full sweep (~15-25 min)
python plot_results.py --input results.csv --outdir charts/
python validate_results.py --input results.csv                # predicted-vs-measured error + OOM check
```

Dependencies: `pip install -r requirements.txt` (torch, pandas, matplotlib). 8-bit Adam
configs need `pip install -r requirements-optional.txt` (bitsandbytes) separately — it's
proxy-sensitive, so it's kept out of the base install. If it's missing, `benchmark.py`
detects that at startup and skips those config rows rather than crashing.

If `torch.cuda.is_available()` is `False` on a machine with a real GPU, the installed
torch is a CPU-only build — fix with
`pip install torch --index-url https://download.pytorch.org/whl/cu121` (see
`RUNBOOK.md` step 2 for the `cu124` fallback and troubleshooting).

## Architecture

**Two halves that must stay in sync**: `predictions.py` (what memory *should* be, from
formulas) and `train.py`/`memory_accounting.py` (what memory *actually is*, measured on a
running model). `benchmark.py` runs both for the same config and puts them side by side
in one CSV row. Never treat "measured" as derivable from "predicted" or vice versa — the
whole point of this codebase is catching where they diverge.

**Why every config runs in its own subprocess**: `benchmark.py` parent mode enumerates the
sweep and spawns `python benchmark.py --child '<json>'` once per config. This is
deliberate, not incidental — CUDA allocator state (fragmentation, cached blocks) persists
within a process, so running configs back-to-back in one process would let earlier configs
contaminate later ones' measurements. Child mode prints exactly one JSON line to stdout
(everything else — progress, OOM messages — goes to stderr) so the parent can parse
cleanly. An `OutOfMemoryError` in a child is caught and recorded as `oom=True` in the CSV,
not treated as a crash — OOM boundaries are a finding this benchmark exists to locate, not
a failure mode to avoid.

**Direct measurement, not inference**: `memory_accounting.py`'s
`measured_param/grad/optimizer_bytes()` sum real tensor sizes
(`element_size()*nelement()`) by walking the model/optimizer directly, and are
device-agnostic (exercised on CPU in unit tests). `train.py` snapshots allocator state at
specific points (`alloc_after_model_gb`, `alloc_after_optimizer_gb`, `peak_forward_gb`) via
`torch.cuda.max_memory_allocated`/`reset_peak_memory_stats`. `measured_activations_gb` is
computed as `max_allocated_gb - alloc_after_optimizer_gb` — a subtraction of two *measured*
quantities. Don't reintroduce the earlier pattern of inferring one side by subtracting a
*predicted* value from a measured total; that's circular and was the root cause of the bug
described below.

**The profiling step**: each config runs `config.steps + 1` optimizer steps. The extra
step is untimed and exists solely to isolate `peak_forward_gb` (via a
snapshot-before-reset-then-max-combine on `reset_peak_memory_stats`) without corrupting
`step_time_mean_ms`/`step_time_median_ms`, which only average the timed steps.

**Precision reality**: this codebase's `precision` config field only controls whether
`torch.autocast` wraps the forward pass (`amp_bf16`) or not (`fp32`). It does **not** cast
parameter/gradient/optimizer storage — `autocast` only changes compute dtype for
whitelisted ops, never storage dtype. So weights/gradients/optimizer state are fp32 in
*both* precision modes; only `predict_activation_bytes()` and the autocast-weight-cache
term vary by precision. Do not add a precision argument to
`predict_weights/gradients/optimizer_bytes()` — that would misrepresent what the code
actually does.

**OFAT (one-lever-at-a-time) sweep design**: `sweep_config.py` builds every sweep from one
fixed `baseline`, changing exactly one field per sweep so results are directly
interpretable (`SINGLE_LEVER_TAGS` + `differing_fields()` enforce this in tests). The
`combined` sweep and the `checkpointing_batch`/`checkpointing_seq` crosses are the
deliberate, explicitly-tagged exceptions that stack levers on purpose — don't "fix" them
to be single-lever, and don't add new multi-lever sweeps without a similarly explicit tag
and test coverage.

**Formula explainability constraint**: every term in `predictions.py` must be explainable
in one sentence to an engineer reading the article (this drove the whole rewrite — see
History). No curve-fitting, no unexplained constants. Overhead terms that *are* empirical
(`CUDA_CONTEXT_OVERHEAD_GB`, `FRAGMENTATION_GB`, `CHECKPOINTING_FRAGMENTATION_GB`) are kept
as their own named, documented constants rather than folded into other terms or tuned to
force validation passes — if a prediction misses `validate_results.py`'s ±10% bar, that's
signal about the formula, not license to adjust a fragmentation constant until it passes.

**Attention memory dispatch**: `model.py` builds layers from `nn.TransformerEncoderLayer`,
which internally calls `self_attn(..., need_weights=False)` — this dispatches to PyTorch's
fused `scaled_dot_product_attention` (flash/memory-efficient kernel), meaning the full
seq×seq attention probability matrix is never materialized. `predictions.py`'s activation
formula deliberately zeroes that term for this reason; don't add it back in without
confirming the model no longer takes the SDPA path.

## History (why the code looks the way it does)

The formulas in `predictions.py` were rewritten wholesale after a review found the
*original* formulas assumed bf16 mixed-precision storage (2+2+12 bytes/param for
weights+grads+optimizer) that the training loop never actually implements (it's
fp32-everywhere, 4+4+8 bytes/param) — the two schemes happened to sum to the same total by
coincidence, which is why the bug went undetected. The same review found the
non-checkpointed activation formula was accidentally using the *checkpointed* formula's
math (and the checkpointed formula was additionally dividing by layer count on top of
that). Current formulas were validated against a real 17-config GPU sweep: 16/17 rows
landed within ±10% of measured `max_allocated_gb`, with the one miss (checkpointing,
~11.6% over) reported honestly rather than tuned away.

A second correction pass (2026-07-31, on a 24-config H100 sweep) then fixed the
checkpointed formula itself: a CUDA allocator-history replay (`benchmark/debug_ckpt.py`)
showed the saved layer boundaries are fp32 (not compute-dtype — LayerNorm and residual
adds run fp32 under autocast) and that the checkpointed peak is the max of three phase
peaks (backward-recompute, foreach-Adam optimizer-step at 20 bytes/param, forward), not
a sum. All 24 rows now validate (worst non-OOM error 2.2%; every OOM call correct).
`memory-formulas-for-article.md` has the full term-by-term old-vs-new comparison if you
need the history in detail.
