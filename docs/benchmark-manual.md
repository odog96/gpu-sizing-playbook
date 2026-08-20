# Manual: GPU Training Memory Benchmark

What this is: a benchmark that measures **peak GPU memory during training** for a
~1B-parameter transformer, across a set of one-lever-at-a-time config changes, on a
single 80GB GPU (the original validation sweep ran on an A100 80GB; the current
2026-07-31 reference sweep in `benchmark/reference-run/` ran on an H100 80GB). It validates the formulas in
`../articles/1-training/article-1-training.md` ("Training: The Four Line Items of GPU
Memory") by comparing predicted vs. measured memory.

It does **not** measure model quality, throughput as a performance claim, or anything
about fine-tuning/inference — those are separate, later work. Training data is random
noise; only its shape and dtype matter.

---

## 1. Codebase map

```
benchmark/
├── model.py          transformer architecture, built from config
├── predictions.py     per-tensor memory formulas (weights/grads/optimizer/autocast-cache/activations)
├── memory_accounting.py  direct measurement helpers (measured_param/grad/optimizer_bytes), device-agnostic
├── sweep_config.py     Config dataclass + sweep generation (baseline, batch, seq, optimizer, checkpointing, combined)
├── train.py            GPU training loop (CUDA only)
├── benchmark.py         orchestrator + subprocess entry point (the main script you run)
├── plot_results.py      reads results.csv, writes 4 charts (no GPU needed)
├── validate_results.py  checks results.csv predicted-vs-measured error and OOM calls (no GPU needed)
├── fixtures/
│   └── results_sample.csv   plausible CSV (built from a real sweep's shape) for testing plot_results.py without a GPU
└── tests/                unittest suite for everything that doesn't require a GPU (23 tests, ~5s)

(GPU commands, runtimes, expected OOMs live in `../docs/benchmark-runbook.md`.)
```

### Module responsibilities

| File | What it does | Depends on |
|---|---|---|
| `model.py` | `TinyTransformer` (pre-norm encoder, sinusoidal positions, no learned params in the positional encoding so param count stays exactly `12·d²·L` + embeddings). `count_parameters()` = actual count; `analytical_param_count()` = formula. | — |
| `predictions.py` | Named per-tensor formulas (weights/gradients/optimizer/autocast-weight-cache/activations), at 1 GB = 1e9 bytes. Params/gradients/optimizer state are always fp32 in this codebase -- autocast never casts storage, only forward compute -- so those formulas don't take a precision argument; only the activation formula and the autocast-cache term vary by precision. `predict_line_items()` bundles everything into `allocated_total` (compare to `max_allocated_gb`) and `reserved_total` (compare to `max_reserved_gb`; includes a documented CUDA-context + fragmentation allowance and drives `predicted_oom`). | — |
| `memory_accounting.py` | `measured_param_bytes`/`measured_grad_bytes`/`measured_optimizer_bytes`: real sums over real tensors (`element_size()*nelement()`), device-agnostic so they're exercised on CPU by the unit tests, not just live GPU runs. Used to validate the static formulas directly instead of inferring them by subtraction. | — |
| `sweep_config.py` | `Config` dataclass (every model/training knob + a `lever`/`lever_value` tag for CSV grouping; `precision` is `"amp_bf16"` or `"fp32"`). `make_baseline()`, four single-lever sweep functions, `combined_sweep()` (multi-lever, by design), `fp32_reference()`, and `build_all_configs()` which assembles all 17. | — |
| `train.py` | `run_training(config)`: builds the model on CUDA, synthetic random tokens/targets, `config.steps` optimizer steps plus one extra untimed profiling step, discards the first 2 as warmup, times the rest. `precision="amp_bf16"` wraps the forward pass in `torch.autocast`; `precision="fp32"` runs everything in plain fp32 -- neither mode casts parameter storage. Returns direct measurements (`measured_param_bytes`, `measured_grad_bytes`, `measured_optimizer_bytes`, `alloc_after_model_gb`, `alloc_after_optimizer_gb`, `peak_forward_gb`) alongside the true overall peak allocated/reserved bytes. GPU only -- there is no CPU code path. | `model.py`, `memory_accounting.py` |
| `benchmark.py` | **Parent mode**: builds the sweep, spawns one fresh subprocess per config (`python benchmark.py --child '<json>'`) so CUDA allocator state can't leak between configs, collects each child's JSON line, writes `results.csv`, prints a grouped summary. **Child mode**: builds one model, runs it, catches `OutOfMemoryError` and records `oom=True` instead of crashing, prints one JSON line to stdout (everything else goes to stderr). | `model.py`, `predictions.py`, `sweep_config.py`, `train.py` |
| `plot_results.py` | Loads `results.csv` via pandas, renders 4 matplotlib charts as PNG (200dpi) + SVG. OOM rows are drawn as hatched bars at the capacity ceiling, never dropped. Single `PALETTE` dict at the top for restyling. | `results.csv` (or the fixture) |
| `validate_results.py` | Reads `results.csv`, prints a per-row predicted/measured/error/OOM-call table, exits nonzero if any non-OOM row exceeds 10% error or any OOM call is wrong. Reads the CSV's own `predicted_allocated_gb`/`predicted_oom` columns -- doesn't recompute anything itself. | `results.csv` |

### The sweep (24 configs)

Baseline: d=2048, 20 layers, batch 256, seq 100, amp_bf16 (autocast bf16 compute, fp32
storage — see `predictions.py`'s module docstring), standard Adam, no checkpointing.
Every sweep below changes exactly **one** field from that baseline (enforced by
`tests/test_sweep_config.py`); the combined sweep and the two checkpointing crosses are
the deliberate exceptions — they stack levers on purpose.

| Lever | Values | Configs |
|---|---|---|
| `baseline` | — | 1 |
| `batch_size` | 256, 1024, 4096 | 3 |
| `seq_len` | 100, 512, 1024, 2048 | 4 |
| `optimizer` | adam, adam8bit | 2 (1 without bitsandbytes) |
| `checkpointing` | off, on (at baseline batch/seq) | 2 |
| `checkpointing_batch` | checkpointing=True at every batch_size value (256, 1024, 4096) | 3 |
| `checkpointing_seq` | checkpointing=True at every seq_len value (100, 512, 1024, 2048) | 4 |
| `combined` | baseline / 8bit / checkpoint / both, all at batch=4096, seq=2048 | 4 (2 without bitsandbytes) |
| `precision` | fp32 (vs. the amp_bf16 baseline) | 1 |

**24 total** (**21** without `bitsandbytes` installed — `benchmark.py` detects this at
startup and skips the `adam8bit` rows automatically, no crash). `checkpointing_batch`
and `checkpointing_seq` exist because checkpointing turned out to be the single most
effective lever (the only one whose predicted/measured ratio lands close to 1.0), but
the original 17-config sweep only exercised it at the baseline point and the one extreme
combined point — these two crosses map how far it actually extends the batch/seq
ceiling, not just its effect at one fixed point.

---

## 2. How to run it

### No GPU needed (formulas, sweep enumeration, charts)

Nothing here trains a model — there's no CPU training path in this codebase, only
GPU. These tests cover the parts that don't need a GPU to verify: the parameter-count
formula, the four prediction formulas, sweep enumeration, and chart generation against
a fixture CSV.

```bash
cd benchmark
pip install -r ../requirements.txt          # torch, pandas, matplotlib
python -m unittest discover -s tests        # ~5s, 23 tests
python plot_results.py --input fixtures/results_sample.csv --outdir /tmp/charts
python validate_results.py --input fixtures/results_sample.csv
```
The last command exercises the full charting pipeline against fixture data — useful
for restyling charts without touching a GPU.

### GPU (the real thing)

Full step-by-step is in `benchmark-runbook.md` — that's the authoritative reference with exact
commands, expected runtimes, and which configs are expected to OOM. Short version:

```bash
nvidia-smi                                                    # confirm GPU visible
python -c "import torch; print(torch.cuda.is_available())"  # confirm CUDA torch
pip install -r ../requirements-optional.txt                  # optional: 8-bit Adam

python benchmark.py --smoke-test                             # <1 min sanity check
python benchmark.py --output results.csv                     # full sweep, ~10-20 min
python plot_results.py --input results.csv --outdir charts/  # generate real charts
python validate_results.py --input results.csv               # check predicted vs. measured error + OOM calls
```

### CLI flags (`benchmark.py`)

| Flag | Default | Purpose |
|---|---|---|
| `--smoke-test` | off | one tiny GPU config (d=256, 2 layers, batch 32), <1 min |
| `--d` / `--layers` | 2048 / 20 | override baseline model size |
| `--batch-values` | `256,1024,4096` | batch sweep list |
| `--seq-values` | `100,512,1024,2048` | sequence-length sweep list |
| `--optimizer-values` | `adam,adam8bit` | optimizer sweep list |
| `--checkpointing-values` | `False,True` | checkpointing sweep list |
| `--output` | `results.csv` | where the CSV lands |
| `--timeout` | 900 | seconds allowed per config subprocess |
| `--child '<json>'` | — | internal; don't pass this by hand |

### `results.csv` columns

Config fields (`d`, `layers`, `batch`, `seq_len`, `precision`, `optimizer`,
`checkpointing`, ...) + `lever`/`lever_value` + `n_params`, then:

- **Predicted** (`predictions.py`, per-tensor formulas): `predicted_weights_gb`,
  `predicted_gradients_gb`, `predicted_optimizer_gb`, `predicted_autocast_weight_cache_gb`
  (nonzero only under `amp_bf16`), `predicted_activations_gb`, `predicted_allocated_gb`
  (non-checkpointed: sum of the above; checkpointed: max of three phase peaks --
  backward-recompute, optimizer-step, forward -- see `predict_line_items` in
  `predictions.py`; compare to `max_allocated_gb`), `predicted_reserved_gb`
  (`predicted_allocated_gb` + a documented CUDA-context + fragmentation allowance --
  compare to `max_reserved_gb`), `predicted_oom` (`predicted_reserved_gb` over the
  fleet's ~79.25 GiB usable capacity).
- **Measured, direct** (`memory_accounting.py` + `train.py` instrumentation, not
  inferred): `measured_param_bytes`, `measured_grad_bytes`, `measured_optimizer_bytes`
  (real sums over real tensors), `alloc_after_model_gb`, `alloc_after_optimizer_gb`,
  `peak_forward_gb` (allocator snapshots at specific points in the run), and
  `measured_activations_gb` (= `max_allocated_gb` - `alloc_after_optimizer_gb`, a
  subtraction of two *measured* quantities, not a measured total minus a predicted one).
- **Measured, peak**: `max_allocated_gb`, `max_reserved_gb`, `ratio`
  (`max_allocated_gb` / `predicted_allocated_gb`), `oom`.
- **Timing**: `wall_time_s`, `step_time_mean_ms`, `step_time_median_ms`.

`oom=True` is a valid result (OOM boundaries are a finding, not a failure); a genuine
crash shows as `ERROR` in the memory columns instead, so the two aren't confused. Note
each config now runs `steps + 1` optimizer steps internally (one extra, untimed
profiling step used to isolate `peak_forward_gb` without disturbing the true overall
peak) -- `step_time_mean_ms`/`step_time_median_ms` are unaffected, since that extra
step's timing is excluded from both.

### Re-running charts against different data

`plot_results.py` never touches the GPU or imports `torch` — point it at any CSV with
the same columns (the real `results.csv`, `fixtures/results_sample.csv`, a filtered
subset, whatever):

```bash
python plot_results.py --input results.csv --outdir charts/
```

That's also how `tests/test_plot_results.py` verifies chart generation without a GPU.
To restyle, edit the single `PALETTE` dict at the top of `plot_results.py` — nothing
else in the file references colors directly.

---

## 3. Where things are headed if you extend this

Fine-tuning (Article 2) is the first extension of this benchmark and lives in a
parallel set of sibling scripts — see § 4 below. Inference (Article 3, coming later)
will follow the same pattern: new sibling files (`predictions_inference.py`,
`benchmark_inference.py`, etc.), not new flags bolted onto `benchmark.py`.

The rule Article 2 established, worth carrying forward: sibling scripts share the
subprocess-per-config discipline, the JSON-line child contract, the direct-tensor
measurement approach, and the CSV+charts+validate triple. Everything else (formulas,
sweep, model construction) is domain-specific and gets its own file.

---

## 4. Fine-tuning sweep

Article 2's benchmark: LoRA fine-tuning of TinyLlama-1.1B on A100 80 GB. Sibling
scripts to §1's training set; nothing in §1 was modified to add this.

```
benchmark/
├── predictions_finetune.py       fine-tune formulas: 5 line items + adapter placement
├── sweep_config_finetune.py       fine-tune Config + 6 sweeps + build_all_configs_finetune
├── model_finetune.py              build_finetune_model (TinyLlama + PEFT LoRA wrapper)
├── memory_accounting_finetune.py  trainable/frozen split helpers
├── train_finetune.py              GPU fine-tune training loop
├── benchmark_finetune.py           parent/child orchestrator (the main fine-tune script)
├── plot_finetune_results.py        5 fine-tune-specific charts
├── validate_finetune_results.py    predicted-vs-measured + OOM cross-check
├── debug_finetune.py               allocator-history diagnostic (3 targeted probes)
├── fixtures/results_finetune_sample.csv   plausible fixture for testing plots without a GPU
```

### Fine-tune sweep (25 configs by default)

Baseline: TinyLlama-1.1B, bf16 base storage, LoRA rank 8, all seven target modules
(`q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj`), adapters on all layers,
batch 8, seq 512, amp_bf16 compute, Adam, no checkpointing. Every sweep below changes
exactly **one** field from this baseline (enforced by
`tests/test_sweep_config_finetune.py`); `combined` is the deliberate exception.

| Lever | Values | Configs |
|---|---|---|
| `baseline` | — | 1 |
| `batch_size` | 4, 8, 16, 32 | 4 |
| `seq_len` | 256, 512, 1024, 2048 | 4 |
| `lora_rank` | 4, 8, 16, 64 | 4 |
| `checkpointing` | off, on | 2 |
| `adapter_placement` | all, upper-11, upper-6, upper-3 | 4 |
| `base_precision` | bf16, fp32 (+ int8/int4 if `bitsandbytes` is installed) | 2 |
| `combined` | baseline / adam8bit / checkpointing / both, all at batch 32 × seq 2,048 | 4 (2 without bitsandbytes) |

**25 total** (**23** without `bitsandbytes` — `benchmark_finetune.py` detects this at
startup and skips the `adam8bit` and INT8/INT4 rows automatically, no crash).

### CLI flags (`benchmark_finetune.py`)

| Flag | Default | Purpose |
|---|---|---|
| `--smoke-test` | off | one tiny config (batch 2, seq 128, still loads TinyLlama), <2 min |
| `--batch-values` | `4,8,16,32` | batch sweep list |
| `--seq-values` | `256,512,1024,2048` | seq_len sweep list |
| `--rank-values` | `4,8,16,64` | LoRA rank sweep list |
| `--checkpointing-values` | `False,True` | checkpointing sweep list |
| `--placement-values` | `all,upper-11,upper-6,upper-3` | adapter placement sweep list |
| `--base-precision-values` | `bf16,fp32` | base storage precision sweep list |
| `--output` | `results_finetune.csv` | where the CSV lands |
| `--timeout` | 900 | seconds allowed per config subprocess |
| `--child '<json>'` | — | internal; don't pass by hand |

### `results_finetune.csv` columns

Config fields (`base_model_name`, `base_storage_precision`, `lora_rank`,
`lora_target_modules`, `lora_adapter_layers`, `batch`, `seq_len`, `precision`,
`optimizer`, `checkpointing`, `steps`) + `lever`/`lever_value`, then:

- **Architecture** (derived from the loaded base, not overridden): `hidden_size`,
  `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`,
  `intermediate_size`, `vocab_size`, `n_layers_needing_backward`.
- **Param counts**: `n_frozen_params`, `n_trainable_params`.
- **Predicted** (`predictions_finetune.py`): `predicted_frozen_weights_gb`,
  `predicted_adapter_weights_gb`, `predicted_gradients_gb`, `predicted_optimizer_gb`,
  `predicted_autocast_weight_cache_gb`, `predicted_activations_gb`,
  `predicted_allocated_gb`, `predicted_reserved_gb`, `predicted_oom`.
- **Measured, direct** (`memory_accounting_finetune.py` + `train_finetune.py`):
  `measured_param_bytes`, `measured_frozen_param_bytes`,
  `measured_trainable_param_bytes`, `measured_grad_bytes`, `measured_optimizer_bytes`,
  `alloc_after_model_gb`, `alloc_after_optimizer_gb`, `peak_forward_gb`,
  `measured_activations_gb`.
- **Measured, peak**: `max_allocated_gb`, `max_reserved_gb`, `ratio`, `oom`.
- **Timing**: `wall_time_s`, `step_time_mean_ms`, `step_time_median_ms`, `final_loss`.

Same conventions as Article 1's CSV: `oom=True` is a valid finding, not a failure;
`ERROR` in memory columns means the child crashed for a reason other than OOM.
`lora_target_modules` is serialized as a `|`-joined string.

### No GPU needed (formulas, sweep enumeration, charts)

```bash
cd benchmark
pip install -r ../requirements.txt          # torch, pandas, matplotlib
# NOTE: transformers/peft NOT needed for the CPU-only test suite. The tests use hand-
# rolled stand-ins for the LoRA structure so they run without pulling ~1 GB of HF deps.
python -m unittest discover -s tests
python plot_finetune_results.py --input fixtures/results_finetune_sample.csv --outdir /tmp/charts_ft
python validate_finetune_results.py --input fixtures/results_finetune_sample.csv
```

### GPU (the real thing)

Full step-by-step is in `benchmark-runbook.md` § "Fine-tuning sweep" (F1–F9). Short version:

```bash
cd benchmark
python benchmark_finetune.py --smoke-test                              # <2 min sanity check
python benchmark_finetune.py --output results_finetune.csv             # full sweep, ~25-40 min on A100 80GB
python plot_finetune_results.py --input results_finetune.csv --outdir charts_finetune/
python validate_finetune_results.py --input results_finetune.csv
python debug_finetune.py > debug_finetune_out.json                     # optional; allocator-history diagnostic
```

