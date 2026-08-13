#!/usr/bin/env python3
"""Peak-GPU-memory-during-fine-tuning benchmark for LoRA-on-TinyLlama-1.1B.

Sibling of benchmark.py -- same parent/child subprocess architecture, same JSON-line
child contract, same OOM handling, same profiling-step trick. Different sweep, different
CSV schema, different formula module.

Parent mode: build the sweep, spawn one fresh subprocess per config, collect one JSON
line per child, write results_finetune.csv, print a grouped summary.

Child mode: build TinyLlama + LoRA on CUDA, run config.steps + 1 optimizer steps on
synthetic input_ids, measure peak CUDA memory, print one JSON line to stdout.

Everything below is designed to leave benchmark.py untouched.
"""
import argparse
import csv
import dataclasses
import json
import statistics
import subprocess
import sys
import time

import torch

from predictions_finetune import GB, predict_line_items_finetune
from sweep_config_finetune import (
    Config,
    build_all_configs_finetune,
    make_baseline_finetune,
)

CSV_FIELDS = [
    "lever", "lever_value",
    "base_model_name", "base_storage_precision", "lora_rank",
    "lora_target_modules", "lora_adapter_layers",
    "batch", "seq_len", "precision", "optimizer", "checkpointing", "steps",
    # Architecture derived from the loaded base
    "hidden_size", "num_hidden_layers", "num_attention_heads",
    "num_key_value_heads", "intermediate_size", "vocab_size",
    "n_frozen_params", "n_trainable_params", "n_layers_needing_backward",
    # Predicted line items
    "predicted_frozen_weights_gb", "predicted_adapter_weights_gb",
    "predicted_gradients_gb", "predicted_optimizer_gb",
    "predicted_autocast_weight_cache_gb", "predicted_activations_gb",
    "predicted_allocated_gb", "predicted_reserved_gb", "predicted_oom",
    # Measured
    "oom",
    "max_allocated_gb", "max_reserved_gb", "ratio",
    "measured_param_bytes", "measured_frozen_param_bytes",
    "measured_trainable_param_bytes", "measured_grad_bytes",
    "measured_optimizer_bytes",
    "alloc_after_model_gb", "alloc_after_optimizer_gb", "peak_forward_gb",
    "measured_activations_gb",
    # Timing
    "wall_time_s", "step_time_mean_ms", "step_time_median_ms", "final_loss",
]


def run_child(config):
    """Runs one config on CUDA and prints one JSON result line to stdout."""
    # Lazy imports: the parent doesn't need transformers/peft; only the child does. Keeps
    # `--help` fast and avoids importing PEFT into every subprocess invocation of the
    # parent for progress printing.
    from train_finetune import run_training_finetune

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    result = {
        "oom": False,
        "n_trainable_params": None, "n_frozen_params": None,
        "max_allocated_bytes": None, "max_reserved_bytes": None,
        "wall_time_s": None, "step_times_ms": [], "final_loss": None,
        "measured_param_bytes": None,
        "measured_trainable_param_bytes": None,
        "measured_frozen_param_bytes": None,
        "measured_grad_bytes": None, "measured_optimizer_bytes": None,
        "alloc_after_model_gb": None, "alloc_after_optimizer_gb": None,
        "peak_forward_gb": None, "base_arch": None,
    }
    try:
        t0 = time.perf_counter()
        train_result = run_training_finetune(config)
        result["wall_time_s"] = time.perf_counter() - t0
        for k in [
            "n_trainable_params", "n_frozen_params",
            "max_allocated_bytes", "max_reserved_bytes",
            "step_times_ms", "final_loss",
            "measured_param_bytes", "measured_trainable_param_bytes",
            "measured_frozen_param_bytes",
            "measured_grad_bytes", "measured_optimizer_bytes",
            "alloc_after_model_gb", "alloc_after_optimizer_gb",
            "peak_forward_gb", "base_arch",
        ]:
            result[k] = train_result[k]
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[child] OOM at {config.lever}={config.lever_value}: {e}", file=sys.stderr)
            result["oom"] = True
            # Base arch may not have made it into the result yet; the parent will
            # use the predictions module with an approximate arch from the config's
            # base_model_name if possible.
        else:
            raise

    print(json.dumps(result))


def run_one_config(config, script_path, timeout_s):
    """Spawns a fresh subprocess for one config and returns its parsed JSON result."""
    proc = subprocess.run(
        [sys.executable, script_path, "--child", json.dumps(config.as_dict())],
        capture_output=True, text=True, timeout=timeout_s,
    )
    if proc.stderr:
        for line in proc.stderr.strip().splitlines():
            print(f"  | {line}", file=sys.stderr)
    stdout_lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    if not stdout_lines:
        print(f"  ! no JSON result from child (exit {proc.returncode}); treating as failure",
              file=sys.stderr)
        return {"oom": False, "failed": True, "n_trainable_params": None,
                "n_frozen_params": None, "max_allocated_bytes": None,
                "max_reserved_bytes": None, "step_times_ms": []}
    return json.loads(stdout_lines[-1])


def build_row(config, child_result):
    base_arch = child_result.get("base_arch")
    oom = child_result.get("oom", False)
    crashed = child_result.get("failed", False)

    if base_arch is not None:
        predicted = predict_line_items_finetune(
            n_base=child_result["n_frozen_params"],
            n_adapter=child_result["n_trainable_params"],
            base_storage_precision=config.base_storage_precision,
            precision=config.precision,
            optimizer=config.optimizer,
            batch=config.batch, seq_len=config.seq_len,
            d_model=base_arch["hidden_size"],
            n_layers_total=base_arch["num_hidden_layers"],
            num_heads=base_arch["num_attention_heads"],
            num_kv_heads=base_arch["num_key_value_heads"],
            ff_intermediate=base_arch["intermediate_size"],
            vocab=base_arch["vocab_size"],
            adapter_layers=config.lora_adapter_layers,
            checkpointing=config.checkpointing,
            dropout_p=max(base_arch.get("attention_dropout", 0.0),
                          base_arch.get("hidden_dropout", 0.0)),
        )
        arch_cols = {
            "hidden_size": base_arch["hidden_size"],
            "num_hidden_layers": base_arch["num_hidden_layers"],
            "num_attention_heads": base_arch["num_attention_heads"],
            "num_key_value_heads": base_arch["num_key_value_heads"],
            "intermediate_size": base_arch["intermediate_size"],
            "vocab_size": base_arch["vocab_size"],
            "n_layers_needing_backward": predicted["n_layers_needing_backward"],
        }
    else:
        # OOM before we even got the base built, or child crashed. Skip predictions.
        predicted = {
            "frozen_weights": None, "adapter_weights": None,
            "gradients": None, "optimizer": None, "autocast_weight_cache": None,
            "activations": None, "allocated_total": None, "reserved_total": None,
            "predicted_oom": None, "n_layers_needing_backward": None,
        }
        arch_cols = {k: None for k in (
            "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "intermediate_size", "vocab_size",
            "n_layers_needing_backward",
        )}

    max_allocated_gb = (child_result["max_allocated_bytes"] / GB
                       if child_result.get("max_allocated_bytes") else None)
    max_reserved_gb = (child_result["max_reserved_bytes"] / GB
                      if child_result.get("max_reserved_bytes") else None)
    predicted_allocated_gb = predicted["allocated_total"] / GB if predicted["allocated_total"] else None
    predicted_reserved_gb = predicted["reserved_total"] / GB if predicted["reserved_total"] else None
    ratio = (max_allocated_gb / predicted_allocated_gb
             if (max_allocated_gb and predicted_allocated_gb) else None)

    alloc_after_optimizer_gb = child_result.get("alloc_after_optimizer_gb")
    measured_activations_gb = (
        max_allocated_gb - alloc_after_optimizer_gb
        if (max_allocated_gb is not None and alloc_after_optimizer_gb is not None)
        else None
    )

    def _memory_cell(value_gb):
        if value_gb:
            return round(value_gb, 4)
        return "OOM" if oom else "ERROR" if crashed else None

    step_times = child_result.get("step_times_ms") or []
    row = config.as_dict()
    # lora_target_modules is a tuple; stringify for CSV.
    row["lora_target_modules"] = "|".join(config.lora_target_modules)
    row.update(arch_cols)
    row.update({
        "n_frozen_params": child_result.get("n_frozen_params"),
        "n_trainable_params": child_result.get("n_trainable_params"),
        "predicted_frozen_weights_gb": _round(predicted["frozen_weights"], GB),
        "predicted_adapter_weights_gb": _round(predicted["adapter_weights"], GB),
        "predicted_gradients_gb": _round(predicted["gradients"], GB),
        "predicted_optimizer_gb": _round(predicted["optimizer"], GB),
        "predicted_autocast_weight_cache_gb": _round(predicted["autocast_weight_cache"], GB),
        "predicted_activations_gb": _round(predicted["activations"], GB),
        "predicted_allocated_gb": _round(predicted["allocated_total"], GB),
        "predicted_reserved_gb": _round(predicted["reserved_total"], GB),
        "predicted_oom": predicted["predicted_oom"],
        "oom": oom,
        "max_allocated_gb": _memory_cell(max_allocated_gb),
        "max_reserved_gb": _memory_cell(max_reserved_gb),
        "ratio": round(ratio, 3) if ratio else None,
        "measured_param_bytes": child_result.get("measured_param_bytes"),
        "measured_frozen_param_bytes": child_result.get("measured_frozen_param_bytes"),
        "measured_trainable_param_bytes": child_result.get("measured_trainable_param_bytes"),
        "measured_grad_bytes": child_result.get("measured_grad_bytes"),
        "measured_optimizer_bytes": child_result.get("measured_optimizer_bytes"),
        "alloc_after_model_gb": round(child_result["alloc_after_model_gb"], 4)
                                if child_result.get("alloc_after_model_gb") is not None else None,
        "alloc_after_optimizer_gb": round(alloc_after_optimizer_gb, 4)
                                    if alloc_after_optimizer_gb is not None else None,
        "peak_forward_gb": round(child_result["peak_forward_gb"], 4)
                           if child_result.get("peak_forward_gb") is not None else None,
        "measured_activations_gb": round(measured_activations_gb, 4)
                                   if measured_activations_gb is not None else None,
        "wall_time_s": round(child_result["wall_time_s"], 3)
                       if child_result.get("wall_time_s") else None,
        "step_time_mean_ms": round(statistics.mean(step_times), 3) if step_times else None,
        "step_time_median_ms": round(statistics.median(step_times), 3) if step_times else None,
        "final_loss": round(child_result["final_loss"], 4) if child_result.get("final_loss") else None,
    })
    return row


def _round(value_bytes, unit):
    return round(value_bytes / unit, 4) if value_bytes is not None else None


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in CSV_FIELDS})


def print_summary(rows):
    print("\n=== Summary (grouped by lever) ===")
    levers = sorted(set(r["lever"] for r in rows), key=lambda l: (l != "baseline", l))
    for lever in levers:
        print(f"\n[{lever}]")
        for r in [r for r in rows if r["lever"] == lever]:
            if r["oom"]:
                oom_flag = "  (correctly predicted)" if r.get("predicted_oom") else "  <-- NOT predicted by the formula"
                print(f"  {r['lever_value']:>12}: OOM{oom_flag}")
                continue
            if not isinstance(r["max_allocated_gb"], (int, float)):
                print(f"  {r['lever_value']:>12}: ERROR (child crashed -- see log, not an OOM)")
                continue
            flag = ""
            if r["ratio"] is not None and not (0.9 <= r["ratio"] <= 1.1):
                flag = "  <-- predicted allocated outside +/-10% of measured"
            print(f"  {r['lever_value']:>12}: allocated={r['max_allocated_gb']:.2f} GB  "
                  f"reserved={r['max_reserved_gb']:.2f} GB  predicted={r['predicted_allocated_gb']:.2f} GB  "
                  f"ratio={r['ratio']}{flag}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--child", metavar="JSON", help=argparse.SUPPRESS)
    p.add_argument("--smoke-test", action="store_true",
                   help="one tiny GPU config, <2 min (still loads TinyLlama)")
    p.add_argument("--batch-values", type=str, default="4,8,16,32")
    p.add_argument("--seq-values", type=str, default="256,512,1024,2048")
    p.add_argument("--rank-values", type=str, default="4,8,16,64")
    p.add_argument("--checkpointing-values", type=str, default="False,True")
    p.add_argument("--placement-values", type=str, default="all,upper-11,upper-6,upper-3")
    p.add_argument("--base-precision-values", type=str, default="bf16,fp32")
    p.add_argument("--output", type=str, default="results_finetune.csv")
    p.add_argument("--timeout", type=int, default=900,
                   help="seconds allowed per config subprocess")
    return p.parse_args()


def _parse_bool_list(s):
    return [v.strip().lower() == "true" for v in s.split(",")]


def _bitsandbytes_available():
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def main():
    args = parse_args()

    if args.child is not None:
        raw = json.loads(args.child)
        # Tuple round-trips as list through JSON; the dataclass tolerates a list, but
        # normalize back to tuple so downstream comparisons behave.
        raw["lora_target_modules"] = tuple(raw["lora_target_modules"])
        run_child(Config(**raw))
        return

    if not torch.cuda.is_available():
        print("CUDA is not available -- this benchmark measures GPU memory and requires a GPU.",
              file=sys.stderr)
        sys.exit(1)

    baseline = make_baseline_finetune()

    if args.smoke_test:
        # Smallest config that still loads TinyLlama and exercises the whole pipeline.
        # Kept realistic-shaped (batch 2, seq 128) rather than tinier so the smoke test
        # would still fail on a genuinely broken configuration.
        configs = [dataclasses.replace(
            baseline, batch=2, seq_len=128,
            lever="smoke_test", lever_value="smoke_test",
        )]
    else:
        configs = build_all_configs_finetune(
            baseline,
            batch_values=tuple(int(v) for v in args.batch_values.split(",")),
            seq_values=tuple(int(v) for v in args.seq_values.split(",")),
            rank_values=tuple(int(v) for v in args.rank_values.split(",")),
            checkpointing_values=tuple(_parse_bool_list(args.checkpointing_values)),
            placement_values=tuple(args.placement_values.split(",")),
            base_precision_values=tuple(args.base_precision_values.split(",")),
        )

    # Filter out configs that need bitsandbytes if it isn't importable, same pattern as
    # benchmark.py. Two gates: adam8bit AND int8/int4 base precision.
    if not _bitsandbytes_available():
        n_before = len(configs)
        skipped_reasons = []
        remaining = []
        for c in configs:
            if c.optimizer == "adam8bit":
                skipped_reasons.append(f"{c.lever}={c.lever_value}: adam8bit")
                continue
            if c.base_storage_precision in ("int8", "int4"):
                skipped_reasons.append(f"{c.lever}={c.lever_value}: base_storage={c.base_storage_precision}")
                continue
            remaining.append(c)
        if skipped_reasons:
            print(f"bitsandbytes not installed -- skipping {n_before - len(remaining)} config(s):",
                  file=sys.stderr)
            for r in skipped_reasons:
                print(f"  - {r}", file=sys.stderr)
            configs = remaining

    script_path = __file__
    rows = []
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {config.lever}={config.lever_value} "
              f"(batch={config.batch} seq={config.seq_len} rank={config.lora_rank} "
              f"placement={config.lora_adapter_layers} base={config.base_storage_precision} "
              f"{config.precision} {config.optimizer} ckpt={config.checkpointing})",
              file=sys.stderr)
        child_result = run_one_config(config, script_path, args.timeout)
        row = build_row(config, child_result)
        rows.append(row)
        if row["oom"]:
            status = "OOM"
        elif isinstance(row["max_allocated_gb"], (int, float)):
            status = f"{row['max_allocated_gb']:.2f} GB allocated"
        else:
            status = "ERROR (child crashed -- see stderr above, not an OOM)"
        print(f"  -> {status}", file=sys.stderr)

    write_csv(rows, args.output)
    print(f"\nWrote {len(rows)} rows to {args.output}")
    print_summary(rows)


if __name__ == "__main__":
    main()
