#!/usr/bin/env python3
"""Peak-GPU-memory-during-training benchmark for a ~1B parameter transformer.

Orchestrator + subprocess entry point in one file:
  - Parent mode (default): builds the sweep of configs, runs each in a fresh subprocess
    (so CUDA allocator state can't leak between configs), collects results, writes
    results.csv, prints a summary.
  - Child mode (--child '<json config>'): builds one model, runs config.steps optimizer
    steps on synthetic data, measures peak CUDA memory, prints a single JSON result line
    to stdout. Everything else goes to stderr so the parent can parse stdout cleanly.

The only output that matters is peak VRAM. Loss values are checked only for finiteness
(proof the loop ran), never for quality.
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

from model import analytical_param_count
from predictions import GB, predict_line_items
from sweep_config import Config, build_all_configs, make_baseline
from train import run_training

CSV_FIELDS = [
    "lever", "lever_value",
    "d", "layers", "heads", "ff_mult", "vocab", "batch", "seq_len",
    "precision", "optimizer", "checkpointing", "steps",
    "n_params",
    "predicted_weights_gb", "predicted_gradients_gb", "predicted_optimizer_gb",
    "predicted_autocast_weight_cache_gb", "predicted_activations_gb",
    "predicted_allocated_gb", "predicted_reserved_gb", "predicted_oom",
    "max_allocated_gb", "max_reserved_gb", "ratio",
    "measured_param_bytes", "measured_grad_bytes", "measured_optimizer_bytes",
    "alloc_after_model_gb", "alloc_after_optimizer_gb", "peak_forward_gb",
    "measured_activations_gb",
    "oom", "wall_time_s", "step_time_mean_ms", "step_time_median_ms",
]


def run_child(config):
    """Runs one config on the current CUDA device and prints a JSON result line."""
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    result = {
        "oom": False,
        "n_params": None,
        "max_allocated_bytes": None,
        "max_reserved_bytes": None,
        "wall_time_s": None,
        "step_times_ms": [],
        "final_loss": None,
        "measured_param_bytes": None,
        "measured_grad_bytes": None,
        "measured_optimizer_bytes": None,
        "alloc_after_model_gb": None,
        "alloc_after_optimizer_gb": None,
        "peak_forward_gb": None,
    }
    try:
        t0 = time.perf_counter()
        train_result = run_training(config)
        result["wall_time_s"] = time.perf_counter() - t0
        result["n_params"] = train_result["n_params"]
        result["step_times_ms"] = train_result["step_times_ms"]
        result["final_loss"] = train_result["final_loss"]
        # run_training owns peak tracking (it needs a controlled mid-run reset to isolate
        # forward-pass-only peak; see its docstring), so its returned peak is authoritative
        # rather than re-read here.
        result["max_allocated_bytes"] = train_result["max_allocated_bytes"]
        result["max_reserved_bytes"] = train_result["max_reserved_bytes"]
        result["measured_param_bytes"] = train_result["measured_param_bytes"]
        result["measured_grad_bytes"] = train_result["measured_grad_bytes"]
        result["measured_optimizer_bytes"] = train_result["measured_optimizer_bytes"]
        result["alloc_after_model_gb"] = train_result["alloc_after_model_gb"]
        result["alloc_after_optimizer_gb"] = train_result["alloc_after_optimizer_gb"]
        result["peak_forward_gb"] = train_result["peak_forward_gb"]
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[child] OOM at {config.lever}={config.lever_value}: {e}", file=sys.stderr)
            result["oom"] = True
            # Model construction may not have completed; fall back to the analytical
            # formula so the CSV row still carries predicted line items.
            result["n_params"] = analytical_param_count(config.d, config.layers, config.vocab, config.ff_mult)
        else:
            raise

    print(json.dumps(result))  # the only stdout line -- parent parses exactly this


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
        print(f"  ! no JSON result from child (exit {proc.returncode}); treating as failure", file=sys.stderr)
        return {"oom": False, "n_params": None, "max_allocated_bytes": None,
                "max_reserved_bytes": None, "wall_time_s": None, "step_times_ms": [], "failed": True}
    return json.loads(stdout_lines[-1])


def build_row(config, child_result):
    n_params = child_result["n_params"]
    predicted = predict_line_items(
        n_params, config.precision, config.optimizer,
        config.batch, config.seq_len, config.d, config.layers, config.ff_mult, config.vocab,
        config.checkpointing,
    ) if n_params else {
        "weights": None, "gradients": None, "optimizer": None, "autocast_weight_cache": None,
        "activations": None, "allocated_total": None, "reserved_total": None, "predicted_oom": None,
    }

    # Keep "OOM" (a valid, expected result) distinct from "crashed" (a real bug worth
    # noticing) -- both leave memory columns empty, but only OOM should read as OOM.
    oom = child_result.get("oom", False)
    crashed = child_result.get("failed", False)
    max_allocated_gb = child_result["max_allocated_bytes"] / GB if child_result.get("max_allocated_bytes") else None
    max_reserved_gb = child_result["max_reserved_bytes"] / GB if child_result.get("max_reserved_bytes") else None
    predicted_allocated_gb = predicted["allocated_total"] / GB if predicted["allocated_total"] else None
    predicted_reserved_gb = predicted["reserved_total"] / GB if predicted["reserved_total"] else None
    ratio = (max_allocated_gb / predicted_allocated_gb) if (max_allocated_gb and predicted_allocated_gb) else None

    # Measured (not inferred) activation memory: total peak minus what was already
    # resident right after the first optimizer.step() -- i.e. two real measurements
    # subtracted from each other, not a measured total minus a *predicted* static term.
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
    row.update({
        "n_params": n_params,
        "predicted_weights_gb": _round(predicted["weights"], GB),
        "predicted_gradients_gb": _round(predicted["gradients"], GB),
        "predicted_optimizer_gb": _round(predicted["optimizer"], GB),
        "predicted_autocast_weight_cache_gb": _round(predicted["autocast_weight_cache"], GB),
        "predicted_activations_gb": _round(predicted["activations"], GB),
        "predicted_allocated_gb": _round(predicted["allocated_total"], GB),
        "predicted_reserved_gb": _round(predicted["reserved_total"], GB),
        "predicted_oom": predicted["predicted_oom"],
        "max_allocated_gb": _memory_cell(max_allocated_gb),
        "max_reserved_gb": _memory_cell(max_reserved_gb),
        "ratio": round(ratio, 3) if ratio else None,
        "measured_param_bytes": child_result.get("measured_param_bytes"),
        "measured_grad_bytes": child_result.get("measured_grad_bytes"),
        "measured_optimizer_bytes": child_result.get("measured_optimizer_bytes"),
        "alloc_after_model_gb": round(child_result["alloc_after_model_gb"], 4) if child_result.get("alloc_after_model_gb") is not None else None,
        "alloc_after_optimizer_gb": round(alloc_after_optimizer_gb, 4) if alloc_after_optimizer_gb is not None else None,
        "peak_forward_gb": round(child_result["peak_forward_gb"], 4) if child_result.get("peak_forward_gb") is not None else None,
        "measured_activations_gb": round(measured_activations_gb, 4) if measured_activations_gb is not None else None,
        "oom": oom,
        "wall_time_s": round(child_result["wall_time_s"], 3) if child_result.get("wall_time_s") else None,
        "step_time_mean_ms": round(statistics.mean(step_times), 3) if step_times else None,
        "step_time_median_ms": round(statistics.median(step_times), 3) if step_times else None,
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

    # Activation multiplier: measured activation memory (a real subtraction of two
    # measured quantities -- max_allocated_gb minus alloc_after_optimizer_gb, both read
    # directly off the GPU) vs. the predicted activation floor. Surfaced explicitly
    # because it's the single number this benchmark is really for.
    print("\n[activation multiplier vs. sequence length]  (measured activations / predicted floor)")
    for r in [r for r in rows if r["lever"] == "seq_len" and isinstance(r.get("measured_activations_gb"), (int, float))]:
        mult = r["measured_activations_gb"] / r["predicted_activations_gb"] if r["predicted_activations_gb"] else None
        print(f"  seq_len={r['seq_len']:>5}: measured~={r['measured_activations_gb']:.2f} GB  "
              f"predicted_floor={r['predicted_activations_gb']:.2f} GB  multiplier={mult:.2f}x" if mult else "  n/a")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--child", metavar="JSON", help=argparse.SUPPRESS)  # internal subprocess entry
    p.add_argument("--smoke-test", action="store_true", help="one tiny GPU config, <1 min")
    p.add_argument("--d", type=int, default=2048, help="baseline hidden dim")
    p.add_argument("--layers", type=int, default=20, help="baseline layer count")
    p.add_argument("--batch-values", type=str, default="256,1024,4096")
    p.add_argument("--seq-values", type=str, default="100,512,1024,2048")
    p.add_argument("--optimizer-values", type=str, default="adam,adam8bit")
    p.add_argument("--checkpointing-values", type=str, default="False,True")
    p.add_argument("--output", type=str, default="results.csv")
    p.add_argument("--timeout", type=int, default=900, help="seconds allowed per config subprocess")
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
        run_child(Config(**json.loads(args.child)))
        return

    if not torch.cuda.is_available():
        print("CUDA is not available -- this benchmark measures GPU memory and requires a GPU.", file=sys.stderr)
        sys.exit(1)

    if args.smoke_test:
        configs = [dataclasses.replace(
            make_baseline(d=256, layers=2, heads=4, ff_mult=4, vocab=1000, steps=7),
            batch=32, seq_len=64, lever="smoke_test", lever_value="smoke_test",
        )]
    else:
        baseline = make_baseline(d=args.d, layers=args.layers)
        configs = build_all_configs(
            baseline,
            batch_values=tuple(int(v) for v in args.batch_values.split(",")),
            seq_values=tuple(int(v) for v in args.seq_values.split(",")),
            optimizer_values=tuple(args.optimizer_values.split(",")),
            checkpointing_values=tuple(_parse_bool_list(args.checkpointing_values)),
        )

    if not _bitsandbytes_available():
        skipped = [c for c in configs if c.optimizer == "adam8bit"]
        if skipped:
            print(f"bitsandbytes not installed -- skipping {len(skipped)} adam8bit config(s) "
                  f"(see requirements-optional.txt)", file=sys.stderr)
            configs = [c for c in configs if c.optimizer != "adam8bit"]

    script_path = __file__
    rows = []
    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {config.lever}={config.lever_value} "
              f"(d={config.d} layers={config.layers} batch={config.batch} seq={config.seq_len} "
              f"{config.precision} {config.optimizer} ckpt={config.checkpointing})", file=sys.stderr)
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
