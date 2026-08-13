#!/usr/bin/env python3
"""Charts for results_finetune.csv. Runs anywhere with no GPU -- pure matplotlib over a CSV.

Only fine-tune-specific phenomena are charted here; charts Article 1 already produced are
skipped unless Phase B decides to regenerate them for continuity.

Five charts, each as PNG (200 dpi) and SVG:
  1. adapter_placement -- peak allocated across all/upper-11/upper-6/upper-3; the "backward
     reach" story.
  2. rank_flatness -- static columns (adapter weights + gradients + optimizer) plotted
     against LoRA rank, alongside activations; shows adapter columns growing linearly while
     activations are flat -- the article's rank-is-a-nearly-free lever claim.
  3. predicted_vs_measured -- scatter colored by lever, so each lever's residual is
     visible; the pattern that a lever produces predicts where formula drift will show up.
  4. autocast_cache_residual -- one bar for baseline (~0.013 GB) vs Article 1's ~2.0 GB
     reference; the "bf16-base collapses the cache" story.
  5. component_stack -- stacked bars per lever position showing which line item moves.

OOM configs are drawn as hatched bars reaching a capacity ceiling rather than dropped.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PALETTE = {
    "line_items": {
        "frozen_weights": "#2a78d6",     # blue -- the frozen base
        "adapter_weights": "#8a6fd6",    # violet -- the small trainable columns
        "gradients": "#eb6834",          # orange
        "optimizer": "#1baf7a",          # aqua
        "autocast_weight_cache": "#c7c1a3",   # muted -- collapses to megabytes here
        "activations": "#eda100",        # yellow
    },
    "measured": "#0b0b0b",
    "reserved": "#52514e",
    "predicted_floor": "#52514e",
    "capacity_line": "#d03b3b",
    "oom": "#d03b3b",
    "text_muted": "#898781",
    "gridline": "#e1e0d9",
    "axis": "#c3c2b7",
    # Lever colors for the predicted_vs_measured scatter.
    "levers": {
        "baseline": "#0b0b0b",
        "batch_size": "#2a78d6",
        "seq_len": "#eb6834",
        "lora_rank": "#1baf7a",
        "checkpointing": "#eda100",
        "adapter_placement": "#8a6fd6",
        "base_precision": "#d03b3b",
        "combined": "#52514e",
    },
}

# A100 80GB and H100 80GB both report 79.18 GiB usable; using the same 80 GB ceiling as
# Article 1's charts for continuity.
GPU_80GB_CAPACITY_GB = 80.0

LINE_ITEM_ORDER = [
    "frozen_weights", "adapter_weights",
    "gradients", "optimizer", "autocast_weight_cache", "activations",
]
LINE_ITEM_LABELS = {
    "frozen_weights": "Frozen base weights",
    "adapter_weights": "Adapter weights",
    "gradients": "Gradients",
    "optimizer": "Optimizer",
    "autocast_weight_cache": "Autocast cache",
    "activations": "Activations",
}


def _style_axes(ax, ylabel="GB"):
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", color=PALETTE["gridline"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["axis"])
    ax.tick_params(axis="both", labelsize=10.5, colors=PALETTE["text_muted"])
    ax.tick_params(axis="x", colors="#0b0b0b")


def _save(fig, outdir, name):
    fig.tight_layout()
    png = os.path.join(outdir, f"{name}.png")
    svg = os.path.join(outdir, f"{name}.svg")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def _safe_float(v):
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _hatch_oom_bar(ax, x, ceiling, width):
    ax.bar(x, ceiling, width=width, facecolor="none",
           edgecolor=PALETTE["oom"], hatch="////", linewidth=1.2, zorder=3)
    ax.text(x, ceiling * 1.01, "OOM", ha="center", va="bottom",
            fontsize=9, color=PALETTE["oom"], fontweight="bold")


def plot_adapter_placement(df, outdir):
    """The 'backward reach' story: activation memory drops with each cut to placement,
    the four static columns barely move at all."""
    sub = df[df["lever"] == "adapter_placement"].copy()
    if sub.empty:
        return None
    order = ["all", "upper-11", "upper-6", "upper-3"]
    sub["_order"] = sub["lever_value"].apply(lambda v: order.index(v) if v in order else 99)
    sub = sub.sort_values("_order")
    x_labels = sub["lever_value"].tolist()
    x = range(len(x_labels))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bottom = [0.0] * len(sub)
    for item in LINE_ITEM_ORDER:
        col = f"predicted_{item}_gb"
        values = sub[col].fillna(0).tolist()
        ax.bar(x, values, bottom=bottom, width=0.55,
               color=PALETTE["line_items"][item],
               label=LINE_ITEM_LABELS[item], zorder=2,
               edgecolor="white", linewidth=0.5)
        bottom = [b + v for b, v in zip(bottom, values)]

    measured = [_safe_float(v) for v in sub["max_allocated_gb"]]
    meas_x = [xi for xi, v in zip(x, measured) if v is not None]
    meas_y = [v for v in measured if v is not None]
    ax.plot(meas_x, meas_y, marker="D", markersize=6, linewidth=1.5,
            color=PALETTE["measured"], label="Measured (allocated)", zorder=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Adapter placement (layers with backward reach)", fontsize=11)
    ax.set_title("Adapter placement: fewer layers of backward reach, less activation memory",
                 fontsize=13, pad=12)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, outdir, "adapter_placement")


def plot_rank_flatness(df, outdir):
    """Rank moves adapter columns linearly; activations don't move at all. Two-panel."""
    sub = df[df["lever"] == "lora_rank"].copy()
    if sub.empty:
        return None
    baseline = df[df["lever"] == "baseline"].copy()
    if not baseline.empty:
        # Also include the baseline (rank=8) so the sweep has all four ranks visible.
        baseline["lora_rank"] = baseline["lora_rank"].astype(int)
        baseline_row = baseline.iloc[[0]].copy()
        baseline_row["lever_value"] = baseline_row["lora_rank"].astype(str)
        sub = pd.concat([sub, baseline_row], ignore_index=True)
    sub["lora_rank"] = sub["lora_rank"].astype(int)
    sub = sub.sort_values("lora_rank")
    x = sub["lora_rank"].tolist()

    fig, (ax_static, ax_dyn) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left panel: three trainable columns, one line each -- all linear in rank.
    for item, color in (
        ("adapter_weights", PALETTE["line_items"]["adapter_weights"]),
        ("gradients", PALETTE["line_items"]["gradients"]),
        ("optimizer", PALETTE["line_items"]["optimizer"]),
    ):
        y = [_safe_float(v) or 0.0 for v in sub[f"predicted_{item}_gb"]]
        # Convert to MB for readability -- these are megabytes, not gigabytes.
        y_mb = [v * 1000 for v in y]
        ax_static.plot(x, y_mb, marker="o", linewidth=1.5, color=color,
                       label=LINE_ITEM_LABELS[item])
    ax_static.set_title("Trainable columns: linear in rank", fontsize=12, pad=8)
    ax_static.set_xlabel("LoRA rank", fontsize=11)
    ax_static.set_ylabel("MB", fontsize=11)
    ax_static.set_xscale("log", base=2)
    ax_static.set_xticks(x)
    ax_static.set_xticklabels([str(v) for v in x])
    _style_axes(ax_static, ylabel="MB")
    ax_static.legend(frameon=False, fontsize=9.5, loc="upper left")

    # Right panel: activations and peak allocated -- flat vs. slight upward drift.
    y_act = [_safe_float(v) or 0.0 for v in sub["predicted_activations_gb"]]
    y_peak = [_safe_float(v) or 0.0 for v in sub["max_allocated_gb"]]
    ax_dyn.plot(x, y_act, marker="s", linewidth=1.5,
                color=PALETTE["line_items"]["activations"], label="Activations")
    ax_dyn.plot(x, y_peak, marker="D", linewidth=1.5,
                color=PALETTE["measured"], label="Peak allocated (measured)")
    ax_dyn.set_title("Activations and total: flat", fontsize=12, pad=8)
    ax_dyn.set_xlabel("LoRA rank", fontsize=11)
    ax_dyn.set_xscale("log", base=2)
    ax_dyn.set_xticks(x)
    ax_dyn.set_xticklabels([str(v) for v in x])
    _style_axes(ax_dyn)
    ax_dyn.legend(frameon=False, fontsize=9.5, loc="lower right")

    fig.suptitle("LoRA rank moves the trainable columns linearly; nothing else moves.",
                 fontsize=13, y=1.02)
    return _save(fig, outdir, "rank_flatness")


def plot_predicted_vs_measured(df, outdir):
    """Scatter colored by lever. Points on the y=x line = perfect prediction."""
    non_oom = df[df["oom"].astype(str).str.lower() != "true"].copy()
    non_oom["_pred"] = non_oom["predicted_allocated_gb"].apply(_safe_float)
    non_oom["_meas"] = non_oom["max_allocated_gb"].apply(_safe_float)
    non_oom = non_oom.dropna(subset=["_pred", "_meas"])
    if non_oom.empty:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    for lever, color in PALETTE["levers"].items():
        rows = non_oom[non_oom["lever"] == lever]
        if rows.empty:
            continue
        ax.scatter(rows["_pred"], rows["_meas"], c=color, s=60,
                   edgecolors="white", linewidth=0.7, label=lever, zorder=3)

    lim = max(non_oom["_pred"].max(), non_oom["_meas"].max()) * 1.05
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.0,
            color=PALETTE["text_muted"], zorder=1)
    # +/-10% band -- the acceptance bar.
    ax.fill_between([0, lim], [0, lim * 0.9], [0, lim * 1.1],
                    color=PALETTE["gridline"], alpha=0.5, zorder=0,
                    label="+/-10% band")

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Predicted allocated (GB)", fontsize=11)
    ax.set_ylabel("Measured allocated (GB)", fontsize=11)
    ax.set_title("Predicted vs. measured, colored by lever", fontsize=13, pad=12)
    ax.grid(color=PALETTE["gridline"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=10.5, colors=PALETTE["text_muted"])
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    return _save(fig, outdir, "predicted_vs_measured")


def plot_autocast_cache_residual(df, outdir):
    """The article's fifth-item story: bf16 base collapses Article 1's ~2 GB cache to
    megabytes."""
    baseline = df[df["lever"] == "baseline"]
    fp32_row = df[(df["lever"] == "base_precision") & (df["lever_value"] == "fp32")]
    if baseline.empty:
        return None

    baseline_cache_gb = _safe_float(baseline.iloc[0]["predicted_autocast_weight_cache_gb"]) or 0
    fp32_cache_gb = _safe_float(fp32_row.iloc[0]["predicted_autocast_weight_cache_gb"]) if not fp32_row.empty else 0
    article1_cache_gb = 2.02  # Article 1's measured baseline cache

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    labels = [
        "Article 1 baseline\n(fp32 base, bf16 compute)",
        "Article 2, fp32 base\n(if you load it fp32)",
        "Article 2 baseline\n(bf16 base, bf16 compute)",
    ]
    values_gb = [article1_cache_gb, fp32_cache_gb, baseline_cache_gb]
    colors = [PALETTE["reserved"], PALETTE["line_items"]["autocast_weight_cache"],
              PALETTE["line_items"]["adapter_weights"]]

    bars = ax.bar(labels, values_gb, color=colors, width=0.55, zorder=2,
                  edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, values_gb):
        text = f"{val:.2f} GB" if val >= 0.05 else f"{val * 1000:.1f} MB"
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(values_gb) * 0.02, text,
                ha="center", va="bottom", fontsize=10, color=PALETTE["measured"])

    ax.set_title("Autocast weight cache: what happens when the base is already bf16",
                 fontsize=13, pad=12)
    _style_axes(ax)
    return _save(fig, outdir, "autocast_cache_residual")


def plot_component_stack(df, outdir):
    """Stacked bars per single-lever sweep endpoint showing which line item moves."""
    # Take one representative row per single-lever sweep plus baseline.
    lever_order = ["baseline", "batch_size", "seq_len", "lora_rank",
                   "checkpointing", "adapter_placement", "base_precision"]
    lever_labels = {"batch_size": "batch=32", "seq_len": "seq=2048", "lora_rank": "rank=64",
                    "checkpointing": "ckpt=True", "adapter_placement": "upper-3",
                    "base_precision": "fp32", "baseline": "baseline"}

    rows = []
    for lever in lever_order:
        sub = df[df["lever"] == lever]
        if sub.empty:
            continue
        if lever == "baseline":
            rows.append(sub.iloc[0])
        else:
            key = lever_labels[lever].split("=")[-1] if "=" in lever_labels[lever] else lever_labels[lever]
            # Pick the row whose lever_value matches the picked lever label suffix.
            match = sub[sub["lever_value"].astype(str) == key]
            rows.append(match.iloc[0] if not match.empty else sub.iloc[-1])

    if not rows:
        return None
    tick_labels = [lever_labels.get(r["lever"], r["lever"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = range(len(rows))
    bottom = [0.0] * len(rows)
    for item in LINE_ITEM_ORDER:
        col = f"predicted_{item}_gb"
        values = [_safe_float(r[col]) or 0.0 for r in rows]
        ax.bar(x, values, bottom=bottom, width=0.55,
               color=PALETTE["line_items"][item],
               label=LINE_ITEM_LABELS[item], zorder=2,
               edgecolor="white", linewidth=0.5)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xticks(list(x))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Single-lever endpoint", fontsize=11)
    ax.set_title("Which line item moves under each lever", fontsize=13, pad=12)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, outdir, "component_stack")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="results_finetune.csv")
    p.add_argument("--outdir", default=".")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)
    df["oom"] = df["oom"].astype(str).str.lower().isin(["true", "1"])

    plot_adapter_placement(df, args.outdir)
    plot_rank_flatness(df, args.outdir)
    plot_predicted_vs_measured(df, args.outdir)
    plot_autocast_cache_residual(df, args.outdir)
    plot_component_stack(df, args.outdir)
    print(f"Charts written to {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
