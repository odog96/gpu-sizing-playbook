#!/usr/bin/env python3
"""Charts for results.csv. Runs anywhere with no GPU -- pure matplotlib over a CSV.

Four charts, each written as PNG (200 dpi) and SVG:
  1. memory_vs_batch    -- predicted line items (stacked) + measured total, across the batch sweep
  2. memory_vs_seqlen   -- same treatment across the sequence sweep, plus a dashed predicted-floor line
  3. lever_impact       -- horizontal bars at the demanding config: baseline/8-bit/checkpoint/both,
                           with an 80 GB capacity line and step-time-vs-baseline annotations
  4. allocated_reserved -- grouped bars per config, allocated vs reserved (the nvidia-smi gap)

OOM configs are drawn as hatched bars reaching the capacity ceiling rather than dropped.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Single palette dict -- restyle everything from here. Values from the categorical
# reference palette (blue/orange/aqua/yellow), validated for CVD-safe adjacency.
PALETTE = {
    "line_items": {
        "weights": "#2a78d6",      # blue
        "gradients": "#eb6834",    # orange
        "optimizer": "#1baf7a",    # aqua
        "autocast_weight_cache": "#8a6fd6",  # violet -- only nonzero under amp_bf16
        "activations": "#eda100",  # yellow
    },
    "measured": "#0b0b0b",       # primary ink -- the "actual" line/marker
    "reserved": "#52514e",       # secondary ink -- the nvidia-smi number
    "predicted_floor": "#52514e",
    "capacity_line": "#d03b3b",  # red -- 80GB ceiling
    "oom": "#d03b3b",
    "text_muted": "#898781",
    "gridline": "#e1e0d9",
    "axis": "#c3c2b7",
}

GPU_80GB_CAPACITY_GB = 80.0
LINE_ITEM_ORDER = ["weights", "gradients", "optimizer", "autocast_weight_cache", "activations"]
LINE_ITEM_LABELS = {
    "weights": "Weights", "gradients": "Gradients", "optimizer": "Optimizer",
    "autocast_weight_cache": "Autocast cache", "activations": "Activations",
}


def _style_axes(ax, ylabel="GB"):
    """Clean and neutral: light y-gridlines only, no borders, muted axis text."""
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
    png_path = os.path.join(outdir, f"{name}.png")
    svg_path = os.path.join(outdir, f"{name}.svg")
    # bbox_inches="tight" grows the canvas to include anything placed outside the
    # axes (e.g. the memory-vs-lever legend, parked to the right of the plot so it
    # can never collide with tall bars).
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, svg_path


def _hatch_oom_bar(ax, x, ceiling, width, label=None):
    ax.bar(x, ceiling, width=width, facecolor="none", edgecolor=PALETTE["oom"],
           hatch="////", linewidth=1.2, zorder=3)
    ax.text(x, ceiling * 1.01, "OOM", ha="center", va="bottom", fontsize=9,
            color=PALETTE["oom"], fontweight="bold")


def plot_memory_vs_lever(df, lever, title, xlabel, outdir, name, show_floor_line=False):
    sub = df[df["lever"] == lever].copy()
    sub = sub.sort_values("lever_value", key=lambda s: s.astype(float))
    x_labels = sub["lever_value"].tolist()
    x = range(len(x_labels))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bottom = [0.0] * len(sub)
    for item in LINE_ITEM_ORDER:
        col = f"predicted_{item}_gb"
        values = sub[col].fillna(0).tolist()
        ax.bar(x, values, bottom=bottom, width=0.55, color=PALETTE["line_items"][item],
               label=LINE_ITEM_LABELS[item], zorder=2, edgecolor="white", linewidth=0.5)
        bottom = [b + v for b, v in zip(bottom, values)]

    ceiling = max(max(bottom), sub["max_reserved_gb"].apply(_safe_float).fillna(0).max(), GPU_80GB_CAPACITY_GB * 0.3)
    for xi, (_, row) in zip(x, sub.iterrows()):
        if row["oom"]:
            _hatch_oom_bar(ax, xi, ceiling * 1.15, 0.55)

    measured = [_safe_float(v) for v in sub["max_allocated_gb"]]
    meas_x = [xi for xi, v in zip(x, measured) if v is not None]
    meas_y = [v for v in measured if v is not None]
    ax.plot(meas_x, meas_y, marker="D", markersize=6, linewidth=1.5, color=PALETTE["measured"],
            label="Measured (allocated)", zorder=4)

    if show_floor_line:
        predicted_total = sub["predicted_allocated_gb"].tolist()
        ax.plot(x, predicted_total, linestyle="--", linewidth=1.3, color=PALETTE["predicted_floor"],
                label="Predicted floor (total)", zorder=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    _style_axes(ax)
    # Legend outside the axes: the stacked bars (and OOM hatches) can reach the top
    # of the plot at any x position, so any in-axes placement risks a collision.
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, outdir, name)


def _safe_float(v):
    try:
        f = float(v)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def plot_lever_impact(df, outdir):
    combined = df[df["lever"] == "combined"].copy()
    order = ["baseline", "adam8bit", "checkpointing", "both"]
    combined["_order"] = combined["lever_value"].apply(lambda v: order.index(v) if v in order else 99)
    combined = combined.sort_values("_order")

    # Reference for the step-time annotation: the cheapest config *within this group*
    # that actually completed (all four rows share the same demanding batch/seq, so this
    # isolates the cost of the lever itself rather than the cost of the larger batch/seq).
    # If the true baseline OOM'd, as it often will at the demanding config, fall back to
    # the next-cheapest survivor and say so explicitly.
    ref_row = next((r for _, r in combined.iterrows() if _safe_float(r["step_time_mean_ms"]) is not None), None)
    ref_step_ms = _safe_float(ref_row["step_time_mean_ms"]) if ref_row is not None else None
    ref_label = ref_row["lever_value"] if ref_row is not None else None

    labels = {"baseline": "Baseline", "adam8bit": "8-bit Adam", "checkpointing": "Checkpointing", "both": "Both"}
    colors = [PALETTE["line_items"]["weights"], PALETTE["line_items"]["gradients"],
              PALETTE["line_items"]["optimizer"], PALETTE["line_items"]["activations"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    y = range(len(combined))
    for yi, (_, row), color in zip(y, combined.iterrows(), colors):
        reserved = _safe_float(row["max_reserved_gb"])
        if row["oom"] or reserved is None:
            _hatch_oom_bar_h(ax, yi, GPU_80GB_CAPACITY_GB, 0.6)
        else:
            ax.barh(yi, reserved, height=0.6, color=color, zorder=2)
            step_ms = _safe_float(row["step_time_mean_ms"])
            if step_ms is not None and ref_step_ms:
                if row["lever_value"] == ref_label:
                    text = f"{step_ms:.0f} ms/step (reference)"
                else:
                    pct = (step_ms / ref_step_ms - 1) * 100
                    sign = "+" if pct >= 0 else ""
                    suffix = "" if ref_label == "baseline" else f" (vs {labels.get(ref_label, ref_label)})"
                    text = f"{sign}{pct:.0f}% step time{suffix}"
                ax.text(reserved + 1.0, yi, text, va="center", fontsize=9.5, color=PALETTE["text_muted"])

    ax.axvline(GPU_80GB_CAPACITY_GB, color=PALETTE["capacity_line"], linewidth=1.3, linestyle=":", zorder=1)
    ax.text(GPU_80GB_CAPACITY_GB, len(combined) - 0.3, " H100 80GB", color=PALETTE["capacity_line"],
            fontsize=9.5, va="top")

    ax.set_yticks(list(y))
    ax.set_yticklabels([labels.get(v, v) for v in combined["lever_value"]])
    ax.set_xlabel("Reserved memory (GB)", fontsize=11)
    ax.set_title("Lever impact at the demanding config: memory and its price", fontsize=13, pad=12)
    ax.grid(axis="x", color=PALETTE["gridline"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=10.5, colors=PALETTE["text_muted"])
    ax.tick_params(axis="y", colors="#0b0b0b")
    return _save(fig, outdir, "lever_impact")


def _hatch_oom_bar_h(ax, y, ceiling, height):
    ax.barh(y, ceiling, height=height, facecolor="none", edgecolor=PALETTE["oom"],
            hatch="////", linewidth=1.2, zorder=3)
    ax.text(ceiling * 1.01, y, "OOM", va="center", fontsize=9, color=PALETTE["oom"], fontweight="bold")


def plot_allocated_vs_reserved(df, outdir):
    sub = df.copy()
    sub["label"] = sub["lever"] + ": " + sub["lever_value"].astype(str)
    ceiling = max(_safe_float(v) or 0 for v in sub["max_reserved_gb"]) or GPU_80GB_CAPACITY_GB
    ceiling = max(ceiling, GPU_80GB_CAPACITY_GB * 0.3)

    fig, ax = plt.subplots(figsize=(7.5, max(4.5, 0.34 * len(sub))))
    y = list(range(len(sub)))
    bar_h = 0.36
    for yi, (_, row) in zip(y, sub.iterrows()):
        allocated = _safe_float(row["max_allocated_gb"])
        reserved = _safe_float(row["max_reserved_gb"])
        if row["oom"] or allocated is None:
            _hatch_oom_bar_h(ax, yi, ceiling * 1.1, bar_h * 2 + 0.05)
            continue
        ax.barh(yi + bar_h / 2 + 0.02, allocated, height=bar_h, color=PALETTE["measured"],
                label="Allocated" if yi == 0 else None, zorder=2)
        ax.barh(yi - bar_h / 2 - 0.02, reserved, height=bar_h, color=PALETTE["reserved"],
                label="Reserved" if yi == 0 else None, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"].tolist(), fontsize=9)
    ax.set_xlabel("GB", fontsize=11)
    ax.set_title("Allocated vs. reserved memory, per config (the nvidia-smi gap)", fontsize=13, pad=12)
    ax.grid(axis="x", color=PALETTE["gridline"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=10, colors=PALETTE["text_muted"])
    ax.tick_params(axis="y", colors="#0b0b0b")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.invert_yaxis()
    return _save(fig, outdir, "allocated_vs_reserved")


def main():
    parser = argparse.ArgumentParser(description="Generate memory benchmark charts from results.csv")
    parser.add_argument("--input", default="results.csv")
    parser.add_argument("--outdir", default=".")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)
    df["oom"] = df["oom"].astype(str).str.lower().isin(["true", "1"])

    plot_memory_vs_lever(df, "batch_size", "Memory vs. batch size", "Batch size", args.outdir, "memory_vs_batch")
    plot_memory_vs_lever(df, "seq_len", "Memory vs. sequence length", "Sequence length", args.outdir,
                          "memory_vs_seqlen", show_floor_line=True)
    plot_lever_impact(df, args.outdir)
    plot_allocated_vs_reserved(df, args.outdir)
    print(f"Charts written to {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
