"""Diagnostic: TinyLlama+LoRA memory composition on real hardware.

Sibling of debug_ckpt.py. Run from benchmark/ on a CUDA box with the fine-tune
dependencies installed:  python debug_finetune.py > debug_finetune_out.json

Three targeted probes:
  1. Per-layer working-set decomposition (SwiGLU/GQA breakdown) at the baseline config.
     Uses torch.cuda.memory._record_memory_history / _snapshot, buckets live blocks at
     peak by exact byte size, matches those bucket sizes back to the tensors the formula
     in predictions_finetune.py names. Confirms whether the per-layer term structure
     matches the arch this benchmark actually ran on.
  2. Autocast-cache residual under bf16 base. Runs the baseline forward with
     cache_enabled=True and again with cache_enabled=False; the delta is the empirical
     cache size. Predicted ~0.013 GB (adapter-only) vs Article 1's ~2.0 GB baseline cache.
     This is the diagnostic the "open empirical question" from Article 2's plan is on.
  3. Adapter-placement backward reach. Builds an upper-3 placement variant, runs one
     forward+backward, asserts p.grad is None for parameters in the lower 19 layers,
     and records the forward-peak delta vs the 'all' baseline -- expected ~3/22 ratio
     on the activation term.
"""
import json
import sys

import torch

sys.path.insert(0, ".")
from sweep_config_finetune import make_baseline_finetune  # noqa: E402
from model_finetune import build_finetune_model  # noqa: E402

GB = 1e9


def summarize_peak(snapshot):
    """Replay alloc/free events, return live-block composition at the peak moment.

    Copied from debug_ckpt.py's summarize_peak -- kept in this file rather than imported
    so this diagnostic is one self-contained script (the import from debug_ckpt would be
    awkward and creates cross-file coupling for a small function).
    """
    trace = snapshot["device_traces"][0]
    live = {}
    cur = 0
    peak = 0
    peak_live = None
    for ev in trace:
        action = ev["action"]
        if action == "alloc":
            live[ev["addr"]] = (ev["size"], ev.get("frames", []))
            cur += ev["size"]
            if cur > peak:
                peak = cur
                peak_live = dict(live)
        elif action == "free_completed":
            blk = live.pop(ev["addr"], None)
            if blk is not None:
                cur -= blk[0]
    if peak_live is None:
        return {"replay_peak_gb": 0, "buckets": []}

    buckets = {}
    for size, frames in peak_live.values():
        b = buckets.setdefault(size, {"count": 0, "frames": None})
        b["count"] += 1
        if b["frames"] is None and frames:
            keep = []
            for f in frames:
                name = f.get("name", "?")
                fname = f.get("filename", "")
                short = fname.split("/site-packages/")[-1].split("/")[-1]
                keep.append(f"{short}:{f.get('line', '?')}:{name}")
            b["frames"] = keep[:6]
    rows = sorted(
        ({"size_mb": s / 1e6, "count": v["count"], "total_gb": s * v["count"] / GB,
          "stack": v["frames"]} for s, v in buckets.items()),
        key=lambda r: -r["total_gb"],
    )
    big = [r for r in rows if r["total_gb"] > 0.02]
    rest = sum(r["total_gb"] for r in rows if r["total_gb"] <= 0.02)
    return {"replay_peak_gb": peak / GB, "buckets": big, "small_blocks_gb": rest}


def _one_step(model, vocab, batch, seq, cache_enabled, record_history):
    """One forward+backward+step. Returns phase peaks and (if record_history) the peak
    composition from a replay of the allocator's event stream."""
    import torch.nn.functional as F  # noqa: F401  (kept for future use if we swap loss)

    dev = torch.device("cuda")
    torch.cuda.synchronize()
    if record_history:
        torch.cuda.memory._record_memory_history(max_entries=400000)
    torch.cuda.reset_peak_memory_stats()

    out = {
        "alloc_steady_gb": torch.cuda.memory_allocated() / GB,
        "cache_enabled": cache_enabled,
    }

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

    input_ids = torch.randint(0, vocab, (batch, seq), device=dev)
    labels = input_ids.clone()

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                        cache_enabled=cache_enabled):
        result = model(input_ids=input_ids, labels=labels)
        loss = result.loss

    torch.cuda.synchronize()
    out["fwd_peak_gb"] = torch.cuda.max_memory_allocated() / GB
    out["alloc_after_fwd_gb"] = torch.cuda.memory_allocated() / GB
    torch.cuda.reset_peak_memory_stats()

    loss.backward()
    torch.cuda.synchronize()
    out["bwd_peak_gb"] = torch.cuda.max_memory_allocated() / GB
    out["alloc_after_bwd_gb"] = torch.cuda.memory_allocated() / GB

    optimizer.step()

    if record_history:
        torch.cuda.synchronize()
        snap = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)
        out["peak_composition"] = summarize_peak(snap)

    return out


def _fresh_model(**overrides):
    """Return a fresh (base + PEFT) model at the baseline config plus any overrides."""
    from dataclasses import replace

    config = replace(make_baseline_finetune(), **overrides)
    model, base_arch = build_finetune_model(config)
    model.to("cuda").train()
    return model, base_arch


def probe_1_per_layer_working_set():
    """Baseline config with allocator history on -- pulls out the per-layer term structure
    empirically. Expected big-bucket sizes (baseline: batch=8, seq=512, d=2048,
    intermediate=5632, bf16):
      B*S*d*2       = 8*512*2048*2 = 16,777,216 (~16 MB)   -- attention-block width tensors
      B*S*d*(kvh/nh)*2 = 8*512*(2048/8)*2 = 2,097,152 (~2 MB) -- GQA-narrowed K/V
      B*S*i*2       = 8*512*5632*2 = 46,137,344 (~46 MB)   -- SwiGLU intermediate-width tensors
      B*S*d*4       = 8*512*2048*4 = 33,554,432 (~32 MB)   -- fp32 residual/norm boundary
    """
    torch.manual_seed(0)
    model, arch = _fresh_model()
    result = _one_step(model, arch["vocab_size"], batch=8, seq=512,
                        cache_enabled=True, record_history=True)
    result["probe"] = "per_layer_working_set"
    result["arch"] = arch
    del model
    torch.cuda.empty_cache()
    return result


def probe_2_autocast_cache_residual():
    """Baseline with cache_enabled=True and again with cache_enabled=False. Reported
    fwd_peak_gb delta is the empirical autocast cache size. Under bf16 base storage the
    predicted delta is n_adapter * 2 bytes (~0.013 GB); Article 1's baseline saw ~2 GB
    when the base was fp32."""
    torch.manual_seed(0)
    model, arch = _fresh_model()
    result_on = _one_step(model, arch["vocab_size"], batch=8, seq=512,
                          cache_enabled=True, record_history=False)
    del model
    torch.cuda.empty_cache()

    torch.manual_seed(0)
    model, arch = _fresh_model()
    result_off = _one_step(model, arch["vocab_size"], batch=8, seq=512,
                            cache_enabled=False, record_history=False)
    del model
    torch.cuda.empty_cache()

    return {
        "probe": "autocast_cache_residual",
        "with_cache": result_on,
        "without_cache": result_off,
        "empirical_cache_gb": result_on["fwd_peak_gb"] - result_off["fwd_peak_gb"],
    }


def probe_3_adapter_placement_backward_reach():
    """Upper-3 placement. Asserts frozen lower-layer params have p.grad is None; records
    the forward-peak delta relative to the 'all' baseline. Expected activation-term ratio
    ~3/22 (0.136)."""
    torch.manual_seed(0)
    model, arch = _fresh_model(lora_adapter_layers="upper-3")

    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-4)
    input_ids = torch.randint(0, arch["vocab_size"], (8, 512), device=dev)
    labels = input_ids.clone()

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model(input_ids=input_ids, labels=labels)
        loss = result.loss
    torch.cuda.synchronize()
    fwd_peak_upper3 = torch.cuda.max_memory_allocated() / GB
    loss.backward()

    # Assert lower layers (indices 0..18 for TinyLlama's 22 layers) have no grad on any
    # base weight. Layer-index detection: PEFT wraps the base as
    # model.base_model.model.model.layers[i] on a Llama model.
    n_layers = arch["num_hidden_layers"]
    n_upper = 3
    lower_layers_have_grad = []
    for name, p in model.named_parameters():
        if p.requires_grad or p.grad is None:
            continue
        # We only care about base-weight params in layers below the shallowest adapter.
        for i in range(n_layers - n_upper):
            if f".layers.{i}." in name:
                lower_layers_have_grad.append(name)
                break

    del model
    torch.cuda.empty_cache()

    return {
        "probe": "adapter_placement_backward_reach",
        "upper_n": n_upper,
        "total_layers": n_layers,
        "fwd_peak_upper3_gb": fwd_peak_upper3,
        "lower_layers_that_unexpectedly_got_grad": lower_layers_have_grad,
        "note": "This list should be empty. If it isn't, PEFT's layers_to_transform did "
                "not stop backward reach where the formula assumes it does; the "
                "adapter_placement_sweep predictions need re-derivation.",
    }


def main():
    results = []
    for probe_fn in (probe_1_per_layer_working_set,
                     probe_2_autocast_cache_residual,
                     probe_3_adapter_placement_backward_reach):
        try:
            results.append(probe_fn())
        except Exception as e:  # keep other probes running if one fails
            results.append({"probe": probe_fn.__name__, "error": repr(e)})
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
