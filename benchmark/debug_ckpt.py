"""Diagnostic: where does peak memory actually come from in checkpointed configs?

Run from benchmark/ on a CUDA box:  python debug_ckpt.py > /tmp/debug_out.json

For each probe config it reports:
  - phase peaks: forward-phase peak, backward-phase peak (separately reset)
  - allocated at phase boundaries
  - with history=True: a replay of the allocator event stream to find the exact
    peak moment and the composition of live blocks at that moment (size buckets
    + a sample python stack per bucket, so each block maps to a named tensor).
"""
import json
import sys

import torch
import torch.nn as nn

sys.path.insert(0, ".")
from model import TinyTransformer

GB = 1e9


def summarize_peak(snapshot):
    """Replay alloc/free events; return live-block composition at the peak moment."""
    trace = snapshot["device_traces"][0]
    live = {}          # addr -> (size, frames)
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
                peak_live = dict(live)  # shallow copy of the live set at this moment
        elif action == "free_completed":
            blk = live.pop(ev["addr"], None)
            if blk is not None:
                cur -= blk[0]
    if peak_live is None:
        return {"replay_peak_gb": 0, "buckets": []}

    # bucket live blocks by exact size
    buckets = {}
    for size, frames in peak_live.values():
        b = buckets.setdefault(size, {"count": 0, "frames": None})
        b["count"] += 1
        if b["frames"] is None and frames:
            # keep a short, informative stack sample: last few python frames
            keep = []
            for f in frames:
                name = f.get("name", "?")
                fname = f.get("filename", "")
                short = fname.split("/site-packages/")[-1].split("/")[-1]
                keep.append(f"{short}:{f.get('line','?')}:{name}")
            b["frames"] = keep[:6]
    rows = sorted(
        ({"size_mb": s / 1e6, "count": v["count"], "total_gb": s * v["count"] / GB,
          "stack": v["frames"]} for s, v in buckets.items()),
        key=lambda r: -r["total_gb"],
    )
    # only report buckets that matter (>50 MB total), rest as a remainder line
    big = [r for r in rows if r["total_gb"] > 0.05]
    rest = sum(r["total_gb"] for r in rows if r["total_gb"] <= 0.05)
    return {"replay_peak_gb": peak / GB, "buckets": big, "small_blocks_gb": rest}


def probe(batch, seq, ckpt, cache_enabled, history, d=2048, layers=20, heads=16,
          ff_mult=4, vocab=1000):
    torch.manual_seed(0)
    dev = torch.device("cuda")
    model = TinyTransformer(vocab, d, layers, heads, ff_mult, seq,
                            use_checkpointing=ckpt).to(dev)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    out = {"batch": batch, "seq": seq, "ckpt": ckpt, "cache_enabled": cache_enabled}

    for step in range(3):
        last = step == 2
        tokens = torch.randint(0, vocab, (batch, seq), device=dev)
        targets = torch.randint(0, vocab, (batch, seq), device=dev)
        if last:
            torch.cuda.synchronize()
            if history:
                torch.cuda.memory._record_memory_history(max_entries=400000)
            torch.cuda.reset_peak_memory_stats()
            out["alloc_steady_gb"] = torch.cuda.memory_allocated() / GB

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            cache_enabled=cache_enabled):
            logits = model(tokens)
            loss = loss_fn(logits.view(-1, vocab), targets.view(-1))

        if last:
            torch.cuda.synchronize()
            out["fwd_peak_gb"] = torch.cuda.max_memory_allocated() / GB
            out["alloc_after_fwd_gb"] = torch.cuda.memory_allocated() / GB
            torch.cuda.reset_peak_memory_stats()

        loss.backward()

        if last:
            torch.cuda.synchronize()
            out["bwd_peak_gb"] = torch.cuda.max_memory_allocated() / GB
            out["alloc_after_bwd_gb"] = torch.cuda.memory_allocated() / GB

        opt.step()

        if last and history:
            torch.cuda.synchronize()
            snap = torch.cuda.memory._snapshot()
            torch.cuda.memory._record_memory_history(enabled=None)
            out["peak_composition"] = summarize_peak(snap)

    del model, opt
    torch.cuda.empty_cache()
    return out


def main():
    results = []
    # 1-3: the three non-OOM checkpointing configs that missed, with full history
    results.append(probe(256, 100, ckpt=True, cache_enabled=True, history=True))
    results.append(probe(1024, 100, ckpt=True, cache_enabled=True, history=True))
    results.append(probe(256, 512, ckpt=True, cache_enabled=True, history=True))
    # 4: autocast-cache toggle on checkpointed baseline (hypothesis 1 test)
    results.append(probe(256, 100, ckpt=True, cache_enabled=False, history=False))
    # 5-6: non-checkpointed control pair for the cache toggle
    results.append(probe(256, 100, ckpt=False, cache_enabled=True, history=False))
    results.append(probe(256, 100, ckpt=False, cache_enabled=False, history=False))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
