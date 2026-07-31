# Corrected training-memory formulas — reference for article-1-training.md and the companion spreadsheet

Source: the `benchmark/` GPU sweep in this repo, cross-checked against real 80GB-GPU
runs (original 17-config sweep: A100 80GB; current 24-config reference sweep in
`runs/20260731-1200/`: H100 80GB). Full derivation, code, and tests are in `benchmark/predictions.py`. This document
exists to (a) tell you exactly what to change in the article and why, and (b) give you
formulas in plain arithmetic — not Python — so they translate directly into spreadsheet
cells.

**Headline finding:** the article's static-memory total (16 GB) turns out to still be
approximately right, but for the wrong reason, and with one new term missing. The
activation formula was wrong by roughly 20x, which is the dominant error — at the
article's own batch=1,024 example, real memory demand is ~187 GB, not ~26 GB.

---

## 1. What was actually wrong

The article's model implicitly assumed a training implementation that doesn't quite
match what the reference code does:

- **Weights/gradients/optimizer state were assumed to be stored in bf16 with an fp32
  optimizer master-copy** (the standard mixed-precision recipe). The actual code just
  moves the model to the GPU and wraps the forward pass in `torch.autocast` — it never
  casts parameter storage. Params, gradients, and optimizer state are **fp32 the whole
  time**, in both "bf16" and fp32 training. Autocast only changes the dtype used *during*
  the matmuls.
- **The activation formula used a "one tensor per layer" floor** (`batch × seq × hidden
  × 2 bytes × layers`). That's actually the formula for what *gradient checkpointing*
  saves — applied to the non-checkpointed case, it undercounts real activations by
  roughly 20x, because a real transformer layer without checkpointing saves ~15
  distinct intermediate tensors, not one.
- **The checkpointed case then divided that already-wrong floor by layer count again**,
  undercounting a further ~40x (predicted 0.1 GB vs. measured ~4 GB at the article's own
  scale).

The net effect on the article's headline number, worked below: not a rounding error.

---

## 2. The corrected model — five line items, not four

Two coincidences make this less disruptive than it sounds: the *fp32-storage* static
total (weights + gradients + optimizer, computed the "wrong" way in the article) happens
to land on the same 16 bytes/param as the *correct* fp32 accounting, just via different
underlying math. So **the article's "16 GB static subtotal" headline number can stay**
— only the explanation changes, and a new term (below) needs to be added alongside it.

### Line item 1: Weights

> Weights = N × 4 bytes

Always 4 bytes/param (fp32) in this training recipe, regardless of "bf16" vs "fp32"
mode — autocast never touches parameter storage.

*1B params → 4 GB* (article said 2 GB, assuming true bf16 storage)

### Line item 2: Gradients

> Gradients = N × 4 bytes

Same reasoning — gradients match parameter dtype (fp32).

*1B params → 4 GB* (article said 2 GB)

### Line item 3: Optimizer state

> Optimizer (standard Adam) = N × 8 bytes   (momentum + variance, both fp32)
> Optimizer (8-bit Adam) = N × 2 bytes   (momentum + variance, both int8)

The article's 12 bytes/param assumed a bf16-master-weight mixed-precision Adam layout.
The real code runs plain fp32 Adam, which is 8 bytes/param, not 12. The 8-bit figure
(2 bytes/param, not 3) is validated against a real measured delta: switching to 8-bit
Adam saved 6.04 GB measured, against 6.07 GB predicted by this formula (0.5% error) —
the article's old 3-bytes/param assumption would have predicted a 9.1 GB saving, 51%
too high.

*1B params, standard Adam → 8 GB* (article said 12 GB)
*1B params, 8-bit Adam → 2 GB* (article said 3 GB)

**New static subtotal: 4 + 4 + 8 = 16 GB** — coincidentally the same headline number as
the article, for a different reason (fp32-everything nets the same bytes/param as the
article's bf16-plus-fp32-master-copy assumption).

### Line item 4 (new): Autocast weight cache

> Autocast cache = N × 2 bytes   (bf16 mode only; 0 in fp32 mode)

Under bf16 autocast, PyTorch casts each fp32 weight to bf16 on the fly for the matmul
and caches that bf16 copy for reuse across the layer's ops during the forward pass. This
is a real, separate allocation — the old formula silently absorbed it into "activations"
(part of where the old 2-4x fudge multiplier was hiding).

*1B params, bf16 mode → 2 GB. Zero in pure fp32 training.*

**Static total including this term: 18 GB in bf16 mode, 16 GB in pure fp32 mode.**

### Line item 5: Activations — rebuilt from scratch

This is the term that actually matters and where the old formula failed. Replace the
single floor formula with explicit per-tensor accounting. Notation: B=batch, S=seq_len,
d=hidden dim, L=layers, f=FFN multiplier (4 is standard), V=output vocab/head size,
b=activation bytes (2 for bf16 compute, 4 for fp32).

**Per transformer layer, non-checkpointed** (everything below is saved for backward):

| Term | Formula | What it is |
|---|---|---|
| Attention block | 7 × B·S·d·b | 7 same-shaped tensors: pre-norm input, QKV-projection input, Q, K, V, attention output, output-projection input |
| Attention probability matrix | **0** (see caveat) | Zeroed because flash/memory-efficient attention kernels never materialize the full seq×seq matrix. If your implementation uses naive/"math" attention instead, add back `B·heads·S²·b` — at large S this term dominates everything else. |
| MLP pre-norm | 2 × B·S·d·b | FFN's own pre-norm input + the input to the first FFN linear layer |
| MLP hidden | 2 × B·S·(f·d)·b | The FFN's expanded-width intermediate tensors (pre- and post-activation-function) |
| Dropout masks | 2·(B·S·d) + B·S·(f·d), × 1 byte | 3 dropout calls per layer (standard transformer default of 0.1 dropout); 1 byte/element masks |

> **Per-layer subtotal = 9·(B·S·d·b) + 2·(B·S·f·d·b) + [2·(B·S·d) + B·S·f·d]**

Multiply by L for the full non-checkpointed model, then add the once-per-model terms
below.

**Once per model** (not per layer):

| Term | Formula |
|---|---|
| Embedding output | B·S·d·b |
| Output head (logits) | B·S·V·b |
| fp32 loss-computation upcast (bf16 mode only) | B·S·V·4 |

> Cross-entropy is numerically unstable in bf16, so autocast forces it to fp32
> internally — that upcasts a second, full-size copy of the output layer's activations.
> **This term doesn't apply if your output head isn't a softmax classifier** (e.g. a
> tabular regression head) — swap in whatever your real output layer produces. It's
> usually small relative to the L-layer accumulated total either way.

**Checkpointed** (rewritten 2026-07-31 after a CUDA allocator-history replay on a real
H100 identified every live block at the peak moment — see §5 for the upgraded
confidence):

> Checkpointed activations (backward peak) = (L+3)·(B·S·d·**4**)  +  7·(B·S·d·b)  +  3·(B·S·f·d·b)  +  B·S·d  +  B·S·V·b

Two facts the earlier version of this formula got wrong, both mechanisms now named:

- **The saved layer boundaries are fp32, not bf16.** Under autocast, LayerNorm and the
  residual adds run in fp32, so the tensor passed *between* layers — exactly what
  checkpointing saves — is 4 bytes/element even in bf16 mode. The earlier formula used
  the 2-byte compute dtype, halving the dominant term. The +3 beyond L: the segment
  being recomputed re-saves its norm tensors, and the incoming gradient w.r.t. the
  segment output is the same shape.
- **Peak timing matters once activations shrink.** Whole-run peak allocated is the max
  of three phase peaks, not a sum:
  1. *Backward-recompute peak* = weights + optimizer state + the activation expression
     above. No weight gradients yet (they haven't accumulated at this moment), no
     autocast weight cache (freed when the forward pass exited autocast).
  2. *Optimizer-step peak* = weights + gradients + optimizer state + one extra
     parameter-sized set of temporaries that torch's default (foreach) Adam
     materializes = 4+4+8+4 = **20 bytes/param** for standard Adam (8-bit Adam updates
     in place — no temporaries). This floor dominates at small B·S: it's why the
     measured checkpointed baseline is ~20.35 GB, not the ~19 GB the backward peak
     alone implies.
  3. *Forward peak* = weights + optimizer state + autocast cache + L fp32 boundaries +
     one segment's forward transient + logits terms. Rarely dominant, matters for
     8-bit-Adam-with-checkpointing configs where the optimizer floor is low.

Validated against the 2026-07-31 24-config H100 sweep: the four checkpointing rows
that previously missed at 10–14% error all land within **0.6%**, and the one wrong OOM
call (checkpointing at seq=1024: predicted fits, actually OOMed) is now called
correctly — with zero tuned constants.

---

## 3. Worked recompute of the article's own examples

Using the article's exact config (1B params, d=2048, 20 layers, seq=100, bf16, standard
Adam) at its two headline batch sizes:

| Scenario | Article's claim | Corrected prediction | Fits? |
|---|---|---|---|
| batch=1,024, no tricks | ~26 GB ("tight fit on 24GB, comfortable on A100 40GB") | **~187 GB** | No — not even an 80GB A100 |
| batch=1,024, checkpointing only | *(not stated)* | ~36 GB | Fits A100 40GB |
| batch=1,024, 8-bit Adam only | *(not stated)* | ~181 GB | No — 8-bit Adam alone barely moves the needle, since activations dominate |
| batch=1,024, both levers | *(not stated)* | ~30 GB | Fits A100 40GB, not quite a 24GB card |
| batch=4,096, no tricks | *(implied, "either move to 80GB or pull two levers")* | **~693 GB** | No, by a huge margin |
| batch=4,096, both levers | ~24 GB ("same run fits in roughly 24GB") | **~83 GB** | **No — doesn't fit even an 80GB A100** |

The batch=4,096-with-both-levers claim isn't just off in magnitude — it flips the
verdict. The article says two levers get you comfortably under 24GB; the corrected
number says the same config still doesn't fit the largest single GPU commonly
available. This is the number most worth fixing before publishing, since it's the one
readers will act on directly.

Real GPU data backs the qualitative shape of this even though these exact batch/seq
combinations haven't been measured: at the sweep's baseline (batch=256, not 1,024),
checkpointing alone cut measured memory from 59.1 GB to 20.3 GB (2.9x), and step time
rose 1027ms → 1363ms (+33% — close to the article's own "~30% extra compute" claim,
which holds up). The batch=1,024/4,096 numbers above are formula extrapolations from
that same validated model, not independent GPU measurements — see the confidence table
below before treating them as final.

---

## 4. Overhead terms (for GPU-fit / OOM calls, not for "how much memory does training use")

Two more terms exist only to translate "allocated" into "will this OOM on a real card,"
and are explicitly **not** derived from first principles — `torch`'s own memory API
can't see CUDA context overhead at all, so there's no way to measure it from inside a
training process:

> Reserved total = Allocated total + CUDA context overhead (~0.6 GB) + fragmentation allowance (~1.2 GB, ~5.2 GB if checkpointing)

The fragmentation figures come from the gap between `reserved` and `allocated` in the
2026-07-31 sweep: ~1.2 GB for non-checkpointed rows, and 3.58–6.12 GB across the three
non-OOM checkpointed configs (checkpointing fragments the allocator's pool more than
steady-state training, but the gap doesn't scale cleanly with batch/seq — it's largest
at the smallest config). The 5.2 GB constant sits inside that range. If your spreadsheet
needs a "will this fit" checkbox, compare **reserved**, not allocated, against GPU
capacity — that's what the OOM call in `validate_results.py` does.

---

## 5. Confidence per term — what to trust, what to flag as provisional

| Term | Confidence | Basis |
|---|---|---|
| Weights, gradients (fp32-always) | **High** | Directly measured on CPU tensors, exact by construction |
| Optimizer state (8 or 2 bytes/param) | **High** | Validated against a real measured delta (0.5% error) |
| Autocast weight cache | **High** | Isolated 2026-07-31 via an `autocast(cache_enabled=False)` toggle on the H100 (`runs/20260731-1200/debug_ckpt_out.json`): peak forward memory drops 1.98 GB with the cache off, vs. 2.02 GB predicted |
| Activation formula, non-checkpointed | **High at small-to-moderate scale** | Baseline config validated to 2.1% against real GPU data |
| Attention-probability-matrix = 0 | **High** | Confirmed via source (this code's attention call forces the flash/SDPA path) *and* empirically ruled out the alternative using this sweep's own OOM data (a materialized matrix would demand ~550 GB, not the observed ~68 GB) |
| Dropout mask term | **Speculative** | Assumes masks are materialized as byte tensors; some fused CUDA dropout kernels instead save only RNG state, which would make this term ~0. Not yet isolated. It's a small fraction of the total either way. |
| Activation formula, checkpointed | **High** | Rebuilt 2026-07-31 from a CUDA allocator-history replay (every live block at peak identified by exact byte size); validates within 0.6% on all four non-OOM checkpointed H100 rows and fixes the one wrong OOM call |
| batch/seq combinations beyond what's been measured (incl. all batch=1,024/4,096 numbers in §3) | **Extrapolated, unverified** | Formula only, no GPU measurement at these exact points yet |
| CUDA context + fragmentation constants | **Order-of-magnitude, not derived** | Structurally invisible to `torch`'s memory API; the checkpointing fragmentation constant now spans three measured gaps (3.58–6.12 GB) that don't scale cleanly, so it stays a single documented empirical term |

**On the one remaining provisional term:** the earlier ~11.6% checkpointing miss is gone
— the checkpointed *activation* formula was rebuilt from allocator-history evidence and
now validates within 0.6% (see the confidence table above). What stays provisional is
the checkpointing *fragmentation* constant: the reserved-minus-allocated gap spans
3.58–6.12 GB across the three measured checkpointed configs and doesn't scale cleanly
with batch or sequence, so it's kept as a single documented empirical term rather than
curve-fit to any one point. A wider checkpointing sweep would either pin it down or show
it needs a batch/seq-dependent form.

---

## 6. Building the spreadsheet

Inputs (one row per scenario, or one set of named cells for a single scenario):

- `N` (param count), `d` (hidden dim), `L` (layers), `heads`, `f` (FFN multiplier, 4 is
  standard), `V` (output head width — vocab size, or your real output dimension)
- `B` (batch), `S` (sequence length)
- `precision` (bf16 / fp32) → sets `b` = 2 or 4, and gates the autocast-cache and
  fp32-upcast terms
- `optimizer` (adam / adam8bit) → sets optimizer bytes/param = 8 or 2
- `checkpointing` (on/off)

Output cells, in order (each is a one-line spreadsheet formula from §2/§4 above):

1. `weights_gb` = N × 4 / 1e9
2. `gradients_gb` = N × 4 / 1e9
3. `optimizer_gb` = N × (IF optimizer="adam8bit", 2, 8) / 1e9
4. `autocast_cache_gb` = IF(precision="bf16", N × 2, 0) / 1e9
5. `per_layer_bytes` = 9×B×S×d×b + 2×B×S×f×d×b + 2×B×S×d + B×S×f×d   *(b = IF(precision="bf16",2,4))*
6. `once_per_model_bytes` = B×S×d×b + B×S×V×b + IF(precision="bf16", B×S×V×4, 0)
7. `activations_gb` = IF(checkpointing,
   `(L+3)×B×S×d×4 + 7×B×S×d×b + 3×B×S×f×d×b + B×S×d + B×S×V×b`,
   `L×per_layer_bytes + once_per_model_bytes`) / 1e9
8. `allocated_total_gb` = IF(checkpointing,
   `MAX(weights_gb + optimizer_gb + activations_gb,`
   `    weights_gb + gradients_gb + optimizer_gb + IF(optimizer="adam", gradients_gb, 0),`
   `    weights_gb + optimizer_gb + autocast_cache_gb + L×B×S×d×4/1e9 + (2×B×S×f×d + 3×B×S×d)×b/1e9 + (B×S×V×b + IF(precision="bf16", B×S×V×4, 0))/1e9)`,
   `weights_gb + gradients_gb + optimizer_gb + autocast_cache_gb + activations_gb`)
   *(checkpointed: max of backward-recompute, optimizer-step, and forward phase peaks — see §2)*
9. `reserved_total_gb` = allocated_total_gb + 0.6 + IF(checkpointing, 5.2, 1.2)
10. `fits_gpu` = reserved_total_gb ≤ (your GPU's usable capacity — 79.18 GiB × 1.0737 ≈ 85.0 for an 80GB H100/A100)

That's a complete, self-contained spreadsheet — no external lookups, no fudge
multiplier, every cell traceable to a named row in §2-4.

---

## 7. Punch list for article-1-draft.md

- **Line item 1 (weights):** change "2 GB" → "4 GB"; drop the "2 bytes per parameter"
  framing, replace with "4 bytes (fp32) — real training recipes rarely store weights in
  true bf16; mixed precision means bf16 *compute*, not bf16 *storage*."
- **Line item 2 (gradients):** 2 GB → 4 GB, same reasoning.
- **Line item 3 (optimizer):** 12 GB → 8 GB. Rewrite "two history statistics plus a
  full-precision master copy" → "two fp32 history statistics; no separate master copy,
  because the weights were never downcast in the first place."
- **New line item (autocast cache):** add a short new section, 2 GB at 1B params, bf16
  only.
- **Static subtotal:** stays "16 GB" as the headline (now weights+gradients+optimizer
  only), but call out the new 18 GB figure once the autocast cache is included in bf16
  mode.
- **Line item 4 (activations):** replace the floor formula and the "plan for 2-4x this
  estimate" fudge-multiplier guidance entirely. The new explicit accounting doesn't need
  a fudge factor — it's within a few percent of measured at validated scales. Replace
  the per-row "~8 MB" worked number with the real corrected figure (~168 MB/row at this
  config, not 8 MB — a 20x difference).
  Also fix the batch example arithmetic itself, not just the multiplier.
- **Checkpointing claim ("cuts this line ~10x for ~30% extra compute"):** keep — it
  holds up under the corrected formula (9.5x cut in the activation line, +33% measured
  step time). This is one of the few numbers in the article that didn't need fixing.
- **"What this means for hardware" section:** replace both worked numbers per the table
  in §3. The batch=4,096-with-both-levers claim is the most important fix — flag to
  readers that it changes from "comfortably fits" to "still doesn't fit an 80GB card."
- **Closing caveat ("your mileage will vary ±20%"):** can be tightened for the
  validated regime (baseline-shaped configs land within ~2-3%) but should stay wider
  (or explicitly flag "extrapolated, unverified") for checkpointed and large-batch/seq
  configs per the confidence table in §5.
