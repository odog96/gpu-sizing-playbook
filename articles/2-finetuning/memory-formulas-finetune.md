# Fine-tuning memory formulas — reference for article-2-finetuning.md and the companion spreadsheet

Source: the `benchmark/` GPU sweep in this repo, validated against a 25-config LoRA
sweep of TinyLlama-1.1B on an H100 80GB (`benchmark/reference-run-finetune/`). Full
derivation, code, and unit tests live in `benchmark/predictions_finetune.py`. This
document exists to (a) tell the article what to say and where each number came from,
and (b) render the formulas in plain arithmetic — not Python — so they translate
directly into spreadsheet cells.

Unlike Article 1's `memory-formulas.md`, this doc has no "what was wrong" history
section. The fine-tuning formulas were built from a validated H100 sweep from the
start; no correction pass was needed. All 23 non-OOM configurations land within
±8.3% of measured, every OOM call is correct.

---

## 1. The six line items

Article 2 splits Article 1's four line items into six, because "n_params" is no
longer a single scalar. Weights become two line items (frozen base + trainable
adapters); everything on the trainable side (gradients, optimizer) is scoped to
the adapter count.

Notation: `N_base` = frozen base parameter count, `N_adapt` = trainable adapter
parameter count, `B` = batch, `S` = sequence length, `d` = hidden dim, `L` = total
layers, `L_bwd` = layers needing backward reach (see §2), `nh` = attention heads,
`nkv` = KV heads (for GQA), `ff` = FFN intermediate width, `V` = vocab size,
`b` = compute-dtype activation bytes (2 for `amp_bf16`, 4 for `fp32`),
`p_drop` = attention/hidden dropout rate.

### Line item 1: Frozen base weights

> Frozen weights = N_base × bytes/param

Bytes/param comes from `base_storage_precision`: `fp32`=4, `bf16`=2, `int8`=1,
`int4`=0.5 (two params packed into one byte). This is the fine-tune-specific
freedom: the base never trains, so it cannot lose small updates to storage
precision, and quantizing the base is a routine choice — not the exotic one it
would be in Article 1.

*1.10B params, bf16 → 2.20 GB. Same base at int4 → 0.55 GB.*

### Line item 2: Adapter weights

> Adapter weights = N_adapt × 4 bytes

LoRA adapters stay fp32 so their small updates aren't rounded away by lower
precision. For a projection of shape `(input × output)`, LoRA replaces it with
`W + B·A` where `A` is `rank × input` and `B` is `output × rank` — that's
`rank × (input + output)` new trainable parameters per target.

For Llama-family models the target set typically covers Q, K, V, O, gate, up,
and down (seven targets). Q and O are square (`hidden × hidden`), but under
grouped-query attention K and V are narrow (`hidden × (hidden × nkv/nh)`),
and the SwiGLU MLP has three rectangular projections between `hidden` and the
wider `intermediate_size`. A back-of-envelope `2 × rank × d × targets × L_adapt`
understates the true count on any GQA + SwiGLU model — for TinyLlama at rank 8
across all 22 layers it gives ~5.0M parameters, versus the measured 6.31M.

*rank 8, all 7 targets, all 22 layers on TinyLlama:*
*per-layer trainable count = rank × ((d+d) + (d+d·nkv/nh) + (d+d·nkv/nh) + (d+d)*
*+ (d+ff) + (d+ff) + (ff+d)) = 8 × 35,840 = 286,720; × 22 layers = 6,307,840.*
*Total: 6.31M × 4 bytes = 25.23 MB, matching the measured tensor bytes exactly.*

### Line item 3: Adapter gradients

> Adapter gradients = N_adapt × 4 bytes

One gradient per trainable parameter, matched to adapter storage precision.
Frozen params have no gradient at all — backprop still flows through them
(the chain rule requires it), but the framework allocates nothing for what it
will never apply.

*Same 25.2 MB as adapter weights.*

### Line item 4: Optimizer state (adapters only)

> Adam = N_adapt × 8 bytes
> Adam8bit = N_adapt × 2 bytes

Same arithmetic as Article 1's optimizer line, applied to <1% of the parameter
count. Adam holds fp32 momentum + fp32 variance (8 bytes); bitsandbytes'
8-bit Adam packs both into int8 with block-wise dynamic quantization (2 bytes).
The lever exists; it doesn't matter at this scale.

*rank 8, all 22 layers, Adam → 50.5 MB. Same config, Adam8bit → 12.6 MB.*

### Line item 5: Autocast weight cache

> Autocast cache
>   = (N_base + N_adapt) × 2 bytes   if bf16 mode AND base is fp32
>   = N_adapt × 2 bytes              if bf16 mode AND base is bf16/int8/int4
>   = 0                              if fp32 mode

Under bf16 autocast, PyTorch caches a bf16 copy of every fp32 weight it casts on
the fly for reuse across the forward pass. When the base is loaded at bf16
storage directly there is nothing to cast for the base, and the term collapses to
the adapter-only residual — megabytes, not gigabytes. Only if the base is loaded
at fp32 (unusual for fine-tuning) does the full Article 1 cache reappear.

*rank 8, bf16 base → 12.6 MB. Same config, fp32 base → 2.21 GB.*

### Line item 6: Activations

Activations follow Article 1's per-tensor accounting with three
architecture-specific adjustments and one placement lever unique to fine-tuning.

**Per Llama-family decoder layer (non-checkpointed):**

| Term | Formula | What it is |
|---|---|---|
| Attention (full-width) | 6 × B·S·d·b | Pre-norm residual, post-RMSNorm input to Q/K/V, Q pre-RoPE, Q post-RoPE (HF's `apply_rotary_pos_emb` returns a fresh Q), attention output, o_proj input |
| Attention (GQA K/V) | 2 × B·S·d·(nkv/nh)·b | K and V are narrowed by the GQA ratio — for TinyLlama, nkv/nh = 4/32 = 1/8 |
| MLP pre-norm | 2 × B·S·d·b | Pre-norm residual + input to gate_proj/up_proj |
| MLP hidden (SwiGLU) | 4 × B·S·ff·b | Gate output (SiLU backward), SiLU output, up_proj output, and the gate·up product feeding down_proj — one more ff-width tensor than Article 1's two-linear MLP |
| Dropout masks | `[2·B·S·d + B·S·ff] · 1` byte | Only if `p_drop > 0`; TinyLlama's config sets both dropout rates to 0, zeroing this term |
| Attention probability matrix | **0** | Zero because `nn.functional.scaled_dot_product_attention` dispatches flash/memory-efficient kernels that never materialize the full seq×seq matrix. If a base forces eager attention, add back `B·nh·S²·b` |

> **Per-layer subtotal (non-checkpointed) = [6·B·S·d + 2·B·S·d·(nkv/nh) + 2·B·S·d + 4·B·S·ff] · b + dropout_masks**

Multiply by `L_bwd` (see §2), then add the once-per-model terms:

| Term | Formula |
|---|---|
| Embedding output | B·S·d·b |
| Final logits | B·S·V·b |
| fp32 logits upcast (bf16 mode only) | B·S·V·4 |
| CE backward fp32 (bf16 mode only) | 2 × B·S·V·4 |

Cross-entropy is numerically unstable in bf16, so autocast forces it to fp32 —
that materializes a full-size logits copy and two more fp32 vocab-width tensors
kept live through backward.

**Checkpointed (phase peaks — the maximum of three moments, not a running sum):**

Under checkpointing, activations look nothing like a running total. The peak
lands on the max of three phase peaks — see §3 for how these compose with the
static line items. The activation term itself, at the backward-recompute moment:

> Checkpointed activations = (L_bwd + 3)·B·S·d·4  +  6·B·S·d·b  +  2·B·S·d·(nkv/nh)·b  +  4·B·S·ff·b  +  [dropout]  +  B·S·V·b  +  2·B·S·V·4 (bf16 only)

Same two facts as Article 1's `memory-formulas.md` §2 checkpointed formula: the
saved layer boundaries are **fp32** (RMSNorm and residual adds run fp32 under
autocast, so the tensor passed *between* layers is 4 bytes/element even in bf16
mode), and the `+3` beyond `L_bwd` covers the segment being recomputed
(re-saved norm tensors plus the incoming gradient w.r.t. the segment output).

*Baseline non-checkpointed: 8.96 GB (calculated), 8.96 GB (measured).*
*Baseline checkpointed activations at their own peak: 2.44 GB (calculated), 2.44 GB (measured).*

---

## 2. Adapter placement — the lever that changes L_bwd

Adapter placement is the fine-tune-specific activation lever. `adapter_layers`
takes two shapes: `all` (adapters on every layer) or `upper-N` (adapters on the
top N layers only). Layers strictly below the shallowest adapter have no
trainable parameter above them via that path, their input has
`requires_grad=False`, and PyTorch's autograd saves nothing for their forward.

> L_bwd = L                if adapter_layers = "all"
> L_bwd = min(N, L)        if adapter_layers = "upper-N"

The result is exact — activations paid across `L_bwd` layers, not `L`. At
`upper-11` on the baseline, activations drop from 8.96 GB to 5.40 GB; at
`upper-3` they drop to 2.82 GB. (The once-per-model terms — embedding output,
final logits, and the two fp32 cross-entropy tensors — sum to ~1.85 GB at the
baseline geometry and set the floor for any placement, which is why the drop
saturates rather than approaching zero.) Same code, one config flag.

---

## 3. Peak composition

**Non-checkpointed:**

> Allocated total = frozen + adapter_weights + gradients + optimizer + autocast_cache + activations + cublas_workspace

**Checkpointed** — the maximum of three phase peaks, not their sum:

1. *Backward-recompute peak* — while the last segment is being recomputed:
   > frozen + adapter_weights + optimizer + activations_ckpt + cublas
   No adapter gradients yet (they accumulate through backward), no autocast
   cache (freed when forward exited autocast).

2. *Optimizer-step peak* — during Adam's parameter update:
   > frozen + adapter_weights + gradients + optimizer + adam_temps + cublas
   `adam_temps = gradients` for standard foreach-Adam (torch materializes one
   param-sized set of temporaries per group), zero for Adam8bit (updates in
   place). Unlike Article 1, this floor doesn't dominate — the trainable group
   is tiny, so the peak is essentially `frozen + optimizer`, not
   `20 · N_base` bytes.

3. *Forward peak* — every layer's saved boundary live, plus one segment's
   forward transient:
   > frozen + adapter_weights + optimizer + autocast_cache + L_bwd · B·S·d·4 + 6·B·S·d·b + 2·B·S·d·(nkv/nh)·b + 4·B·S·ff·b + logits_terms + cublas

> **Allocated total (checkpointed) = max(backward_peak, optimizer_step_peak, forward_peak)**

The article's "Putting it together" table lists the baseline checkpointed peak
as 4.58 GB measured, 4.71 GB calculated — the max is dominated by the forward
peak in this config, not by the optimizer step. This is the crux of why
checkpointing works so much better in fine-tuning than in Article 1's training:
there's no 20-bytes-per-parameter optimizer floor to inherit the ceiling.

---

## 4. cuBLAS workspace

> cuBLAS workspace = (4.316 + 1.871 · L_bwd) × B·S·d·4 bytes    if not checkpointing
> cuBLAS workspace = 0                                          if checkpointing

cuBLAS caches GEMM workspace lazily per unique matmul shape inside the CUDA
context. The two-component count was fit by OLS across four adapter-placement
rows (L_bwd = 3, 6, 11, 22) from the 2026-08-17 H100 sweep: a fixed base-model
term (4.316 buffers regardless of placement) plus a per-adapter-layer term
(1.871 buffers per active adapter layer). Each buffer is `B·S·d·4` bytes —
32 MiB at the baseline geometry.

The **checkpointed → zero** collapse is empirical: under gradient checkpointing
the allocator repeatedly runs against its cap during backward-recompute, and
cuBLAS releases cached workspaces back to the pool rather than holding them
across segments. Keeping the full L_bwd count would over-predict the three
checkpointed rows by 36–72%; zeroing it lands them at +2.8%, +8.1%, +8.1% —
inside the ±10% publishing bar. Same category of empirical claim as the
overhead constants below.

---

## 5. Overhead terms

Same shape as Article 1's, kept as three named constants rather than folded
into other terms:

> Reserved total = Allocated total + CUDA context overhead (0.6 GB) + fragmentation (1.2 GB non-checkpointed, 5.2 GB checkpointed)

Constants copied verbatim from `predictions.py` (Article 1's) — the fine-tune
sweep didn't produce evidence they miss. If a "will this fit" check is needed,
compare **reserved**, not allocated, against GPU capacity — that's what
`validate_finetune_results.py` does.

---

## 6. Confidence per term

Validated against the 25-config reference sweep in `benchmark/reference-run-finetune/`
(23 non-OOM, 2 correctly-predicted OOM).

| Term | Confidence | Basis |
|---|---|---|
| Frozen base weights (bf16/int8/int4 storage) | **High** | Byte-perfect on measured base parameter count × bytes/param |
| Adapter weights, gradients, optimizer state | **High** | Byte-perfect on measured adapter tensor sizes; verified across rank 4/8/16/64 rows |
| Autocast weight cache | **High** | Full-cache path validated in Article 1's reference-run; adapter-only residual verified within measurement noise |
| Activations (non-checkpointed, `adapter_layers=all`) | **High** | Baseline calculated 8.96 GB vs. measured 8.96 GB; every batch/seq/rank single-lever row lands within 0.7% |
| Adapter placement (L_bwd lever) | **High** | Four placement rows (all, upper-11, upper-6, upper-3) land within 0.05% of measured |
| GQA / SwiGLU / RoPE activation terms | **High** | Directly traced through HuggingFace LLaMA `LlamaDecoderLayer.forward` and confirmed against allocator-history replay in `debug_finetune_out.json` |
| SDPA-path attention probability matrix = 0 | **High** | Confirmed via source (SDPA dispatch) and by construction: a materialized matrix would demand hundreds of GB at baseline geometry |
| Activations (checkpointed) | **High** | Three checkpointed rows within +2.8%, +8.1%, +8.1% of measured; phase-peak model matches allocator-history replay |
| cuBLAS workspace, non-checkpointed | **Medium** | Two-component fit from four rows; explains the L_bwd scaling cleanly (R² > 0.999), but the constants are empirical |
| cuBLAS workspace = 0 under checkpointing | **Medium** | Consistent with allocator-history observation; not derived from first principles |
| CUDA context overhead, fragmentation constants | **Order-of-magnitude, not derived** | Structurally invisible to torch's memory API; copied from Article 1 rather than refit |
| Base precision int8 / int4 (bitsandbytes path) | **Predictions-only** | Formula rows exist in the spreadsheet's model presets; the reference sweep didn't include bitsandbytes-quantized rows |

---

## 7. Building the spreadsheet

Inputs (one row per scenario, or a named-cell block):

- `N_base`, `N_adapt` (or the derived formula `2 × rank × d × targets × L_adapt`)
- `d`, `L`, `L_adapt`, `nh`, `nkv`, `ff`, `V`
- `B`, `S`
- `precision` (`amp_bf16` / `fp32`) → sets `b` = 2 or 4, gates autocast cache and CE fp32 terms
- `base_storage_precision` (`fp32` / `bf16` / `int8` / `int4`) → sets base bytes/param
- `optimizer` (`adam` / `adam8bit`) → sets optimizer bytes/param = 8 or 2
- `adapter_layers` (`all` or `upper-N`) → sets `L_bwd`
- `checkpointing` (on/off)

Output cells, one line each:

1. `frozen_gb` = `N_base × base_bytes_per_param / 1e9`
2. `adapter_weights_gb` = `N_adapt × 4 / 1e9`
3. `gradients_gb` = `N_adapt × 4 / 1e9`
4. `optimizer_gb` = `N_adapt × IF(optimizer="adam8bit", 2, 8) / 1e9`
5. `autocast_cache_gb` = `IF(precision="amp_bf16", IF(base_precision="fp32", (N_base+N_adapt)×2, N_adapt×2), 0) / 1e9`
6. `per_layer_bytes` = `[6·B·S·d + 2·B·S·d·(nkv/nh) + 2·B·S·d + 4·B·S·ff] × b + dropout_masks`
7. `once_per_model_bytes` = `B·S·d·b + B·S·V·b + IF(precision="amp_bf16", 3·B·S·V·4, 0)`
8. `activations_gb` = `IF(checkpointing, checkpointed_activations, L_bwd × per_layer_bytes + once_per_model_bytes) / 1e9`
9. `cublas_gb` = `IF(checkpointing, 0, (4.316 + 1.871·L_bwd) × B·S·d·4) / 1e9`
10. `allocated_total_gb` = `IF(checkpointing, MAX(backward_peak, optimizer_step_peak, forward_peak), frozen + adapters + gradients + optimizer + autocast + activations + cublas) / 1e9`
11. `reserved_total_gb` = `allocated_total_gb + 0.6 + IF(checkpointing, 5.2, 1.2)`
12. `fits_gpu` = `reserved_total_gb ≤ 79.18 × 1.0737` *(≈ 85.0 GB usable for an 80GB H100/A100)*

That's the sizing tool, complete. Every cell traceable to a named line item in
§1–4; no fudge multipliers.

`assets/gpu-finetune-memory-sizing.xlsx` implements this. It passed an
audit pass on 2026-08-17 (six findings, three of them 7.8 GB-worth of blocker
bugs, all fixed). Baseline computes 14.535 GB reserved matching
`predict_line_items_finetune()` byte-perfect.

---

## 8. Note on the two OOM rows in the reference run

The reference-run CSV covers 25 configurations. Two — `combined=baseline` and
`combined=adam8bit`, both batch=32 seq=2,048 rank=8 across all 22 layers,
non-checkpointed — measured OOM on 80 GB. Their `predicted_*` cells were
originally empty because `benchmark_finetune.py` gates prediction behind
child-returned architecture info, and the child crashed on the first forward
before returning that info.

The predictor itself is CPU-side deterministic math and can be evaluated
statically for any config using the base model's known architecture. Both rows
were filled in post-hoc via `predict_line_items_finetune()` using TinyLlama's
static arch (`hidden_size=2048`, `num_hidden_layers=22`, `num_attention_heads=32`,
`num_key_value_heads=4`, `intermediate_size=5632`, `vocab_size=32000`) — identical
to what every non-OOM row in the same sweep recorded. Both predict allocated
totals of ~170 GB on an 80 GB card, well above capacity — matching the observed
OOM. This is documented here rather than folded silently into the CSV.

A future benchmark-side fix would have `benchmark_finetune.py` compute
predictions from `AutoConfig.from_pretrained` when the child fails, so this
manual step isn't repeated on future sweeps. Out of scope for this article.
