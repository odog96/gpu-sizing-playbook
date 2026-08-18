# Fine-Tuning: What Changes When Only 1% of the Weights Are Trainable

*Article 2 in a series on sizing GPU infrastructure for foundation models.*

Training a model from scratch is expensive. Fine-tuning is what most teams actually do — take an open-weight base model that already knows English and how to reason, and teach it your domain, your tone, your data. Parameter-efficient fine-tuning (PEFT) methods like LoRA go one step further: freeze the base model entirely and train only a small set of extra weights — the *adapters* — attached to specific projections inside each layer. A billion-parameter base ends up with roughly six million trainable parameters. Less than one percent.

That single change reshuffles every line item in the memory budget. The static costs that dominated Article 1 — gradients, optimizer state — collapse to the trainable count and become rounding error. A new line item appears: the frozen base itself, whose weights still have to live on the card even though they never receive an update. And a new set of levers appears with it, because when the static costs are small, the choices that used to be constraints — precision of the base, placement of the adapters, rank of the adapter matrices — are now free variables.

This article covers fine-tuning on a single GPU with LoRA. Full fine-tuning (updating every weight) uses Article 1's math unchanged; there's no LoRA-specific overlay on top. Multi-GPU strategies are a separate article.

Article 1's line items still apply, but they move. Weights splits into two — frozen base and trainable adapters — since the two have to be tracked separately:

**Frozen base weights.** New line item, and usually the largest static one. The base model has to live on the card even though it never trains — its forward pass is what the adapters modulate. Storage precision matters here in a way it didn't in Article 1, because the base can be stored at BF16, INT8, or INT4 without hurting training: the adapters absorb the quantization error.

**Trainable weights (adapters).** The LoRA adapters themselves — small enough that at rank 8 on a 1B-parameter base, they fit in tens of megabytes.

**Gradients.** One gradient per *trainable* parameter, not per base parameter. Frozen weights have no gradient at all; backprop flows through them but nothing is stored. Same collapse in size as the adapters.

**Optimizer states.** Adam's momentum plus variance, again on the trainable count only. Two orders of magnitude smaller than in Article 1 — the line item that used to dominate the static footprint now measures in megabytes.

**Activations.** Same formula as Article 1, same LLaMA-family architecture, with two adjustments: grouped-query attention narrows the K/V projections, and adapters constrain which layers actually need saved activations. Layers below the shallowest adapter have no trainable parameters above them and can skip saving.

Article 1 called out that activations dominate everything else combined. That was true when the static footprint was 16 bytes per parameter across a billion parameters. In fine-tuning, activations dominate by an even wider margin — the static footprint is now dominated by the *frozen base*, and everything on the trainable side is negligible. If Article 1's picture was "activations are the biggest number, and also where the cheapest lever lives," fine-tuning's picture is "activations plus the frozen base are the whole story, and both have levers."

---

## The line items, the levers, and what the levers cost

All figures below use one worked configuration, carried through the article: **TinyLlama-1.1B base (1.10 billion frozen parameters, 22 layers, hidden dimension 2,048, 32 attention heads, 4 key-value heads for grouped-query attention, feed-forward width 5,632, vocabulary 32,000), LoRA rank 8 on all seven projection targets (Q, K, V, O, gate, up, down) across all 22 layers (6.3 million trainable parameters — 0.57% of the base), sequence length 512 tokens, batch size 8, BF16 base storage, BF16 mixed precision, standard Adam.** Figures labeled *measured* come from a 23-configuration benchmark of this model on an H100 80GB — see the note on measurement at the end. Unlabeled figures in tables are calculated.

| Line item | Memory (calculated) | Levers to reduce it | What the lever costs you |
| :---- | :---- | :---- | :---- |
| Frozen base weights | 2.20 GB | INT8 base (halves it); INT4 / QLoRA (quarters it); smaller base | Quantization risks quality loss; a smaller base is a smaller base |
| Adapter weights | 25 MB | Lower rank; fewer target modules | Lower rank limits expressiveness; skipping the MLP targets is the biggest single-lever save but constrains what the adapter can learn |
| Gradients | 25 MB | (Adapters-only — nothing else contributes) | — |
| Optimizer states | 50 MB | 8-bit Adam (drops to ~13 MB) | Small accuracy risk on an already tiny line item — rarely worth pulling on |
| Autocast weight cache | 13 MB | None worth pulling — collapses to adapter-only when the base is BF16 or lower | Was 2 GB in Article 1 because the base was FP32; here it's a rounding error |
| Activations | ~9 GB | Activation checkpointing; smaller batch; shorter sequence; adapters only on upper N layers | Checkpointing adds training time (see Article 1); shorter sequences limit context; upper-N constrains what the adapter reaches |

Two observations. First, the entire trainable-side static footprint — adapters plus gradients plus optimizer plus autocast cache — is under 120 MB. All four of those line items combined would round to zero on the 80 GB card. In Article 1 they were 18 GB. The line items didn't go away; they just moved from "sizing constraint" to "footnote."

Second, the frozen base is now the largest static number on the page, and it's the one line item that has a real precision lever without the training-instability cost from Article 1. Weights that never receive an update cannot lose small updates to rounding, because there are no updates to lose. That's what makes INT4 (QLoRA) work as a routine choice here where it would be exotic in Article 1.

---

## Running the numbers

### Frozen base weights

frozen weight memory = base parameter count × bytes per parameter

Bytes per parameter is set by *base storage precision*, and this is the key difference from Article 1. Because the base never trains, storing it at BF16, INT8, or INT4 does not risk the training instability that Article 1 warned about. The adapters — which do train — remain at FP32 to preserve their small updates. The base's precision becomes a free variable, ranging from 4 bytes (FP32) to 0.5 bytes (INT4, two parameters packed per byte).

Worked configuration at BF16: 1.10e9 × 2 bytes = **2.20 GB** (calculated; measured 2.20 GB). The same base at INT4 would be **0.55 GB** — a 1.65 GB reduction on the largest static line item, from a single config flag.

QLoRA — the paper that popularized INT4 base + FP32 adapters — is the recipe that makes this practical. On a 70B-parameter base, the difference between BF16 weight storage (140 GB) and INT4 (35 GB) is the difference between "needs a multi-GPU setup" and "fits on one 80 GB card." The article's companion spreadsheet shows the crossover explicitly.

### Adapter weights

adapter memory = trainable parameter count × 4 bytes

Each target projection replaces a matrix `W` (shape `input × output`) with `W + B·A`, where `A` is `rank × input` and `B` is `output × rank`. That's `rank × (input + output)` new trainable parameters per target. For LLaMA-family models the input and output dimensions vary across targets: attention Q and O are `hidden × hidden` (square), grouped-query attention narrows K and V to `hidden × (hidden × kv_heads / heads)`, and the SwiGLU MLP has three rectangular projections between the hidden dimension and the wider intermediate width. Adapters stay at FP32 to preserve the small updates that a lower precision would round away.

Worked configuration: 6.3 million trainable parameters across all 22 layers × 4 bytes = **25 MB** (calculated; measured 25.23 MB). Two orders of magnitude below the frozen line item above it.

The rank sets the adapter's expressive capacity: rank 8 is the LoRA paper's suggested default and the point where most published fine-tunes land. Rank 4 halves this line item at some quality cost; rank 64 grows it 8× to about 200 MB — still a rounding error next to activations — and increases capacity substantially. The target-module set is a larger lever — restricting adapters to attention-only (Q, K, V, O — four targets instead of seven) shrinks this line item by 64%, because the SwiGLU MLP projections are wider than the attention ones and dominate the per-layer count. That constrains the adapter to modulate attention only, not the feed-forward path.

### Gradients

gradient memory = trainable parameter count × 4 bytes

Same shape as adapter weights, for the same reason as in Article 1: one gradient per trainable weight, matched to trainable-weight storage precision. Frozen parameters have no gradient stored — backprop still flows through them (the chain rule requires it), but the framework does not allocate storage for what it will never apply.

Worked configuration: 6.3e6 × 4 bytes = **25 MB** (calculated; measured 25.23 MB).

### Optimizer states

optimizer memory = trainable parameter count × 8 bytes

For Adam: momentum (4 bytes) + variance (4 bytes) per trainable parameter. Same arithmetic as Article 1 — the only change is what "trainable" counts. In Article 1 it was every parameter; here it's the adapters.

Worked configuration: 6.3e6 × 8 bytes = **50 MB** (calculated; measured 50.46 MB).

8-bit Adam was worth pulling on in Article 1 because the optimizer line item was 8 GB there and every gigabyte counted. Here it would take this line from 50 MB to 13 MB. The savings exist; they don't matter. The lever is worth keeping in mind for the *combined* case at the end of the article, where every megabyte helps push the peak below the OOM wall — but it's not the first thing to reach for.

### Autocast weight cache

The autocast weight cache — Article 1's fifth line item, a BF16 copy of every FP32 weight that autocast produces on the fly and reuses across the forward pass — was 2 GB there because the base was FP32. Fine-tuning changes this. When the base is loaded at BF16 (or lower) directly, there is nothing to cast for the base: it is already in the compute dtype. The cache collapses to the adapter-only residual — FP32 adapters cast to BF16 on the fly — which at rank 8 is roughly 13 MB.

Worked configuration: 6.3e6 × 2 bytes = **13 MB** (calculated; measured within the peak's noise floor).

If a base is loaded at FP32 (unusual for fine-tuning, but possible), the full Article 1 cache reappears and the 2 GB comes back. The spreadsheet models this; the article assumes BF16 base for the running example.

### cuBLAS workspace

One more line item, empirical rather than derived from architecture: PyTorch's matrix-multiply library caches workspace buffers inside the CUDA context, keyed by matmul shape. Each buffer is `batch × sequence × hidden × 4 bytes` — about 32 MiB at the worked geometry — and the count scales with the number of layers that host adapters. The reference-run sweep fits the count as roughly four base-model buffers plus two more per adapter layer, giving about 46 buffers at the baseline configuration.

Worked configuration, non-checkpointed: **~1.5 GB** (calculated; matches measured to within the fit's residual). Under activation checkpointing the allocator releases these caches between recompute segments and the term collapses to zero — one of the reasons checkpointing works as well as it does here.

### Activations

Activations follow the same formula as Article 1 with two architectural adjustments and one placement lever specific to fine-tuning.

The first adjustment is grouped-query attention. LLaMA-family models split attention heads into query heads and key-value heads at different counts — TinyLlama-1.1B uses 32 query heads and 4 KV heads, a ratio of 8:1. The saved K and V tensors are therefore `num_kv_heads / num_heads` as wide as the query tensor, which reduces the attention-side activation footprint. It's a real save — around 1 byte per element per layer in this configuration — but not a large one relative to the feed-forward hidden layer, which the next adjustment covers.

The second adjustment is SwiGLU. LLaMA replaces Article 1's two-linear MLP (up-project, activate, down-project) with a three-linear SwiGLU: a gate projection, an up projection, and a down projection. The gate output is multiplied by the up output before being fed to the down projection. That multiply chain saves *four* same-shape hidden-layer tensors, not the two from Article 1: the gate output for the SiLU backward, the SiLU output as one multiply operand, the up-projection output as the other, and the multiply product as the input to the down projection. Same order of magnitude as Article 1's MLP hidden layer, but larger by a factor of two.

Worked configuration, non-checkpointed: **8.96 GB** (calculated; measured 8.96 GB). The formula in `predictions_finetune.py` breaks this into six named terms per layer plus four once-per-model terms; the article's numbers match those to within 0.05%.

The third adjustment — the fine-tuning-specific one — is *adapter placement*. Article 1 assumed every layer trains, so every layer saves activations. In fine-tuning you can attach adapters to only the top N layers. Layers below the shallowest adapter have no trainable parameters above them via that path, their inputs are `requires_grad=False`, and PyTorch's autograd saves nothing for their forward pass.

The lever is exact: adapters on the upper N layers save activations across N layers, not 22. At upper-11, activations drop from 8.96 GB to 5.40 GB (calculated); at upper-3, they drop to 2.82 GB (calculated). This is the only single-lever choice that reaches the largest line item on the page without touching batch, sequence, or checkpointing.

#### Activation checkpointing

Activation checkpointing does the same thing here as it does in Article 1: store only layer inputs, recompute the interior on the backward pass. Saved inputs still cost 4 bytes per element even under BF16 — LayerNorm and residual adds run FP32, so the tensor passed between layers is FP32. The peak still lands on the max of three moments (backward-recompute, optimizer step, forward with all inputs saved), not on a running sum.

Worked configuration, checkpointed: activations drop from 8.96 GB to **2.44 GB** (calculated) — a 3.7× reduction on the dominant line item. Total allocated drops from 12.80 GB to 4.58 GB (both measured), a 2.8× reduction on the whole run.

The training-time cost is the same as in Article 1: roughly a third more wall clock per step, because part of the forward pass runs twice.

---

## Putting it together

Same worked configuration as above: TinyLlama-1.1B base, LoRA rank 8 on all seven targets across all 22 layers, batch 8, sequence 512, BF16 base, BF16 mixed precision, standard Adam.

| Line item | Without activation checkpointing | With activation checkpointing |
| :---- | :---- | :---- |
| Frozen base weights | 2.20 GB | 2.20 GB |
| Adapter weights | 0.025 GB | 0.025 GB |
| Gradients | 0.025 GB | 0.025 GB |
| Optimizer states | 0.050 GB | 0.050 GB |
| Autocast weight cache | 0.013 GB | 0.013 GB |
| Activations | ~9.0 GB | ~2.4 GB (at their own peak) |
| cuBLAS workspace | ~1.5 GB | released under checkpointing |
| **Peak allocated (measured)** | **12.80 GB** | **4.58 GB** |
| Allocator overhead: cached blocks and fragmentation (measured) | ~1.7 GB | ~1.0 GB |
| **Total to size against (measured reserved)** | **14.49 GB** | **5.57 GB** |

Add another 0.6 GB on top of the reserved figure for the CUDA context — same driver and kernel-library overhead as Article 1.

Two observations. First, the entire trainable-side static footprint — adapters plus gradients plus optimizer plus autocast cache — is 113 MB. Line them up next to Article 1's 18 GB and you can see why fine-tuning is on a different order of hardware. Second, checkpointing on the training run in Article 1 didn't reduce the peak by much (activations at the backward peak fell, but the optimizer-step peak took over as the ceiling). In fine-tuning, checkpointing *does* cut the peak by nearly 3×, because there is no 20-bytes-per-parameter optimizer floor to inherit the ceiling. The optimizer step touches only the adapters; the frozen base is a spectator.

The practical read: TinyLlama-1.1B at these settings fine-tunes on a 24 GB consumer card with headroom to spare. With checkpointing on, it fits with room to run a batch four times larger. This is the size gap the article's opening paragraph named — the same math from Article 1, now demanding roughly one-fifth the hardware for the same model.

---

## Checked against hardware

Every formula in this article was validated against a 23-configuration LoRA fine-tune sweep of TinyLlama-1.1B on an H100 80GB — varying batch size, sequence length, LoRA rank, adapter placement, base storage precision, and checkpointing one lever at a time for the core sweeps, plus four deliberately-stacked combinations at the extremes. For every configuration that completed, predicted memory landed within ~8.3% of measured; every configuration that ran out of memory was predicted to run out. The two configurations that OOM'd — batch 32 with sequence 2,048, at rank 8 across all layers, with and without 8-bit Adam — needed checkpointing to fit, and checkpointing rescued both.

![Predicted vs. measured peak allocated memory for the 23-configuration LoRA fine-tune sweep on TinyLlama-1.1B, colored by lever. Points sit on or inside the shaded ±10% band around the y=x diagonal; the one point below the line at ~41 GB predicted / ~38 GB measured is the stacked-lever combined configuration where the formula is 8.3% high.](predicted-vs-measured.png)  
*Every configuration in the sweep, formula against reality: 23 points, one 8.3% miss, no fudge multipliers. Colors mark which lever was moved from the baseline.*

The fit boundaries for TinyLlama-1.1B at the worked settings — measured where the sweep covered them, calculated where it did not:

| | No levers | With checkpointing |
| :---- | :---- | :---- |
| Batch size (at seq 512) | 32 fits (measured 44 GB); formula predicts OOM near batch 48 | 32 fits (calculated 14 GB); formula predicts much larger sizes fit |
| Sequence length (at batch 8) | 2,048 fits (measured 44 GB); formula predicts OOM near seq 3,000 | 2,048 fits (calculated 11 GB); formula predicts much larger sizes fit |
| Combined (batch 32, seq 2,048) | measured OOM on 80 GB | measured 38 GB, fits with headroom |

That last row is the pattern this article was written to make visible. The stacked-lever configuration — every dial pushed hard — does not fit on the 80 GB card without checkpointing. Turn checkpointing on and it fits with 40 GB of headroom. Same model, same card, one config flag.

The worst prediction miss was on the two stacked-lever checkpointed configurations, where the formula predicts +8.3% over what was measured. The formula's phase-peak model treats the forward-with-checkpointing peak as if all saved boundaries plus one segment's transient are live simultaneously; the allocator in practice releases some of the intermediate transients slightly earlier under memory pressure. That's an honest formula-side gap, not a fudge factor's worth of slack. It is reported rather than tuned away, matching Article 1's disclosure of its own 11.6% miss from the earlier training sweep.

---

## What these numbers do not cover

**Full fine-tuning.** These formulas cover LoRA and its variants — recipes where a frozen base hosts a small trainable adapter. Full fine-tuning updates every weight; it uses Article 1's math directly, sized to the base model's parameter count. The line-item collapse from this article does not apply.

**Adapters beyond LoRA.** The general shape — frozen base plus small trainable adjustment — carries over to related methods (IA³, DoRA, prefix-tuning), but the specific `2 × rank × hidden × targets × layers × 4` count is LoRA's. Other adapter families need their own trainable-parameter count; the rest of the memory story is unchanged.

**Ragged batches and data pipelines.** Same caveats as Article 1: peak memory is driven by the longest sequence in the batch, not the average, and GPU-resident data pipelines can hold significant VRAM outside the model. Budget for both separately.

**A note on measurement.** Figures labeled measured come from a 23-configuration LoRA benchmark on a single H100 80GB. The static line items are measured directly — by summing the actual bytes of every base parameter, adapter parameter, gradient, and optimizer tensor on the device — not inferred. Activations are the difference between two *measured* allocator readings. Peak-composition claims come from replaying the CUDA allocator's event history (`benchmark/debug_finetune.py`).

A companion spreadsheet — `gpu-finetune-memory-sizing.xlsx` at the repo root — turns this math into a sizing tool. Plug in your base model (LLaMA-2-7B, LLaMA-3-8B, and larger presets are included), your LoRA rank and target set, your batch and sequence, and it tells you whether the job fits on the GPU you have. And whether QLoRA on a smaller card would fit it more comfortably.

---

**Next:** Inference — where a different memory consumer, the KV cache, takes over. *(coming soon)*
