# Fine-Tuning: The Memory Budget with a Frozen Base

*Article 2 in a series on sizing GPU infrastructure for foundation models.*

Training a model from scratch is time consuming and can be expensive, which is why many teams rely on fine-tuning — take an open-weight base model that already knows English and how to reason, and teach it your domain, your tone, and your data. Parameter-efficient fine-tuning (PEFT) methods like LoRA (Low-Rank Adaptation) go one step further: freeze the base model entirely and train only a small set of extra weights — the *adapters*. Each adapter attaches to one of the linear layers inside a transformer block — its *projections*, in the language of the architecture — and works by taking the high-dimensional representation flowing through that layer, projecting it down into a much smaller space, adjusting it there, and projecting it back up. Some information is lost in the low-rank detour, but enough survives to meaningfully adapt the model, and because only the adapter parameters train, the trainable count is a small fraction of the base: a billion-parameter base ends up with roughly six million trainable parameters. Less than one percent.

Going from full model training to fine-tuning changes how we think about each line item in the memory budget. The static costs that dominated the training article — gradients, optimizer state — become rounding error. A new line item appears: the frozen base itself, whose weights still have to reside in the card's memory even though they never receive an update.

This article covers fine-tuning on a single GPU with LoRA. Full fine-tuning (updating every weight, no adapters) uses the training article's math directly — this article's LoRA-specific accounting doesn't apply. Multi-GPU strategies will be covered in future series.

This article builds directly on the training article's four-line-item framework. The definitions for weights, gradients, optimizer states, activations, activation checkpointing, and the BF16 weight cache all still apply here. Readers who haven't read the training article should start there — the fine-tuning story is much shorter when you don't have to reconstruct the baseline.

When sizing GPU memory for a LoRA fine-tune, two line items dominate: activations and the frozen base. Together they hold 11.2 GB in the worked configuration below. Everything else combined is less than 1% of the total memory budget.

---

## The full memory picture

All figures below use one worked configuration, carried through the article: **TinyLlama-1.1B base (1.10 billion frozen parameters, 22 layers, hidden dimension 2,048, 32 attention heads, 4 key-value heads for grouped-query attention, feed-forward width 5,632, vocabulary 32,000), LoRA rank 8 on all seven projection targets — Q, K, V, O (the four attention projections) and gate, up, down (the three MLP projections) — across all 22 layers (6.3 million trainable parameters, 0.57% of the base), sequence length 512 tokens, batch size 8, BF16 base storage, BF16 mixed precision, standard Adam.** Rank sets the adapter's capacity (8 is the LoRA paper's default). Figures labeled *measured* come from a 23-configuration benchmark of this model on an H100 80GB — see the note on measurement at the end. Unlabeled figures in tables are calculated.

| Line item | No checkpointing | With checkpointing | Levers to reduce it |
| :---- | :---- | :---- | :---- |
| Frozen base weights | 2.20 GB | 2.20 GB | INT8 base (halves it); INT4 base — QLoRA — quarters it; smaller base |
| Adapter weights | 0.025 GB | 0.025 GB | — |
| Gradients | 0.025 GB | 0.025 GB | — |
| Optimizer states | 0.050 GB | 0.050 GB | — |
| Autocast weight cache | 0.013 GB | 0.013 GB | — |
| Activations | ~9.0 GB | ~2.4 GB (at their own peak) | Activation checkpointing; smaller batch; shorter sequence; adapters only on upper N layers |
| Matrix-multiply workspace | ~1.5 GB | released under checkpointing | — |
| **Peak allocated (measured)** | **12.80 GB** | **4.58 GB** |  |
| Allocator overhead: cached blocks and fragmentation (measured) | ~1.7 GB | ~1.0 GB |  |
| **Total to size against (measured reserved)** \* | **14.49 GB** | **5.57 GB** |  |

\* Add 0.6 GB on top for the CUDA context — driver overhead that lives outside the allocator's pool, same fixed cost the training article accounted for.

The frozen base is the largest static number on the page, and it's the one line item that has a real precision lever without the training-instability cost that showed up during full training. Weights that never receive an update cannot lose small updates to rounding, because there are no updates to lose. That's what makes INT4 (QLoRA — INT4 base + FP32 adapters) work as a routine choice here where it would be exotic in full training. During full training the four small items above summed to 18 GB — a sizing constraint. Here they're a footnote.

The drivers behind each row:

**Frozen base weights.** The base sits in memory because the forward pass runs through it, even though it never trains. The base can be held at BF16, INT8, or INT4 without measurable loss to fine-tune quality, since they are not the weights we are training, only the adapter weights are.

**Activations.** Same rules as during training — the forward pass is unchanged by which weights are frozen. The fine-tuning-specific lever is adapter placement: attach adapters to only the top N layers, and the framework skips saving activations below the shallowest adapter.

**Everything else.** Adapter weights, gradients, optimizer states, and the autocast cache all scale with the trainable count — 113 MB combined at these settings. The matrix-multiply workspace adds ~1.5 GB when checkpointing is off and vanishes when it's on.

---

## Running the numbers

### Frozen base weights

frozen weight memory = base parameter count × bytes per parameter

The benefit of freezing the base is that the base weights, which are not being modified, can be represented in lower precision without harming the fine-tune. The adapter weights, which we *are* modifying, stay at FP32 to preserve the accuracy of their small updates. The base's precision therefore becomes a free variable, ranging from 4 bytes (FP32) down through 2 bytes (BF16), 1 byte (INT8, 8-bit integer), and 0.5 bytes (INT4, 4-bit integer — two parameters packed per byte) — 8× cheaper storage on the largest static line item on the page, without the training-instability cost that would follow from doing the same in full training.

Worked configuration at BF16: 1.10e9 × 2 bytes = **2.20 GB** (calculated; measured 2.20 GB). The same base at INT4 would be **0.55 GB** — a 1.65 GB reduction on the largest static line item, from a single config flag.

QLoRA — the paper that popularized INT4 base + FP32 adapters — is the recipe that makes this practical. On a 70B-parameter base, the difference between BF16 weight storage (140 GB) and INT4 (35 GB) is the difference between "needs a multi-GPU setup" and "fits on one 80 GB card." The article's companion spreadsheet shows the crossover explicitly.

### Activations

Activations follow the training-article rules — same forward pass, same saved tensors, same dependence on batch size, sequence length, and checkpointing. Two things adjust the footprint in this article: the base architecture (specifics vary by model family — see the appendix note on LLaMA's GQA and SwiGLU if you want the mechanism) and adapter placement, the one lever that is unique to fine-tuning.

Worked configuration, non-checkpointed: **8.96 GB** (calculated; measured 8.96 GB). The formula in `predictions_finetune.py` breaks this into six named terms per layer plus four once-per-model terms; the article's numbers match those to within 0.05%.

The fine-tuning-specific lever is *adapter placement*. Full training assumes every layer trains, so every layer saves activations. In fine-tuning you can attach adapters to only the top N layers. Layers below the shallowest adapter have no trainable parameters downstream, and the framework saves no activations for their forward pass.

The lever is exact: adapters on the upper N layers save activations across N layers, not 22. At upper-11, activations drop from 8.96 GB to 5.40 GB (calculated); at upper-3, they drop to 2.82 GB (calculated). This is the only single-lever choice that reaches the largest line item on the page without touching batch, sequence, or checkpointing.

#### Activation checkpointing

Activation checkpointing works exactly as described in the training article — same mechanism, same 4-bytes-per-element budget. The fine-tuning outcome is different: the optimizer step's memory demand doesn't take over the peak the way it does in full training, so cutting activations cuts the peak directly.

Worked configuration, checkpointed: activations drop from 8.96 GB to **2.44 GB** (calculated) — a 3.7× reduction on the dominant line item. Total allocated drops from 12.80 GB to 4.58 GB (both measured), a 2.8× reduction on the whole run.

The training-time cost is the same as before: roughly a third more wall clock per step, because part of the forward pass runs twice.

### Everything else

All four items below scale with the trainable-parameter count and collapse to rounding error when that count is <1% of the base. Formulas are here for anyone chasing a specific number, but no single item has a lever worth pulling on its own.

*Adapter weights* — trainable parameter count × 4 bytes. LoRA replaces each target projection `W` with a small trainable pair whose product is added on top; the count depends on the rank and on how many targets receive adapters per layer. Adapters stay at FP32 so their small updates aren't rounded away. Worked configuration: 6.3M parameters × 4 bytes = **25 MB** (calculated; measured 25.23 MB). Rank sets adapter capacity (8 is the LoRA paper's default).

*Gradients* — trainable parameter count × 4 bytes. The framework allocates gradient storage only for trainable weights; the frozen base has none. Worked configuration: **25 MB** (calculated; measured 25.23 MB).

*Optimizer states* — trainable parameter count × 8 bytes for Adam (momentum + variance). Same arithmetic as during full training, applied to <1% of the parameter count. Worked configuration: **50 MB** (calculated; measured 50.46 MB). 8-bit Adam would cut this to ~13 MB — a lever from the training article that stops earning its keep here.

*Autocast weight cache* — 2 bytes per trainable weight when the base is already loaded at BF16, because there is nothing left to cast for the base. Only the FP32 adapters produce a cast-and-cache residual. Worked configuration: 6.3M × 2 bytes = **13 MB** (calculated; measured within the peak's noise floor). If the base is loaded at FP32 (unusual for fine-tuning), the full training-era cache reappears — the spreadsheet models this.

### Matrix-multiply workspace

Matrix-multiply workspace is a cache the framework holds for the many matrix multiplications a fine-tune step performs. It's empirical rather than derived from architecture, and we don't go further into its mechanics here because there is no lever specific to it. Worked configuration, non-checkpointed: **~1.5 GB** (matches measurements to within a few tenths of a GB across the sweep). Under activation checkpointing the allocator releases these caches between recompute segments and the term drops to essentially zero — it comes with the run when checkpointing is off, and goes away when it's on.

---

## Two take-aways

First, the entire trainable-side static footprint — adapters plus gradients plus optimizer plus autocast cache — is 113 MB. Line them up next to the 18 GB that dominated full training and you can see why fine-tuning is on a different order of hardware.

Second, in full training, activation checkpointing shrinks activations dramatically — but the optimizer step has its own memory demand (about 20 bytes per trainable parameter), and once activations shrink, that demand becomes the new peak. In fine-tuning, that constraint disappears: the optimizer step only touches the adapters, so checkpointing cuts the peak by nearly 3× instead of running into a new ceiling. The frozen base is a spectator to the update.

What this means in practice: TinyLlama-1.1B at these settings fine-tunes on a 24 GB consumer card with headroom to spare. With checkpointing on, it fits with room to run a batch four times larger. This is the size gap the opening paragraph named — the same math as full training, now demanding roughly one-fifth the hardware for the same model.

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

The worst prediction miss was on the two stacked-lever checkpointed configurations, where the formula predicts +8.3% over what was measured. The formula estimates the checkpointed peak by assuming all saved layer boundaries and one segment's transient tensors are alive at the same time; the allocator in practice releases some of the intermediate transients slightly earlier under memory pressure. That's an honest formula-side gap, not a fudge factor's worth of slack. It is reported rather than tuned away, matching the training article's disclosure of its own 11.6% miss from that sweep.

---

## What these numbers do not cover

**Full fine-tuning.** These formulas cover LoRA and its variants — recipes where a frozen base hosts a small trainable adapter. Full fine-tuning updates every weight; it uses the training article's math directly, sized to the base model's parameter count. The line-item collapse from this article does not apply.

**Adapters beyond LoRA.** The general shape — frozen base plus small trainable adjustment — carries over to related PEFT families (IA³, DoRA, and prefix-tuning, for instance), but the specific per-target parameter arithmetic is LoRA's. Each other family needs its own trainable-parameter count; the rest of the memory story is unchanged.

**Ragged batches and data pipelines.** Same caveats as during full training: peak memory is driven by the longest sequence in the batch, not the average, and GPU-resident data pipelines can hold significant VRAM outside the model. Budget for both separately.

**A note on measurement.** Figures labeled measured come from a 23-configuration LoRA benchmark on a single H100 80GB. The static line items are measured directly — by summing the actual bytes of every base parameter, adapter parameter, gradient, and optimizer tensor on the device — not inferred. Activations are the difference between two *measured* allocator readings. Peak-composition claims come from replaying the CUDA allocator's event history (`benchmark/debug_finetune.py`).

A companion spreadsheet — `assets/gpu-finetune-memory-sizing.xlsx` — turns this math into a sizing tool. Plug in your base model (LLaMA-2-7B, LLaMA-3-8B, and larger presets are included), your LoRA rank and target set, your batch and sequence, and it tells you whether the job fits on the GPU you have. And whether QLoRA on a smaller card would fit it more comfortably.

---

## Appendix: LLaMA architecture notes

Two LLaMA-family details show up in the activation math above. Neither is a lever — the reader doesn't tune them to size the job — but they explain why TinyLlama's per-layer activation footprint differs from a plain-vanilla transformer's. Skip this section unless you want the mechanism.

*Grouped-query attention (GQA).* LLaMA-family models split attention heads into query heads and key-value heads at different totals — TinyLlama-1.1B uses 32 query heads and 4 KV heads, a ratio of 8:1. The saved K and V tensors are therefore `num_kv_heads / num_heads` as wide as the query tensor, which shrinks the attention-side activation footprint. It's a real save — around 1 byte per element per layer in this configuration — but small compared with the feed-forward hidden layer covered next.

*Gated MLP (SwiGLU).* Instead of the classic two-linear MLP, LLaMA's variant uses three linear projections, saving roughly twice as many intermediate tensors per layer — about four instead of two — which is why the per-layer activation footprint is larger than in a plain transformer.

---

**Next:** Inference — where a different memory consumer, the KV cache, takes over. *(coming soon)*
