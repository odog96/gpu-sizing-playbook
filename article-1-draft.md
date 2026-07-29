# Training: The Four Line Items of GPU Memory

*Article 1 in a series on sizing GPU infrastructure for foundation models. \[Article 0 — the decision chain — is here.\]*

You've decided to train a foundation model. The first question that gates everything else: will it fit on your GPUs? Get this wrong and you find out the expensive way — an out-of-memory error hours into a run.

The good news is that training memory is predictable. It comes down to four line items, and you can estimate each one on the back of a napkin before you provision anything. This article walks the math for a 1-billion-parameter model, then gives you the levers to pull when the total doesn't fit. The companion spreadsheet does the arithmetic for you.

## The working example

We'll size a 1B-parameter tabular foundation model — the kind NVIDIA's blueprint trains at 29M parameters, scaled up. The same math applies to an LLM; only the input shape changes.

| Setting | Value |
| :---- | :---- |
| Parameters (N) | 1 billion |
| Training precision | BF16 (2 bytes per value) |
| Architecture | 20 layers, hidden dimension 2,048 |
| Features per row (the "sequence")¹ | 100 |
| Batch size² | 1,024 rows |
| Optimizer | Adam (mixed precision) |

¹ *Sequence length is how many tokens the model processes per example. For an LLM that's words in a passage; for a tabular model, each column becomes a token, so a 100-column row is a sequence of 100\.* ² *Batch size is how many examples the model processes simultaneously in one training step. Bigger batches train faster per epoch but cost memory.*

## Line item 1: Model weights — 2 GB

The weights are the model itself — the values training exists to learn. At 2 bytes per parameter:

> 1B parameters × 2 bytes \= **2 GB**

This is the number everyone knows. It's also the smallest of the four. If you sized your GPU by weights alone, you'd be off by roughly a factor of ten.

## Line item 2: Gradients — 2 GB

During training, gradients are a register of changes: for every weight, the change to be made at the next update as the model improves. One gradient per parameter, same precision as the weights:

> 1B parameters × 2 bytes \= **2 GB**

## Line item 3: Optimizer states — 12 GB

The optimizer decides how to apply those changes, and to do it well it keeps history — running statistics about how each parameter's gradients have behaved. Standard mixed-precision Adam stores three 32-bit values per parameter (two history statistics plus a full-precision master copy of the weights for numerical stability):

> 1B parameters × 12 bytes \= **12 GB**

Read that again: the optimizer's bookkeeping costs six times the model itself. This is the line item that surprises people, and it's a fixed cost — it doesn't shrink with batch size.

**Static subtotal: 2 \+ 2 \+ 12 \= 16 GB**, before we've processed a single row of data.

## Line item 4: Activations — the variable cost

Activations are the intermediate outputs each layer produces as data flows through the model. If the weights are the coefficients of a giant system of equations, activations are the values passing through it — and every one must be kept in memory until the backward pass uses it to compute gradients.

Unlike the first three line items, activations scale with your data settings:

> Activations ≈ Batch × Sequence × Hidden dim × 2 bytes × Layers

Per row in our example: 100 × 2,048 × 2 bytes × 20 layers ≈ **8 MB per row**. At batch size 1,024:

> 1,024 rows × 8 MB ≈ **8 GB**

Honest caveat: this is a floor. Attention matrices and feed-forward intermediates add a multiplier — plan for 2–4× this estimate depending on architecture. The spreadsheet exposes that multiplier so you can set it to match your stack.

The critical insight: at batch 1,024 activations are manageable; at batch 4,096 the floor alone is \~33 GB and dominates everything else. Activations are where memory blows up — and where you have the most control.

## The total, and your levers

| Line item | Memory | Your levers |
| :---- | :---- | :---- |
| Weights | 2 GB | Lower-precision training (BF16 is already standard; FP8 on newest hardware) |
| Gradients | 2 GB | Follows weight precision |
| Optimizer states | 12 GB | 8-bit Adam (\~12 GB → \~3 GB); shard across GPUs with ZeRO (see appendix) |
| Activations | \~8–32 GB | **Biggest levers live here:** reduce batch size (recover throughput with gradient accumulation); reduce sequence length/feature count if training goals allow; activation checkpointing — recompute instead of store, cutting this line \~10× for \~30% extra compute |
| Framework overhead | \~2 GB | Effectively fixed |
| **Total** | **\~26–50 GB** |  |

## What this means for hardware

At batch 1,024 with no tricks, our 1B model wants \~26 GB — a tight fit on a 24 GB card (RTX 4090, A10G), comfortable on an A100 40 GB. Want batch 4,096? Either move to an 80 GB card, or pull two levers — 8-bit Adam and activation checkpointing — and the same run fits in roughly 24 GB.

That's the pattern to internalize: the static 16 GB is your entry fee, activations are your throttle, and the levers trade compute time for memory. The spreadsheet lets you test combinations before you provision anything.

If your total exceeds the largest single GPU you can get, you're not stuck — you're in multi-GPU territory. See the appendix.

---

## Appendix: When one GPU isn't enough

Say your sizing lands at 120 GB and your largest available card is an A100 80 GB. You need parallelization, and there are two fundamentally different strategies:

**Data parallelism** — every GPU holds a full copy of the model; each processes a different slice of the batch. This scales *throughput*, not capacity: per-GPU memory doesn't drop. The exception is ZeRO-style sharding, which splits the optimizer states (and optionally gradients and weights) across GPUs — with 4 GPUs, that 12 GB optimizer line becomes 3 GB each. Use when the model fits on one GPU but you want bigger effective batches or faster epochs.

**Tensor parallelism** — the model itself is split across GPUs, each holding a slice of every layer. This is how you fit a model that's too big for any single card: weights, gradients, optimizer states, and activations all divide across the group. The cost is heavy inter-GPU communication, so it wants fast interconnect (NVLink) and works best within a single node.

The decision rule: **if the static footprint fits on one GPU, stay data-parallel (add ZeRO if memory is tight). If it doesn't, tensor parallelism is mandatory, and everything else layers on top.** Full multi-node training strategy is its own article — for sizing purposes, this decision rule is what you need.

---

*Approximations throughout: real frameworks fragment memory, architectures vary, and your mileage will vary ±20%. Size for headroom. Next up — fine-tuning, where most of these line items shrink dramatically.*  
