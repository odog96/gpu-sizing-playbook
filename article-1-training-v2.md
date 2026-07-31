# Training: The Four Line Items of GPU Memory

*Article 1 in a series on sizing GPU infrastructure for foundation models.*

For most predictive models, GPU memory is an afterthought. You train the model, it fits, you move on. Transformer models change that. Memory becomes a primary planning concern, and the decisions that drive it are made before training starts — not discovered hours into a run.

What this article offers is a straightforward way to think about how pre-training choices map to hardware requirements. Model size, sequence length, batch size, and weight representation each push memory in a specific direction. Once you can see which choice drives which cost, you can also see your levers to bring memory down, and what each of those levers costs you.

This article covers training on a single GPU. Multi-GPU strategies are a separate article.

Memory during a full training run comes down to four line items:

**Weights.** The most straightforward. The number of parameters is the number of weights. Your storage precision tells you how many bytes each one takes. A 16-bit floating point format is 2 bytes per weight, so a 1-billion-parameter model needs about 2 GB just to hold the weights.

**Gradients.** During the backward pass, each weight gets a gradient — the adjustment to apply to that weight. One gradient per weight, same formula as above.

**Optimizer states.** The optimizer controls the learning rate dynamically, and to do that it needs history from earlier training steps. For Adam in a mixed-precision setup, that history plus a full-precision master copy of the weights works out to 12 bytes per weight.

**Activations.** If you naively treat the weights in a model as the coefficients in a system of equations, the activations are the inputs flowing through at a given moment. (This is a stylized example — the real structure is more involved.) Memory here is a function of three choices: sequence length, how long your training examples are in tokens; batch size, how many examples you process at once; and activation checkpointing, which lets you keep a partial record of activations rather than all of them at once. That last one has outsized benefits, with a cost in training time.

The first three line items are static. They do not change whether your batch size is 1 or 4,096. The fourth is dynamic, and it is usually the largest number on the page — often by a wide margin.

---

## The line items, the levers, and what the levers cost

All figures below use one worked configuration, carried through the whole article: **1.01 billion parameters, 20 layers, hidden dimension 2,048, sequence length 100 tokens, batch size 256 examples, BF16 storage precision, Adam optimizer with FP32 master weights.**

| Line item | Memory | Levers to reduce it | What the lever costs you |
| :---- | :---- | :---- | :---- |
| Weights | 2.0 GB | Lower storage precision (BF16 → INT8); fewer parameters | Reduced accuracy, and below 8-bit a real risk of training instability; a smaller model is a smaller model |
| Gradients | 2.0 GB | Gradient accumulation across micro-batches | Trades wall-clock time for memory; the total memory saved comes from the smaller micro-batch, not from the gradients themselves |
| Optimizer states | 12.1 GB | 8-bit Adam; a stateless optimizer such as SGD | 8-bit Adam carries a small accuracy risk; SGD often converges slower or to a worse result |
| Activations | \~43 GB | Activation checkpointing; smaller batch size; shorter sequence length | Activation checkpointing adds roughly a third to training time; smaller batches can slow or destabilize convergence; shorter sequences limit what the model can learn |

Two observations. First, optimizer states are six times the weights — the largest static cost, and the one most often left out of a budget. Second, activations dominate everything else combined, and they are also where your cheapest lever lives.

---

## Running the numbers

### Weights

weight memory \= parameter count × bytes per parameter

Bytes per parameter is set by storage precision: 4 bytes for FP32, 2 bytes for BF16 or FP16, 1 byte for INT8.

Worked configuration: 1.01e9 × 2 bytes \= **2.0 GB** (calculated).

### Gradients

gradient memory \= parameter count × bytes per parameter

Same shape as weights, because there is exactly one gradient per weight.

Worked configuration: 1.01e9 × 2 bytes \= **2.0 GB** (calculated).

### Optimizer states

optimizer memory \= parameter count × bytes per parameter of optimizer state

For Adam in mixed precision, three values are held per parameter, all in FP32:

- Momentum (first moment): 4 bytes  
- Variance (second moment): 4 bytes  
- Master weight copy: 4 bytes

That is 12 bytes per parameter. Worked configuration: 1.01e9 × 12 bytes \= **12.1 GB** (calculated).

The master weight copy exists because applying small updates to BF16 weights loses precision. The optimizer keeps a full-precision copy, applies updates to it, and casts back down.

Worth noting: mixed precision does not reduce your static memory. BF16 weights plus BF16 gradients plus 12 bytes of Adam state is 16 bytes per parameter. FP32 weights plus FP32 gradients plus 8 bytes of plain Adam state is also 16 bytes per parameter. Mixed precision buys you speed and lower activation memory, not a smaller static footprint.

If this line is your constraint, 8-bit Adam stores momentum and variance in 1 byte each instead of 4, taking the total from 12 bytes to roughly 6 bytes per parameter — about **6 GB** for this model (calculated).

### Activations

Activations are the only line item that scales with your data, and they are the reason a model that looks like it needs 16 GB will not fit on a 40 GB card.

activation memory ≈ layers × batch size × sequence length × hidden dimension × bytes per element per layer

Where:

- **layers** is the number of transformer blocks (20 here)  
- **batch size** is examples processed simultaneously (256 here)  
- **sequence length** is tokens per example (100 here)  
- **hidden dimension** is the model's internal width (2,048 here)  
- **bytes per element per layer** is the term that surprises people: roughly **40 to 45 bytes** without activation checkpointing

Worked configuration: 20 × 256 × 100 × 2,048 × 44 bytes ≈ **43 GB** (measured on an A100 80GB; see the note on measurement below).

That last constant is where most back-of-the-envelope estimates go wrong. The intuitive guess is 2 bytes per element per layer — one BF16 tensor of shape (batch × sequence × hidden dimension) saved per layer. The real number is roughly twenty times that, because a transformer block does not save one intermediate tensor. It saves fifteen to twenty of them: the layer-norm inputs, the query/key/value projections, the attention output, the feed-forward inputs, and the feed-forward hidden layer, which is typically four times the hidden dimension on its own. All of them are needed again in the backward pass.

If you take one number from this article, take that one. Estimating activations at 2 bytes per element per layer will tell you a job fits when it needs twenty times the memory you budgeted.

#### What "batch" actually means here

Batch size and sequence length both appear as straight multipliers, which means what really drives activation memory is their product: **tokens per batch**, not rows per batch.

For anyone coming from tabular machine learning, this is the trap. A batch of 256 rows sounds small. If each row tokenizes into 100 features, that is 25,600 tokens per batch flowing through every layer. Doubling your feature count has exactly the same memory effect as doubling your batch size.

#### Batch size and sequence length

Both scale activation memory linearly. Halving either one halves this line item.

Going from a batch size of 256 examples to 128 examples takes activations from roughly 43 GB to roughly 21.5 GB (calculated). The other three line items — weights, gradients, optimizer states — do not move at all. They stay at 16.1 GB combined.

Sequence length behaves the same way in the formula above, with one caveat. If your implementation stores the full attention probability matrix, memory grows with the square of sequence length rather than linearly, and long sequences get expensive fast. Memory-efficient attention implementations avoid storing that matrix. If long sequences are a requirement rather than a preference, confirming you have memory-efficient attention is the first thing to check.

#### Activation checkpointing

Activation checkpointing is the highest-leverage decision on this page.

The default behavior is to store every intermediate tensor from the forward pass, because the backward pass needs them to compute gradients. Activation checkpointing stores only the input to each layer and discards the interior. When the backward pass reaches a layer, the framework recomputes that layer's interior from the stored input, uses it, and discards it again.

So instead of holding every layer's full working set at once, you hold:

(layers × batch size × sequence length × hidden dimension × 2 bytes)   ← the saved layer inputs

\+ one layer's full working set                                         ← the recomputation peak

Worked configuration: 2.1 GB of saved layer inputs plus about 2.1 GB for the single layer being recomputed ≈ **4.2 GB** (calculated). The measured value on an A100 80GB for this configuration was **4.1 GB**.

That is roughly 43 GB down to 4 GB — a tenfold reduction on the largest line item in the budget.

The cost is training time. You are running part of the forward pass twice. In the benchmark run behind this article, per-step time went from 1,034 ms to 1,363 ms, a **32% increase** (measured). That is a real cost, but it is predictable, and it is almost always a better trade than not being able to train at all.

---

## Putting it together

Same worked configuration: 1.01 billion parameters, 20 layers, hidden dimension 2,048, sequence length 100 tokens, batch size 256 examples, BF16 storage precision, Adam with FP32 master weights.

| Line item | Without activation checkpointing | With activation checkpointing |
| :---- | :---- | :---- |
| Weights | 2.0 GB | 2.0 GB |
| Gradients | 2.0 GB | 2.0 GB |
| Optimizer states | 12.1 GB | 12.1 GB |
| Activations | \~43 GB | \~4.1 GB |
| **Total allocated** | **\~59 GB** | **\~20 GB** |
| Allocator overhead and CUDA context | \~1.5 GB | \~5 GB |
| **Total to size against** | **\~61 GB** | **\~25 GB** |

The practical read: without activation checkpointing, this job needs an 80 GB card and has no room to grow. With it, the same job fits on a 40 GB card with headroom to raise the batch size. One configuration flag moved this training run across a hardware tier.

That is the pattern worth internalizing. The static line items are set by decisions you have probably already made — model size, optimizer, precision. The dynamic line item is set by decisions you can still change, and it is usually the one that determines what hardware you need.

---

## What these numbers do not cover

Three honest caveats, because sizing from a number you do not understand is worse than not having it.

**Allocated versus reserved.** The formulas above predict *allocated* memory — bytes held by live tensors. What the driver actually takes from the card is *reserved* memory, which includes cached free blocks and fragmentation. Size your GPU against reserved. The gap was about 1.5 GB in the run without activation checkpointing and about 5 GB with it, since checkpointing frees and reallocates constantly and fragments the pool.

**Ragged batches.** These numbers assume fixed-shape examples. Real batches have variable lengths, and peak memory is driven by the longest sequence in the batch, not the average. If you use dynamic padding, you will run hotter than these predictions.

**Data pipeline memory.** The four line items account for the model. They do not account for your data sitting in GPU memory. GPU-resident pipelines — RAPIDS and cuDF in particular — can hold significant VRAM outside the model, and for smaller models that pipeline can be the dominant consumer. Budget for it separately.

**A note on measurement.** Figures labeled measured come from a training benchmark on a single A100 80GB. Total peak memory was measured directly; the split between static line items and activations is inferred by subtracting the calculated static terms, since the GPU reports one number for the whole process rather than a per-category breakdown.

The companion spreadsheet turns this math into a sizing tool — plug in your model size, batch size, and precision, and it tells you whether the job fits on the GPU you have.  
