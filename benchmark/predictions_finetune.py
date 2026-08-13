"""Per-tensor memory formulas for parameter-efficient fine-tuning (LoRA on a frozen base).

Parallels predictions.py -- named terms, no fudge factors, CPU-testable. Every function
here is one independently-defensible term explainable in one sentence to an engineer
reading the article. The formulas differ from Article 1's pretraining set in five specific
places, each named as its own function or gloss:

  1. Weights split into two line items -- frozen base + trainable adapters -- because
     "n_params" is no longer a single scalar.
  2. Gradients cover only the trainable count (adapter params). Frozen params have no
     gradient.
  3. Optimizer state (Adam or Adam8bit) covers only the trainable count. Same 8-bytes-
     per-trainable-param arithmetic as Article 1, applied to <1% of the parameter count.
  4. Autocast weight cache becomes a residual: the base is loaded at bf16 storage directly,
     so there is nothing to cast for the base. Only the fp32 adapter weights are cast on
     the fly. Under Article 1 this term was ~n_params * 2 bytes (~2 GB at 1B params); here
     it drops to ~n_adapter * 2 bytes (megabytes at rank 8).
  5. Activations rebuilt for a Llama-family decoder-only layer:
       - GQA: K and V projections are (num_kv_heads/num_heads) as wide as Q.
       - SwiGLU MLP: three linears (gate, up, down), and the multiply-then-down_proj chain
         saves four intermediate B*S*intermediate_size tensors, not two.
       - Dropout: TinyLlama's attention_dropout/hidden_dropout default to 0, so the mask
         term vanishes. The formula accepts a dropout_p parameter so a base with nonzero
         dropout re-adds the term without editing the formula.
       - Adapter placement: activations are paid only over the layers that need backward
         reach. If adapters sit only on the upper N layers, layers below the shallowest
         adapter contribute zero to saved activations (their forward has no upstream
         gradient consumer once the input to that block has requires_grad=False).

The three overhead constants (CUDA_CONTEXT_OVERHEAD_GB, FRAGMENTATION_GB,
CHECKPOINTING_FRAGMENTATION_GB) are initial values copied verbatim from predictions.py.
They may need re-fitting against the A100 80GB fine-tune sweep -- that's a second-pass
adjustment against measured data, not first-pass tuning to make numbers match.
"""

GB = 1e9

# Precision -> bytes/param mapping. int4 is 0.5 (two params packed into one byte); it's
# a float here because the base-storage line item multiplies by it. int8/int4 rows appear
# in the sweep only when bitsandbytes is importable (see benchmark_finetune.py's parent-
# startup gate).
BYTES_PER_PARAM_BY_PRECISION = {"fp32": 4, "bf16": 2, "int8": 1, "int4": 0.5}
ADAPTER_STORAGE_BYTES = 4  # LoRA adapters stay fp32 so their small updates aren't lost


def predict_frozen_weights_bytes(n_base, base_storage_precision):
    """Frozen base weights: parameter count times storage bytes, unchanged whether the
    weight is trainable or not."""
    return int(n_base * BYTES_PER_PARAM_BY_PRECISION[base_storage_precision])


def predict_adapter_weights_bytes(n_adapter):
    """LoRA adapters stored in fp32 by default so their small updates aren't rounded
    away by low-precision storage."""
    return n_adapter * ADAPTER_STORAGE_BYTES


def predict_adapter_gradients_bytes(n_adapter):
    """One gradient per trainable parameter, in fp32; frozen base params have no gradient
    at all -- backprop still flows through them, but nothing gets stored."""
    return n_adapter * ADAPTER_STORAGE_BYTES


def predict_adapter_optimizer_bytes(n_adapter, optimizer):
    """Adam holds momentum + variance (8 bytes) per trainable param; adam8bit stores each
    in int8 (2 bytes). Same arithmetic as Article 1, applied to <1% of the parameter count."""
    if optimizer == "adam8bit":
        return n_adapter * 2
    return n_adapter * 8


def predict_autocast_weight_cache_bytes(n_base, n_adapter, precision, base_storage_precision):
    """Under bf16 autocast PyTorch caches a bf16 copy of each fp32 weight it casts on the
    fly. When the base is loaded at bf16 storage directly there is nothing to cast for the
    base, so the term collapses to the adapter-only residual: ~n_adapter * 2 bytes
    (megabytes at rank 8) instead of Article 1's ~n_base * 2 bytes (~2 GB at 1B params).
    Only if the base is loaded at fp32 does the full Article 1 cache reappear."""
    if precision != "amp_bf16":
        return 0
    if base_storage_precision == "fp32":
        return (n_base + n_adapter) * 2
    return n_adapter * 2


def _activation_dtype_bytes(precision):
    return 2 if precision == "amp_bf16" else 4


def _per_layer_working_set_bytes(batch, seq_len, d_model, num_heads, num_kv_heads,
                                 ff_intermediate, activation_bytes, dropout_p):
    """One Llama-style decoder layer's saved intermediates, without checkpointing.

    Terms (each explainable in one sentence):

      Attention block saves 5 same-shape B*S*d bf16 tensors -- the pre-norm residual,
      the post-RMSNorm input to Q/K/V, Q, the attention output, and the input to o_proj
      -- plus 2 GQA-narrowed B*S*(d*num_kv_heads/num_heads) tensors for K and V. The full
      seq x seq attention probability matrix is deliberately zeroed here: transformers
      dispatches SDPA/flash by default, so it is never materialized. If a base is loaded
      that forces eager attention, add back batch*num_heads*seq_len^2*activation_bytes.

      SwiGLU MLP saves 1 pre-norm B*S*d, 1 post-norm B*S*d that feeds gate_proj and
      up_proj, and 4 same-shape B*S*intermediate_size tensors (gate_proj output for the
      SiLU backward, SiLU output and up_proj output as multiply operands, and the multiply
      product as input to down_proj). That is one more B*S*intermediate tensor than
      Article 1's two-linear MLP; SwiGLU's price for the extra gate branch.

      Dropout masks are 1 byte/element and applied at 3 points per layer in a Llama block
      when hidden_dropout > 0 (post-attention, inside MLP, and residual). TinyLlama's
      config sets attention_dropout and hidden_dropout to 0, so this term is zero for the
      baseline; the formula accepts dropout_p so a base with nonzero dropout re-adds it.
    """
    bsd = batch * seq_len * d_model * activation_bytes
    bsd_kv = batch * seq_len * (d_model * num_kv_heads / num_heads) * activation_bytes
    bsi = batch * seq_len * ff_intermediate * activation_bytes

    attn_full_width = 5 * bsd
    attn_kv = 2 * bsd_kv
    mlp_prenorm = 2 * bsd
    mlp_hidden = 4 * bsi

    if dropout_p > 0:
        dropout_masks = 2 * (batch * seq_len * d_model) + (batch * seq_len * ff_intermediate)
    else:
        dropout_masks = 0

    return {
        "attn_full_width": attn_full_width,
        "attn_kv_gqa": attn_kv,
        "mlp_prenorm": mlp_prenorm,
        "mlp_hidden_swiglu": mlp_hidden,
        "dropout_masks": dropout_masks,
    }


def _once_per_model_bytes(batch, seq_len, d_model, vocab, precision, activation_bytes):
    """Terms that exist once per forward pass, not once per layer."""
    embedding_output = batch * seq_len * d_model * activation_bytes
    final_logits = batch * seq_len * vocab * activation_bytes
    # cross_entropy is on autocast's forced-fp32 list for numerical stability, so under
    # amp_bf16 it upcasts the logits internally, materializing a second, fp32-sized copy.
    # fp32 mode has no autocast, so no separate upcast copy.
    logits_fp32_upcast = batch * seq_len * vocab * 4 if precision == "amp_bf16" else 0
    return {
        "embedding_output": embedding_output,
        "final_logits": final_logits,
        "logits_fp32_upcast": logits_fp32_upcast,
    }


def resolve_layers_needing_backward(n_layers_total, adapter_layers):
    """Adapter placement -> how many layers still need saved activations.

    'all' -> every layer has adapters, so every layer needs backward reach. 'upper-N' ->
    adapters on the top N layers only; the layers below the shallowest adapter have no
    trainable param above them via that path either, but crucially their own outputs feed
    the shallowest adapter -- so their outputs need requires_grad=True and their forwards
    still save. Only layers strictly BELOW the shallowest adapter are safe to skip.

    Wait: reread. Layers 0..(L-N-1) are frozen and their input has requires_grad=False
    (embedding is frozen too under standard LoRA-on-Llama targeting projection linears
    only). Layer (L-N-1)'s output is the input to layer (L-N), which is the shallowest
    adapter. Layer (L-N-1) has no trainable parameters and its input is requires_grad=
    False, so its output is also requires_grad=False -- autograd saves nothing for it.
    The shallowest adapter layer (L-N) is the first that saves. Every layer from (L-N)
    upward saves.

    So the answer is exactly N: the count of layers from the shallowest adapter to the
    top, inclusive.
    """
    if adapter_layers == "all":
        return n_layers_total
    if adapter_layers.startswith("upper-"):
        n = int(adapter_layers.split("-", 1)[1])
        return min(n, n_layers_total)
    raise ValueError(f"unknown adapter_layers: {adapter_layers!r}")


def predict_activation_bytes(batch, seq_len, d_model, n_layers_needing_backward,
                             num_heads, num_kv_heads, ff_intermediate, vocab,
                             precision, checkpointing, dropout_p=0.0):
    """Named per-tensor activation accounting for a Llama-family decoder-only layer.

    Two shapes: non-checkpointed sums the full working set across every layer that needs
    backward reach; checkpointed keeps only the fp32 layer boundaries plus one segment's
    recompute working set, matching Article 1's story that the tensor passed *between*
    layers is fp32 even under bf16 autocast because RMSNorm and residual adds run fp32.
    """
    b = _activation_dtype_bytes(precision)
    per_layer = _per_layer_working_set_bytes(
        batch, seq_len, d_model, num_heads, num_kv_heads, ff_intermediate, b, dropout_p
    )
    once = _once_per_model_bytes(batch, seq_len, d_model, vocab, precision, b)
    per_layer_total = sum(per_layer.values())
    once_total = sum(once.values())

    if not checkpointing:
        return per_layer_total * n_layers_needing_backward + once_total

    # Checkpointed: (a) the tensor passed *between* layers is fp32 (RMSNorm output +
    # residual add both run fp32 under autocast); (b) the peak lands early in backward
    # while the last segment is being recomputed. All saved boundaries are live, plus
    # 3 more same-shape fp32 tensors (the recomputed segment's re-saved norm tensors
    # and the incoming gradient w.r.t. the segment output), plus the recomputed
    # segment's bf16 working set (attention + MLP), plus one dropout mask if applicable,
    # plus the still-live logits.
    residual_boundary_fp32 = batch * seq_len * d_model * 4
    saved_boundaries = (n_layers_needing_backward + 3) * residual_boundary_fp32

    bsd = batch * seq_len * d_model * b
    bsd_kv = batch * seq_len * (d_model * num_kv_heads / num_heads) * b
    bsi = batch * seq_len * ff_intermediate * b
    # One segment's forward transients under recomputation: attention block (5 bsd + 2 bsd_kv)
    # plus SwiGLU MLP hidden set (4 bsi). The prenorm and boundary terms are already
    # accounted for in saved_boundaries.
    recompute_working_set = 5 * bsd + 2 * bsd_kv + 4 * bsi

    dropout_mask = batch * seq_len * d_model if dropout_p > 0 else 0
    live_logits = batch * seq_len * vocab * b
    return int(saved_boundaries + recompute_working_set + dropout_mask + live_logits)


# --- Overhead terms; see predictions.py for the full narrative. Initial values copied
# verbatim; refit only after the A100 80GB fine-tune sweep produces evidence they miss.
CUDA_CONTEXT_OVERHEAD_GB = 0.6
FRAGMENTATION_GB = 1.2
CHECKPOINTING_FRAGMENTATION_GB = 5.2

# 79.18 GiB usable on both A100 80GB and H100 80GB (essentially identical usable capacity).
GPU_CAPACITY_GB = 79.18 * (1024 ** 3) / GB


def predict_line_items_finetune(
    n_base, n_adapter, base_storage_precision, precision, optimizer,
    batch, seq_len, d_model, n_layers_total, num_heads, num_kv_heads,
    ff_intermediate, vocab, adapter_layers, checkpointing, dropout_p=0.0,
):
    """Bundled prediction: every line item plus allocated_total, reserved_total, and
    the OOM call. Returns a dict with the fine-tune-specific keys the CSV writes.
    """
    n_layers_needing_backward = resolve_layers_needing_backward(n_layers_total, adapter_layers)

    frozen_weights = predict_frozen_weights_bytes(n_base, base_storage_precision)
    adapter_weights = predict_adapter_weights_bytes(n_adapter)
    gradients = predict_adapter_gradients_bytes(n_adapter)
    optimizer_bytes = predict_adapter_optimizer_bytes(n_adapter, optimizer)
    autocast_cache = predict_autocast_weight_cache_bytes(
        n_base, n_adapter, precision, base_storage_precision
    )
    activations = predict_activation_bytes(
        batch, seq_len, d_model, n_layers_needing_backward,
        num_heads, num_kv_heads, ff_intermediate, vocab,
        precision, checkpointing, dropout_p,
    )

    static_no_activations = frozen_weights + adapter_weights + gradients + optimizer_bytes

    if not checkpointing:
        allocated_total = static_no_activations + autocast_cache + activations
    else:
        # Same phase-peak logic as predictions.py's checkpointed case, adapted:
        # (1) Backward-recompute peak = frozen + adapters + optimizer + activations.
        #     Weight gradients have not accumulated yet (adapter grads only accumulate
        #     after backward completes for the segment), and autocast cache was freed
        #     when forward exited autocast.
        backward_peak = frozen_weights + adapter_weights + optimizer_bytes + activations
        # (2) Optimizer-step peak = frozen + adapters + gradients + optimizer + Adam
        #     temporaries. Foreach-Adam materializes one param-sized set of temporaries
        #     per group; here the trainable group is small (adapters only), so this
        #     floor is dominated by the frozen-base bytes rather than by 20-bytes/param
        #     as in Article 1.
        foreach_adam_temps = gradients if optimizer == "adam" else 0
        optimizer_step_peak = static_no_activations + foreach_adam_temps
        # (3) Forward peak = frozen + adapters + optimizer + autocast cache + saved fp32
        #     boundaries + one segment's forward transient + logits terms.
        b = _activation_dtype_bytes(precision)
        bsi = batch * seq_len * ff_intermediate * b
        bsd = batch * seq_len * d_model * b
        bsd_kv = batch * seq_len * (d_model * num_kv_heads / num_heads) * b
        once = _once_per_model_bytes(batch, seq_len, d_model, vocab, precision, b)
        segment_forward_transient = 4 * bsi + 5 * bsd + 2 * bsd_kv
        forward_peak = (
            frozen_weights + adapter_weights + optimizer_bytes + autocast_cache
            + n_layers_needing_backward * (batch * seq_len * d_model * 4)
            + segment_forward_transient
            + once["final_logits"] + once["logits_fp32_upcast"]
        )
        allocated_total = max(backward_peak, optimizer_step_peak, forward_peak)

    fragmentation = (CHECKPOINTING_FRAGMENTATION_GB if checkpointing else FRAGMENTATION_GB) * GB
    reserved_total = allocated_total + CUDA_CONTEXT_OVERHEAD_GB * GB + fragmentation

    return {
        "frozen_weights": frozen_weights,
        "adapter_weights": adapter_weights,
        "gradients": gradients,
        "optimizer": optimizer_bytes,
        "autocast_weight_cache": autocast_cache,
        "activations": activations,
        "allocated_total": allocated_total,
        "reserved_total": reserved_total,
        "predicted_oom": reserved_total > GPU_CAPACITY_GB * GB,
        "predicted_trainable_param_count": n_adapter,
        "predicted_frozen_param_count": n_base,
        "n_layers_needing_backward": n_layers_needing_backward,
    }
