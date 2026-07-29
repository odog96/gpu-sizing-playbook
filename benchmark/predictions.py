"""Per-tensor memory formulas, corrected against direct GPU measurement.

Units: 1 GB = 1e9 bytes (decimal), matching the article's arithmetic. Every function
here is one named, independently-defensible term -- no fudge factors, no curve fitting.
Two constants (CUDA_CONTEXT_OVERHEAD_GB, the fragmentation allowances) are not derived
from first principles; they're documented as such below, sourced from the one real sweep
run available, and kept in their own named terms rather than folded into anything else.

Precision reality check (see train.py): this codebase never actually stores parameters,
gradients, or optimizer state in bf16. `.to(device)` never casts dtype, and
torch.autocast only changes the dtype used *during* the forward matmuls -- it doesn't
touch parameter storage. So "precision" only controls two things: (1) whether autocast
wraps the forward pass in bf16 compute (precision="amp_bf16") or everything runs plain
fp32 (precision="fp32"), and (2) the resulting activation dtype. Weights, gradients, and
optimizer state are fp32 in both modes -- that's why predict_weights_bytes and
predict_optimizer_bytes below don't take a precision argument.
"""

GB = 1e9

PARAM_STORAGE_BYTES = 4  # fp32, always -- see module docstring


def predict_weights_bytes(n_params):
    """Parameters are always stored fp32 in this codebase; autocast never casts storage."""
    return n_params * PARAM_STORAGE_BYTES


def predict_gradients_bytes(n_params):
    """Gradients match their parameter's dtype (fp32), so this is the same formula as weights."""
    return n_params * PARAM_STORAGE_BYTES


def predict_optimizer_bytes(n_params, optimizer):
    """Adam: momentum + variance, both fp32 -> 8 bytes/param. 8-bit Adam: the same two
    moments, both int8 -> 2 bytes/param. (The old formula assumed a bf16-master-weight
    mixed-precision layout -- 12 and 3 bytes/param -- that this codebase doesn't
    implement; validated against the measured 6.04 GB delta between adam and adam8bit in
    the baseline sweep, which this formula predicts within 0.5%.)"""
    if optimizer == "adam8bit":
        return n_params * 2
    return n_params * 8


def predict_autocast_weight_cache_bytes(n_params, precision):
    """Under bf16 autocast, PyTorch casts each fp32 weight to bf16 for the matmul and
    caches that bf16 copy for reuse across the layer's ops during the forward pass --
    a real, separate allocation the old formula silently folded into "activations"."""
    if precision == "amp_bf16":
        return n_params * 2
    return 0


def _activation_dtype_bytes(precision):
    return 2 if precision == "amp_bf16" else 4


def _per_layer_working_set_bytes(batch, seq_len, d_model, ff_mult, activation_bytes):
    """One transformer encoder layer's tensors saved for backward, without checkpointing.

    nn.TransformerEncoderLayer (norm_first=True) saves 7 same-shaped (B,S,d) tensors for
    the attention sub-block: the pre-norm input, the post-norm input to the QKV
    projection, the split Q/K/V (3 tensors), the raw attention output, and the input to
    the output projection.

    The attention probability matrix (batch*heads*seq_len^2) is deliberately NOT counted:
    this model calls self_attn with need_weights=False, which PyTorch's MultiheadAttention
    dispatches to scaled_dot_product_attention (flash/memory-efficient kernel on CUDA),
    so the full seq_len x seq_len matrix is never materialized. Confirmed against this
    sweep's own OOM data: the combined config (batch=4096, seq_len=2048) topped out at 68
    GB total, not the ~550 GB a materialized batch*heads*seq_len^2 matrix would require.
    """
    bsd = batch * seq_len * d_model * activation_bytes
    bsfd = batch * seq_len * (ff_mult * d_model) * activation_bytes

    attn_saved = 7 * bsd
    mlp_prenorm = 2 * bsd  # FFN's pre-norm input + the post-norm input to fc1
    mlp_hidden = 2 * bsfd  # fc1 output (pre-activation) + activation-fn output
    # dropout=0.1 is nn.TransformerEncoderLayer's default and this codebase never
    # overrides it. Three dropout calls need their masks saved for backward: dropout1
    # after attention (B*S*d), dropout2 after the FFN (B*S*d), and one more inside the
    # FFN between the activation function and fc2 -- which is B*S*(ff_mult*d) shaped,
    # not B*S*d, since it's applied to fc1's output. 1 byte/element (mask, not payload).
    dropout_masks = 2 * (batch * seq_len * d_model) + (batch * seq_len * ff_mult * d_model)

    return {
        "attn_saved": attn_saved,
        "mlp_prenorm": mlp_prenorm,
        "mlp_hidden": mlp_hidden,
        "dropout_masks": dropout_masks,
    }


def _once_per_model_bytes(batch, seq_len, d_model, vocab, precision, activation_bytes):
    """Terms that exist once per forward pass, not once per layer."""
    embedding_output = batch * seq_len * d_model * activation_bytes
    final_logits = batch * seq_len * vocab * activation_bytes
    # Confirmed empirically (autocast keeps a Linear's output in bf16, but
    # cross_entropy's own output is fp32): cross_entropy is on autocast's forced-fp32
    # list for numerical stability, so it upcasts the logits internally, materializing a
    # second, fp32-sized copy. fp32 mode has no autocast, so no separate upcast copy.
    logits_fp32_upcast = batch * seq_len * vocab * 4 if precision == "amp_bf16" else 0
    return {
        "embedding_output": embedding_output,
        "final_logits": final_logits,
        "logits_fp32_upcast": logits_fp32_upcast,
    }


def predict_activation_bytes(batch, seq_len, d_model, n_layers, ff_mult, vocab, precision, checkpointing):
    """Named per-tensor activation accounting.

    Replaces two bugs in the old formula: (a) the non-checkpointed case used what is
    actually the *checkpointed* formula (one B*S*d tensor per layer), undercounting the
    real ~15-20 distinct tensors a non-checkpointed layer saves by roughly 20x; (b) the
    checkpointed case then divided that already-wrong floor by n_layers, undercounting a
    further ~40x (0.105 GB predicted vs. ~4.1 GB measured at baseline scale).
    """
    b = _activation_dtype_bytes(precision)
    per_layer = _per_layer_working_set_bytes(batch, seq_len, d_model, ff_mult, b)
    once = _once_per_model_bytes(batch, seq_len, d_model, vocab, precision, b)
    per_layer_total = sum(per_layer.values())
    once_total = sum(once.values())

    if not checkpointing:
        return per_layer_total * n_layers + once_total

    # Checkpointing saves only each layer's input (the recomputation boundary) instead of
    # every intermediate tensor, so the steady-state resident cost is n_layers copies of
    # one B*S*d tensor. But during backward, one layer at a time is recomputed forward
    # (with grad tracking on) and then immediately backpropagated through -- while that's
    # happening, that one layer transiently holds its *full* non-checkpointed working set
    # on top of everything else already resident. That transient is a real, if brief,
    # peak, not an approximation error.
    bsd = batch * seq_len * d_model * b
    saved_boundaries = n_layers * bsd
    recompute_peak = per_layer_total  # one layer's full working set, not n_layers worth
    return saved_boundaries + recompute_peak + once_total


# --- Terms that live outside PyTorch's own tensor accounting ---
#
# torch.cuda.memory_allocated()/memory_reserved() only ever report memory the caching
# allocator handed out to tensors. CUDA context init (driver + cuDNN/cuBLAS handle and
# workspace setup) is allocated directly through the CUDA driver and is invisible to
# those calls -- there is no way to measure it from inside the training subprocess using
# torch's own memory API. The figure below is a standard documented order-of-magnitude
# for a single-process CUDA context with cuDNN/cuBLAS in use; it is NOT fit to this run's
# data, and calibrating it precisely would require an nvidia-smi reading alongside a
# results.csv row, which this benchmark doesn't currently capture.
CUDA_CONTEXT_OVERHEAD_GB = 0.6

# The gap between max_reserved_gb and max_allocated_gb in the one real sweep run
# available: ~1.1-1.24 GB for every non-checkpointed row, 5.21 GB for the one
# checkpointing row. Checkpointing's repeated alloc/free of same-shaped recomputation
# buffers apparently fragments the caching allocator's pool more than steady-state
# training does. The checkpointing figure is a single data point -- treat it as
# provisional until more checkpointed configs exist in results.csv.
FRAGMENTATION_GB = 1.2
CHECKPOINTING_FRAGMENTATION_GB = 5.2

# nvidia-smi/torch reported ~79.25 GiB usable on this fleet's A100 80GB (read directly
# off this sweep's own OOM error messages), converted to the decimal-GB (1e9-byte) units
# used everywhere else in this file.
GPU_CAPACITY_GB = 79.25 * (1024 ** 3) / GB


def predict_line_items(n_params, precision, optimizer, batch, seq_len, d_model, n_layers, ff_mult, vocab, checkpointing):
    weights = predict_weights_bytes(n_params)
    gradients = predict_gradients_bytes(n_params)
    optimizer_bytes = predict_optimizer_bytes(n_params, optimizer)
    autocast_cache = predict_autocast_weight_cache_bytes(n_params, precision)
    activations = predict_activation_bytes(batch, seq_len, d_model, n_layers, ff_mult, vocab, precision, checkpointing)

    allocated_total = weights + gradients + optimizer_bytes + autocast_cache + activations
    fragmentation = (CHECKPOINTING_FRAGMENTATION_GB if checkpointing else FRAGMENTATION_GB) * GB
    reserved_total = allocated_total + CUDA_CONTEXT_OVERHEAD_GB * GB + fragmentation

    return {
        "weights": weights,
        "gradients": gradients,
        "optimizer": optimizer_bytes,
        "autocast_weight_cache": autocast_cache,
        "activations": activations,
        "allocated_total": allocated_total,
        "reserved_total": reserved_total,
        "predicted_oom": reserved_total > GPU_CAPACITY_GB * GB,
    }
