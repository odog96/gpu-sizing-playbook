"""GPU training loop for LoRA fine-tuning. Called only by benchmark_finetune.py's child
subprocess, on CUDA.

Synthetic data only: random integer input_ids, matching labels (LM-style next-token
target). Just enough fidelity for correct shapes and a loss that backpropagates through
every layer touched by an adapter -- the content is deliberately meaningless, since we
only care about what gets allocated, not what gets learned.

Precision: identical convention to train.py -- "fp32" runs plain fp32, "amp_bf16" wraps
the forward pass in torch.autocast(bfloat16). Base storage precision is set separately
via base_storage_precision on the Config: "bf16" loads the base at bf16 directly (so no
casting for the base under autocast); "fp32" loads at fp32; "int8"/"int4" load quantized
via bitsandbytes.
"""
import time

import torch

from memory_accounting import measured_grad_bytes, measured_optimizer_bytes, measured_param_bytes
from memory_accounting_finetune import measured_frozen_param_bytes, measured_trainable_param_bytes
from model_finetune import build_finetune_model, count_frozen_params, count_trainable_params

WARMUP_STEPS = 2
GB = 1e9


def run_training_finetune(config):
    """Loads TinyLlama+LoRA, runs config.steps + 1 optimizer steps, returns measurements
    parallel to train.py's return dict plus a few fine-tune-specific keys.

    The +1 profiling step at the end is used to isolate peak_forward_gb without
    disturbing the true overall peak -- same trick as train.py.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    model, base_arch = build_finetune_model(config)
    model.to(device)
    model.train()

    n_trainable = count_trainable_params(model)
    n_frozen = count_frozen_params(model)
    measured_param_bytes_val = measured_param_bytes(model)
    measured_trainable_bytes_val = measured_trainable_param_bytes(model)
    measured_frozen_bytes_val = measured_frozen_param_bytes(model)
    alloc_after_model_gb = torch.cuda.memory_allocated(device) / GB

    # Adapter parameters only -- filter model.parameters() by requires_grad so the
    # optimizer holds state for adapters, not the frozen base. This is what makes
    # optimizer memory scale with n_adapter, not n_total.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == "adam8bit":
        import bitsandbytes as bnb  # optional; parent-startup gate keeps this reachable

        optimizer = bnb.optim.Adam8bit(trainable_params, lr=1e-4)
    else:
        optimizer = torch.optim.Adam(trainable_params, lr=1e-4)

    use_autocast = config.precision == "amp_bf16"
    vocab = base_arch["vocab_size"]

    step_times_ms = []
    final_loss = None
    measured_grad_bytes_val = None
    measured_optimizer_bytes_val = None
    alloc_after_optimizer_gb = None
    peak_forward_bytes = None
    peak_before_profile_bytes = None
    reserved_before_profile_bytes = None

    total_steps = config.steps + 1  # +1 dedicated profiling step -- see train.py
    for step in range(1, total_steps + 1):
        input_ids = torch.randint(0, vocab, (config.batch, config.seq_len), device=device)
        # Language-modeling target: labels = input_ids. HF causal-LM heads internally
        # shift by one; we don't need to do it here. attention_mask defaults to all 1s.
        labels = input_ids.clone()

        profiling_step = step == total_steps
        if profiling_step:
            torch.cuda.synchronize()
            peak_before_profile_bytes = torch.cuda.max_memory_allocated(device)
            reserved_before_profile_bytes = torch.cuda.max_memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss
        else:
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss

        if profiling_step:
            torch.cuda.synchronize()
            peak_forward_bytes = torch.cuda.max_memory_allocated(device)

        loss.backward()

        if step == 1:
            measured_grad_bytes_val = measured_grad_bytes(model)

        optimizer.step()

        if step == 1:
            measured_optimizer_bytes_val = measured_optimizer_bytes(optimizer)
            alloc_after_optimizer_gb = torch.cuda.memory_allocated(device) / GB

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        final_loss = loss.item()
        if WARMUP_STEPS < step <= config.steps:
            step_times_ms.append((t1 - t0) * 1000)

    torch.cuda.synchronize()
    max_allocated_bytes = max(peak_before_profile_bytes, torch.cuda.max_memory_allocated(device))
    max_reserved_bytes = max(reserved_before_profile_bytes, torch.cuda.max_memory_reserved(device))

    return {
        "n_trainable_params": n_trainable,
        "n_frozen_params": n_frozen,
        "n_params": n_trainable + n_frozen,
        "final_loss": final_loss,
        "step_times_ms": step_times_ms,
        "max_allocated_bytes": max_allocated_bytes,
        "max_reserved_bytes": max_reserved_bytes,
        "measured_param_bytes": measured_param_bytes_val,
        "measured_trainable_param_bytes": measured_trainable_bytes_val,
        "measured_frozen_param_bytes": measured_frozen_bytes_val,
        "measured_grad_bytes": measured_grad_bytes_val,
        "measured_optimizer_bytes": measured_optimizer_bytes_val,
        "alloc_after_model_gb": alloc_after_model_gb,
        "alloc_after_optimizer_gb": alloc_after_optimizer_gb,
        "peak_forward_gb": peak_forward_bytes / GB if peak_forward_bytes is not None else None,
        "base_arch": base_arch,
    }
