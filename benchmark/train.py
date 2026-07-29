"""GPU training loop. Used only by benchmark.py's subprocess child, on CUDA.

Synthetic data only: random integer tokens, random targets. Just enough fidelity for
correct shapes and a loss that backpropagates through every layer -- the content is
deliberately meaningless, since we only care about what gets allocated, not what gets
learned.

Precision: "fp32" runs everything (params, grads, optimizer state, compute) in plain
fp32. "amp_bf16" adds torch.autocast around the forward pass for bf16 compute, but does
NOT cast parameter storage -- params/grads/optimizer state stay fp32 in both modes. See
predictions.py's module docstring for why that distinction matters for memory.
"""
import time

import torch
import torch.nn as nn

from memory_accounting import measured_grad_bytes, measured_optimizer_bytes, measured_param_bytes
from model import TinyTransformer, count_parameters

WARMUP_STEPS = 2
GB = 1e9


def run_training(config):
    """Runs config.steps optimizer steps of a TinyTransformer on synthetic data, on CUDA,
    plus one extra, untimed profiling step used to isolate forward-pass peak memory
    without disturbing the run's true overall peak (see the profiling-step block below).

    Returns a dict with parameter count, final loss, per-step timings for the
    steady-state steps (after warmup), a grad-norm sanity check, the measured peak
    allocated/reserved bytes for the whole run, and the direct measurement columns used
    to validate the prediction formulas (measured_param_bytes, measured_grad_bytes,
    measured_optimizer_bytes, alloc_after_model_gb, alloc_after_optimizer_gb,
    peak_forward_gb).
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    model = TinyTransformer(
        vocab_size=config.vocab,
        d_model=config.d,
        n_layers=config.layers,
        n_heads=config.heads,
        ff_mult=config.ff_mult,
        seq_len=config.seq_len,
        use_checkpointing=config.checkpointing,
    ).to(device)
    model.train()

    n_params = count_parameters(model)
    measured_param_bytes_val = measured_param_bytes(model)
    alloc_after_model_gb = torch.cuda.memory_allocated(device) / GB

    if config.optimizer == "adam8bit":
        import bitsandbytes as bnb  # optional dependency; only imported when requested

        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_fn = nn.CrossEntropyLoss()
    use_autocast = config.precision == "amp_bf16"

    step_times_ms = []
    final_loss = None
    measured_grad_bytes_val = None
    measured_optimizer_bytes_val = None
    alloc_after_optimizer_gb = None
    peak_forward_bytes = None
    peak_before_profile_bytes = None
    reserved_before_profile_bytes = None

    total_steps = config.steps + 1  # +1 dedicated, untimed profiling step (see below)
    for step in range(1, total_steps + 1):
        tokens = torch.randint(0, config.vocab, (config.batch, config.seq_len), device=device)
        targets = torch.randint(0, config.vocab, (config.batch, config.seq_len), device=device)

        # The forward-pass-only peak can't be read directly -- torch only exposes a
        # single running peak counter, and resetting it mid-run would erase the history
        # of whatever peak happened earlier. So on the last (profiling) step, we snapshot
        # the true peak-so-far first, reset to isolate this step's forward pass, then
        # recombine with max() after the step completes -- the overall peak this function
        # returns is exact, not approximated.
        profiling_step = step == total_steps
        if profiling_step:
            torch.cuda.synchronize()
            peak_before_profile_bytes = torch.cuda.max_memory_allocated(device)
            reserved_before_profile_bytes = torch.cuda.max_memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)

        torch.cuda.synchronize()  # otherwise we're timing kernel launches, not execution
        t0 = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(tokens)
                loss = loss_fn(logits.view(-1, config.vocab), targets.view(-1))
        else:
            logits = model(tokens)
            loss = loss_fn(logits.view(-1, config.vocab), targets.view(-1))

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
        if WARMUP_STEPS < step <= config.steps:  # steps 1-2 are warmup; the last step is profiling-only
            step_times_ms.append((t1 - t0) * 1000)

    torch.cuda.synchronize()
    max_allocated_bytes = max(peak_before_profile_bytes, torch.cuda.max_memory_allocated(device))
    max_reserved_bytes = max(reserved_before_profile_bytes, torch.cuda.max_memory_reserved(device))

    grad_norm = sum(p.grad.detach().abs().sum().item() for p in model.parameters() if p.grad is not None)

    return {
        "n_params": n_params,
        "final_loss": final_loss,
        "step_times_ms": step_times_ms,
        "grad_norm": grad_norm,
        "max_allocated_bytes": max_allocated_bytes,
        "max_reserved_bytes": max_reserved_bytes,
        "measured_param_bytes": measured_param_bytes_val,
        "measured_grad_bytes": measured_grad_bytes_val,
        "measured_optimizer_bytes": measured_optimizer_bytes_val,
        "alloc_after_model_gb": alloc_after_model_gb,
        "alloc_after_optimizer_gb": alloc_after_optimizer_gb,
        "peak_forward_gb": peak_forward_bytes / GB if peak_forward_bytes is not None else None,
    }
