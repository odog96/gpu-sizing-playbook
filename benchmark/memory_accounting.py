"""Direct measurement of what's actually resident in param/grad/optimizer tensors.

These replace the old approach of inferring the static (weights+gradients+optimizer)
total by trusting the *predicted* formula and subtracting it from measured GPU memory.
Every function here does a real sum over real tensor objects -- element_size() * nelement()
-- so it works identically on CPU or CUDA, which also means it's exercised by the CPU-only
unit tests, not just live GPU runs.
"""
import torch


def measured_param_bytes(model):
    return sum(p.element_size() * p.nelement() for p in model.parameters())


def measured_grad_bytes(model):
    return sum(p.grad.element_size() * p.grad.nelement() for p in model.parameters() if p.grad is not None)


def measured_optimizer_bytes(optimizer):
    """Sums every tensor held in optimizer.state, regardless of optimizer implementation.

    Works for torch.optim.Adam (exp_avg/exp_avg_sq, same dtype as the param) and for
    bitsandbytes' Adam8bit (state1/state2 as int8, plus small per-block quantization
    stats) -- it doesn't assume field names or dtypes, just walks whatever tensors exist.
    """
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.element_size() * value.nelement()
    return total
