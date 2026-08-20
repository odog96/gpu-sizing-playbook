"""Direct measurement helpers specific to fine-tuning: trainable vs frozen split.

The base helpers (measured_param_bytes, measured_grad_bytes, measured_optimizer_bytes)
in memory_accounting.py already work as-is on a PeftModel -- they walk parameters or
optimizer state without caring about requires_grad. This module adds the two wrappers
that filter by requires_grad, so the fine-tune benchmark can measure the frozen and
trainable pools separately and check the split against the analytical LoRA arithmetic.

Kept in a sibling file rather than added to memory_accounting.py, per the "new sibling
scripts, not new flags" convention (docs/benchmark-manual.md § 3).
"""


def measured_trainable_param_bytes(model):
    """Sum of bytes over parameters where requires_grad=True. For a LoRA-wrapped model,
    this is the adapter total: the LoRA A and B factors and, if bias='all' or 'lora_only',
    any bias parameters PEFT unfroze. With bias='none' (the default in this benchmark) it
    is purely adapter weights."""
    return sum(
        p.element_size() * p.nelement()
        for p in model.parameters()
        if p.requires_grad
    )


def measured_frozen_param_bytes(model):
    """Sum of bytes over parameters where requires_grad=False. For a LoRA-wrapped model
    this is the frozen base weights. The two totals sum to measured_param_bytes."""
    return sum(
        p.element_size() * p.nelement()
        for p in model.parameters()
        if not p.requires_grad
    )
