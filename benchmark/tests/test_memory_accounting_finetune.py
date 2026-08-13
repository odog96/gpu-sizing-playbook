"""CPU-only tests: trainable/frozen bytes sum to total, and gradient bytes behave right
before/after backward. Uses a hand-rolled nn.Module rather than pulling TinyLlama into
the test suite -- HF model init on CPU is slow, and none of what these tests check is
Llama-specific."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from memory_accounting import measured_grad_bytes, measured_param_bytes
from memory_accounting_finetune import (
    measured_frozen_param_bytes,
    measured_trainable_param_bytes,
)


def _tiny_lora_stand_in():
    """A tiny model with a frozen 'base' and a small trainable 'adapter' -- structurally
    identical to what PEFT produces on a LoRA-wrapped Llama layer: some parameters have
    requires_grad=False, some have requires_grad=True."""
    # 512 base params (fp32 -> 2048 bytes), 40 adapter params (fp32 -> 160 bytes).
    base = nn.Linear(16, 32, bias=False)   # 512 params
    for p in base.parameters():
        p.requires_grad = False
    adapter = nn.Linear(2, 20, bias=False)  # 40 params, trainable by default
    model = nn.Sequential(base, adapter)
    return model, 512 * 4, 40 * 4


class TestTrainableFrozenSplit(unittest.TestCase):
    def test_frozen_plus_trainable_equals_total(self):
        model, base_bytes, adapter_bytes = _tiny_lora_stand_in()
        self.assertEqual(measured_frozen_param_bytes(model), base_bytes)
        self.assertEqual(measured_trainable_param_bytes(model), adapter_bytes)
        self.assertEqual(
            measured_frozen_param_bytes(model) + measured_trainable_param_bytes(model),
            measured_param_bytes(model),
        )

    def test_all_frozen_gives_zero_trainable(self):
        model, base_bytes, _ = _tiny_lora_stand_in()
        for p in model.parameters():
            p.requires_grad = False
        self.assertEqual(measured_trainable_param_bytes(model), 0)
        self.assertEqual(measured_frozen_param_bytes(model), measured_param_bytes(model))

    def test_all_trainable_gives_zero_frozen(self):
        model, _, _ = _tiny_lora_stand_in()
        for p in model.parameters():
            p.requires_grad = True
        self.assertEqual(measured_frozen_param_bytes(model), 0)
        self.assertEqual(measured_trainable_param_bytes(model), measured_param_bytes(model))


class TestGradientBytesUnderFrozenBase(unittest.TestCase):
    def test_zero_grad_bytes_before_backward(self):
        model, _, _ = _tiny_lora_stand_in()
        self.assertEqual(measured_grad_bytes(model), 0)

    def test_only_trainable_params_get_gradients_after_backward(self):
        # A gradient tensor is created only for parameters with requires_grad=True,
        # which is what makes fine-tune gradients scale with n_adapter, not n_base.
        model, _, adapter_bytes = _tiny_lora_stand_in()

        # Forward requires the shapes to actually align. Use two matmuls in sequence:
        # base is 16->32, adapter is 2->20. Route via a fixed projection to compose.
        # Simplest is to do them independently and sum losses.
        x_base = torch.randn(4, 16)
        x_adapter = torch.randn(4, 2)
        base_out = model[0](x_base)
        adapter_out = model[1](x_adapter)
        loss = base_out.sum() + adapter_out.sum()
        loss.backward()

        # measured_grad_bytes walks p.grad only for params that got a grad. Frozen params
        # in this codebase have p.grad is None; only the adapter's 40 params get grads.
        self.assertEqual(measured_grad_bytes(model), adapter_bytes)


if __name__ == "__main__":
    unittest.main()
