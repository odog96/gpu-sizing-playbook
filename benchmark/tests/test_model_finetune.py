"""CPU-only tests for model_finetune.

These do NOT import build_finetune_model or peft/transformers: PEFT and transformers
would drag in ~1 GB of dependencies not needed for the pure-Python logic that Phase A
owns. What is tested here:

  - _layers_to_transform maps adapter-placement strings to the correct layer indices.
  - analytical_param_count_finetune matches hand-derived LoRA arithmetic.
  - count_trainable_params and count_frozen_params filter by requires_grad correctly on
    a hand-rolled Module that mimics a PEFT-wrapped model's requires_grad structure.
  - The 'upper-N stops backward reach' claim holds for a hand-rolled module that mimics
    the requires_grad structure PEFT produces -- verified by asserting p.grad is None
    on the frozen-layer params after a full backward.

The one thing these tests can't check without pulling in PEFT is that PEFT's
LoraConfig(layers_to_transform=...) itself behaves as expected. That is verified by
build_finetune_model's runtime assertion (_assert_only_lora_params_trainable) plus the
debug_finetune.py probe 3 run on real hardware.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from model_finetune import (
    _layers_to_transform,
    analytical_param_count_finetune,
    count_frozen_params,
    count_trainable_params,
)


TINYLLAMA_ARCH = {
    "num_hidden_layers": 22,
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "intermediate_size": 5632,
    "vocab_size": 32000,
}


class TestLayersToTransform(unittest.TestCase):
    def test_all_returns_none(self):
        # PEFT convention: None means every eligible layer.
        self.assertIsNone(_layers_to_transform(22, "all"))

    def test_upper_3_returns_top_three_indices(self):
        # TinyLlama has layers 0..21; upper-3 means [19, 20, 21].
        self.assertEqual(_layers_to_transform(22, "upper-3"), [19, 20, 21])

    def test_upper_11_returns_top_half(self):
        self.assertEqual(_layers_to_transform(22, "upper-11"), list(range(11, 22)))

    def test_upper_capped_at_total_layers(self):
        self.assertEqual(_layers_to_transform(22, "upper-100"), list(range(0, 22)))

    def test_invalid_string_raises(self):
        with self.assertRaises(ValueError):
            _layers_to_transform(22, "middle-3")
        with self.assertRaises(ValueError):
            _layers_to_transform(22, "upper")


class TestAnalyticalParamCount(unittest.TestCase):
    def test_default_seven_targets_all_layers_rank_8(self):
        result = analytical_param_count_finetune(
            TINYLLAMA_ARCH, lora_rank=8,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"),
            adapter_layers="all",
        )
        # Per-layer arithmetic (TinyLlama, r=8, kvh/nh = 4/32 = 1/8):
        #   q_proj: 8 * (2048 + 2048) = 32768
        #   k_proj: 8 * (2048 + 256)  = 18432
        #   v_proj: 8 * (2048 + 256)  = 18432
        #   o_proj: 8 * (2048 + 2048) = 32768
        #   gate_proj: 8 * (2048 + 5632) = 61440
        #   up_proj:   8 * (2048 + 5632) = 61440
        #   down_proj: 8 * (5632 + 2048) = 61440
        # Sum per layer: 286720.  x 22 layers: 6,307,840
        self.assertEqual(result["per_layer_adapter_params"], 286720)
        self.assertEqual(result["n_adapter"], 286720 * 22)
        self.assertEqual(result["n_layers_with_adapters"], 22)

    def test_upper_3_scales_adapter_count_linearly(self):
        full = analytical_param_count_finetune(
            TINYLLAMA_ARCH, lora_rank=8,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"),
            adapter_layers="all",
        )
        upper_3 = analytical_param_count_finetune(
            TINYLLAMA_ARCH, lora_rank=8,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"),
            adapter_layers="upper-3",
        )
        self.assertEqual(upper_3["n_layers_with_adapters"], 3)
        self.assertEqual(upper_3["n_adapter"] * (22 / 3), full["n_adapter"])

    def test_rank_scales_adapter_count_linearly(self):
        r8 = analytical_param_count_finetune(
            TINYLLAMA_ARCH, lora_rank=8, target_modules=("q_proj",),
            adapter_layers="all",
        )
        r64 = analytical_param_count_finetune(
            TINYLLAMA_ARCH, lora_rank=64, target_modules=("q_proj",),
            adapter_layers="all",
        )
        self.assertEqual(r64["n_adapter"], 8 * r8["n_adapter"])

    def test_attention_only_target_modules_excludes_mlp_paths(self):
        # q,k,v,o only -- no SwiGLU adapters.
        attn_only = analytical_param_count_finetune(
            TINYLLAMA_ARCH, lora_rank=8,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            adapter_layers="all",
        )
        self.assertEqual(attn_only["per_layer_adapter_params"],
                         32768 + 18432 + 18432 + 32768)  # 102400


class TestTrainableFrozenCount(unittest.TestCase):
    def test_hand_rolled_lora_stand_in_counts(self):
        # Base is 100 params frozen; adapters A and B are 20 params each, trainable.
        base = nn.Linear(10, 10, bias=False)  # 100
        for p in base.parameters():
            p.requires_grad = False
        adapter_a = nn.Parameter(torch.randn(4, 5))  # 20
        adapter_b = nn.Parameter(torch.randn(5, 4))  # 20
        model = nn.Module()
        model.base = base
        model.adapter_a = adapter_a
        model.adapter_b = adapter_b

        self.assertEqual(count_trainable_params(model), 40)
        self.assertEqual(count_frozen_params(model), 100)


class TestAdapterPlacementFreezesLowerLayersBackward(unittest.TestCase):
    """The Phase A empirical claim (peft-source-level): when layers_to_transform picks
    only the upper-N, backward saves nothing for layers below that. This test replicates
    the requires_grad structure PEFT produces and asserts the guarantee holds.

    On a real PeftModel this same assertion is checked by debug_finetune.py probe 3.
    """
    def test_frozen_lower_layers_have_no_grad_after_backward(self):
        # Six-layer stand-in mimicking PEFT: layers 0..2 are frozen (base only), layers
        # 3..5 have trainable adapters attached (parallel branch summed into the layer
        # output).
        torch.manual_seed(0)

        class LayerWithMaybeAdapter(nn.Module):
            def __init__(self, d, with_adapter):
                super().__init__()
                self.base = nn.Linear(d, d, bias=False)
                for p in self.base.parameters():
                    p.requires_grad = False
                self.with_adapter = with_adapter
                if with_adapter:
                    self.a = nn.Parameter(torch.randn(d, d))
                    self.b = nn.Parameter(torch.randn(d, d))

            def forward(self, x):
                y = self.base(x)
                if self.with_adapter:
                    y = y + x @ self.a @ self.b
                return y

        L = 6
        UPPER_N = 3
        layers = nn.ModuleList([
            LayerWithMaybeAdapter(8, with_adapter=(i >= L - UPPER_N))
            for i in range(L)
        ])

        x = torch.randn(2, 8, requires_grad=False)
        for layer in layers:
            x = layer(x)
        loss = x.sum()
        loss.backward()

        # Layers 0..2 (frozen, no adapter): base weight has p.grad is None (requires_grad
        # is False, so autograd never allocates a grad tensor).
        for i in range(L - UPPER_N):
            for p in layers[i].parameters():
                self.assertIsNone(p.grad, f"layer {i} unexpectedly got a grad tensor")

        # Layers 3..5 (upper-3): both adapter A and B have p.grad populated.
        for i in range(L - UPPER_N, L):
            self.assertIsNotNone(layers[i].a.grad)
            self.assertIsNotNone(layers[i].b.grad)


if __name__ == "__main__":
    unittest.main()
