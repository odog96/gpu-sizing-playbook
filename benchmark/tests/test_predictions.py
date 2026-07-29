"""Known-value assertions for the corrected formulas. Formula drift must break a test.

Also includes the static-vs-measured cross-check from the prediction-formula review:
predicted weights+gradients+optimizer bytes must match what a real (CPU-built, so no
GPU needed here) model+optimizer actually holds, within 2%. That's what makes the
"static total was predicted correctly" claim a checked fact instead of an assumption.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from memory_accounting import measured_grad_bytes, measured_optimizer_bytes, measured_param_bytes
from predictions import (
    GB,
    GPU_CAPACITY_GB,
    predict_activation_bytes,
    predict_autocast_weight_cache_bytes,
    predict_gradients_bytes,
    predict_line_items,
    predict_optimizer_bytes,
    predict_weights_bytes,
)


class TestStaticFormulas(unittest.TestCase):
    def test_weights_are_always_fp32_regardless_of_precision_lever(self):
        # This codebase never casts parameter storage -- only autocast's forward compute
        # changes with the precision lever. See predictions.py's module docstring.
        self.assertAlmostEqual(predict_weights_bytes(1_000_000_000) / GB, 4.0, places=6)

    def test_gradients_match_weights_formula(self):
        self.assertEqual(predict_gradients_bytes(1_000_000_000), predict_weights_bytes(1_000_000_000))

    def test_adam_optimizer_is_8_bytes_per_param(self):
        self.assertAlmostEqual(predict_optimizer_bytes(1_000_000_000, "adam") / GB, 8.0, places=6)

    def test_8bit_adam_optimizer_is_2_bytes_per_param(self):
        self.assertAlmostEqual(predict_optimizer_bytes(1_000_000_000, "adam8bit") / GB, 2.0, places=6)

    def test_autocast_weight_cache_only_applies_under_amp_bf16(self):
        self.assertAlmostEqual(predict_autocast_weight_cache_bytes(1_000_000_000, "amp_bf16") / GB, 2.0, places=6)
        self.assertEqual(predict_autocast_weight_cache_bytes(1_000_000_000, "fp32"), 0)


class TestActivationFormula(unittest.TestCase):
    def test_baseline_config_matches_hand_derived_per_tensor_sum(self):
        # batch=256, seq=100, d=2048, 20 layers, ff_mult=4, vocab=1000, amp_bf16.
        # Hand-derived: 7 attn tensors + 2 mlp-prenorm + 2 mlp-hidden + 3 dropout masks,
        # per layer x 20, plus embedding/logits/fp32-upcast once per model.
        floor_gb = predict_activation_bytes(256, 100, 2048, 20, 4, 1000, "amp_bf16", checkpointing=False) / GB
        self.assertAlmostEqual(floor_gb, 42.2015, places=3)

    def test_checkpointing_is_not_floor_divided_by_layer_count(self):
        # The old bug: checkpointed predicted activations = (batch*seq_len*d_model*2*
        # n_layers) / n_layers = 0.105 GB at baseline scale, against a measured ~4.1 GB
        # (~40x under). The corrected formula must land far closer to the measured figure.
        old_buggy_checkpointed_gb = (256 * 100 * 2048 * 2 * 20) / 20 / GB  # == 0.1049 GB
        full = predict_activation_bytes(256, 100, 2048, 20, 4, 1000, "amp_bf16", checkpointing=False)
        checkpointed_gb = predict_activation_bytes(256, 100, 2048, 20, 4, 1000, "amp_bf16", checkpointing=True) / GB
        self.assertGreater(checkpointed_gb, old_buggy_checkpointed_gb * 10)
        self.assertLess(checkpointed_gb * GB, full)

    def test_fp32_precision_uses_4_byte_activations_and_no_upcast_term(self):
        bf16_gb = predict_activation_bytes(256, 100, 2048, 20, 4, 1000, "amp_bf16", checkpointing=False) / GB
        fp32_gb = predict_activation_bytes(256, 100, 2048, 20, 4, 1000, "fp32", checkpointing=False) / GB
        # fp32 activations aren't simply 2x bf16 activations, because the fp32-upcast
        # term (which only exists under amp_bf16) is removed, not doubled.
        self.assertGreater(fp32_gb, bf16_gb)
        self.assertLess(fp32_gb, bf16_gb * 2)


class TestLineItemsAndOomPrediction(unittest.TestCase):
    def test_baseline_config_predicted_within_10pct_of_measured(self):
        # n_params from the real sweep's baseline row.
        predicted = predict_line_items(
            1_011_262_440, "amp_bf16", "adam", 256, 100, 2048, 20, 4, 1000, checkpointing=False
        )
        predicted_allocated_gb = predicted["allocated_total"] / GB
        measured_allocated_gb = 59.1452  # from results.csv, baseline row
        pct_error = abs(predicted_allocated_gb - measured_allocated_gb) / measured_allocated_gb
        self.assertLess(pct_error, 0.10)

    def test_demanding_combined_config_correctly_predicts_oom(self):
        predicted = predict_line_items(
            1_010_728_960, "amp_bf16", "adam8bit", 4096, 2048, 2048, 20, 4, 1000, checkpointing=True
        )
        self.assertTrue(predicted["predicted_oom"])

    def test_gpu_capacity_constant_matches_the_a100_80gb_fleet(self):
        # 79.25 GiB usable, as read off this sweep's own OOM error messages.
        self.assertAlmostEqual(GPU_CAPACITY_GB, 79.25 * (1024 ** 3) / GB, places=2)


class TestStaticVsMeasured(unittest.TestCase):
    """The Task 2 cross-check: predicted static bytes (weights+gradients+optimizer) vs.
    what a real model+optimizer actually holds, measured directly. Runs on CPU -- these
    are pure tensor-accounting facts about PyTorch, not GPU behavior, so no GPU is
    needed to verify them."""

    def test_predicted_static_bytes_match_measured_within_2pct(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        x = torch.randn(8, 512)
        target = torch.randn(8, 512)
        loss = nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()

        n_params = sum(p.numel() for p in model.parameters())
        predicted_static = (
            predict_weights_bytes(n_params) + predict_gradients_bytes(n_params)
            + predict_optimizer_bytes(n_params, "adam")
        )
        measured_static = measured_param_bytes(model) + measured_grad_bytes(model) + measured_optimizer_bytes(optimizer)

        pct_error = abs(predicted_static - measured_static) / measured_static
        self.assertLess(pct_error, 0.02, f"predicted={predicted_static} measured={measured_static}")


if __name__ == "__main__":
    unittest.main()
