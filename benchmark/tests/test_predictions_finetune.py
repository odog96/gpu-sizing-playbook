"""Known-value assertions for the fine-tune formulas. Formula drift must break a test.

Small hand-derived config throughout so the arithmetic is checkable line by line:
  n_base=1,000,000, n_adapter=10,000, d=64, layers=2, num_heads=4, num_kv_heads=1,
  ff_intermediate=256, vocab=100, batch=2, seq_len=32, precision=amp_bf16,
  base_storage_precision=bf16, optimizer=adam, checkpointing=False, dropout_p=0.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from predictions_finetune import (
    GB,
    GPU_CAPACITY_GB,
    predict_activation_bytes,
    predict_adapter_gradients_bytes,
    predict_adapter_optimizer_bytes,
    predict_adapter_weights_bytes,
    predict_autocast_weight_cache_bytes,
    predict_frozen_weights_bytes,
    predict_line_items_finetune,
    resolve_layers_needing_backward,
)


# --- Baseline config for hand-derivation ---
SMALL = dict(
    n_base=1_000_000, n_adapter=10_000,
    d_model=64, n_layers_total=2, num_heads=4, num_kv_heads=1,
    ff_intermediate=256, vocab=100, batch=2, seq_len=32,
    precision="amp_bf16", base_storage_precision="bf16", optimizer="adam",
    adapter_layers="all", checkpointing=False, dropout_p=0.0,
)


class TestStaticFormulas(unittest.TestCase):
    def test_frozen_weights_at_bf16_are_2_bytes_per_param(self):
        self.assertEqual(predict_frozen_weights_bytes(1_000_000, "bf16"), 2_000_000)

    def test_frozen_weights_at_fp32_are_4_bytes_per_param(self):
        self.assertEqual(predict_frozen_weights_bytes(1_000_000, "fp32"), 4_000_000)

    def test_frozen_weights_at_int8_are_1_byte_per_param(self):
        self.assertEqual(predict_frozen_weights_bytes(1_000_000, "int8"), 1_000_000)

    def test_frozen_weights_at_int4_are_half_byte_per_param(self):
        # 0.5 bytes/param -> int() truncation is correct: two params pack into one byte.
        self.assertEqual(predict_frozen_weights_bytes(1_000_000, "int4"), 500_000)

    def test_adapter_weights_always_fp32(self):
        # LoRA adapters stay fp32 for update stability.
        self.assertEqual(predict_adapter_weights_bytes(10_000), 40_000)

    def test_adapter_gradients_match_adapter_weight_formula(self):
        self.assertEqual(predict_adapter_gradients_bytes(10_000),
                         predict_adapter_weights_bytes(10_000))

    def test_adam_optimizer_is_8_bytes_per_trainable_param(self):
        self.assertEqual(predict_adapter_optimizer_bytes(10_000, "adam"), 80_000)

    def test_adam8bit_optimizer_is_2_bytes_per_trainable_param(self):
        self.assertEqual(predict_adapter_optimizer_bytes(10_000, "adam8bit"), 20_000)


class TestAutocastWeightCache(unittest.TestCase):
    def test_bf16_base_collapses_cache_to_adapter_only(self):
        # Base is already bf16 -- nothing to cast for the base; only fp32 adapters get
        # a bf16 cache copy. This is the residual case Article 2's fifth item hangs on.
        self.assertEqual(
            predict_autocast_weight_cache_bytes(1_000_000, 10_000, "amp_bf16", "bf16"),
            10_000 * 2,
        )

    def test_fp32_base_reproduces_article_1_cache(self):
        # If a base is loaded fp32, the full ~n_params * 2 cache is back.
        self.assertEqual(
            predict_autocast_weight_cache_bytes(1_000_000, 10_000, "amp_bf16", "fp32"),
            (1_000_000 + 10_000) * 2,
        )

    def test_fp32_precision_zeroes_the_cache_regardless_of_base(self):
        # No autocast, no cache.
        self.assertEqual(
            predict_autocast_weight_cache_bytes(1_000_000, 10_000, "fp32", "bf16"),
            0,
        )
        self.assertEqual(
            predict_autocast_weight_cache_bytes(1_000_000, 10_000, "fp32", "fp32"),
            0,
        )


class TestActivationFormula(unittest.TestCase):
    def test_small_config_matches_hand_derived_per_tensor_sum(self):
        # b = 2 (amp_bf16). bsd = 2*32*64*2 = 8192; bsd_kv = 2*32*64*(1/4)*2 = 2048;
        # bsi = 2*32*256*2 = 32768.
        # Per layer: 6*8192 + 2*2048 + 2*8192 + 4*32768 + 0
        #          = 49152 + 4096 + 16384 + 131072 = 200704
        # (6 attention bsd tensors: pre-norm, post-norm, Q-in, Q-post-RoPE, attn-out, o_proj-in)
        # Once: embed 8192 + logits 12800 + fp32_upcast 25600 + ce_backward 2*25600 = 97792.
        # Non-checkpointed, layers=2 -> 200704*2 + 97792 = 499200.
        result = predict_activation_bytes(
            batch=2, seq_len=32, d_model=64, n_layers_needing_backward=2,
            num_heads=4, num_kv_heads=1, ff_intermediate=256, vocab=100,
            precision="amp_bf16", checkpointing=False, dropout_p=0.0,
        )
        self.assertEqual(result, 499200)

    def test_gqa_narrows_kv_projections_relative_to_mha(self):
        # Same shape but num_kv_heads == num_heads (MHA) vs num_kv_heads=1 (max GQA).
        # MHA saves 2 * bsd for K and V; GQA(kv=1/4) saves 2 * bsd/4 -- ~1.5*bsd less.
        gqa = predict_activation_bytes(
            2, 32, 64, 2, num_heads=4, num_kv_heads=1, ff_intermediate=256, vocab=100,
            precision="amp_bf16", checkpointing=False,
        )
        mha = predict_activation_bytes(
            2, 32, 64, 2, num_heads=4, num_kv_heads=4, ff_intermediate=256, vocab=100,
            precision="amp_bf16", checkpointing=False,
        )
        self.assertLess(gqa, mha)
        # Difference: 2 layers * 2 * (bsd - bsd/4) = 2 * 2 * (8192 - 2048) = 24576 bytes
        self.assertEqual(mha - gqa, 24576)

    def test_dropout_zero_zeroes_the_mask_term(self):
        # TinyLlama default: dropout_p=0 -> no mask memory.
        with_dropout = predict_activation_bytes(
            2, 32, 64, 2, 4, 1, 256, 100, "amp_bf16", False, dropout_p=0.1,
        )
        no_dropout = predict_activation_bytes(
            2, 32, 64, 2, 4, 1, 256, 100, "amp_bf16", False, dropout_p=0.0,
        )
        # Per-layer dropout: 2*(2*32*64) + 2*32*256 = 8192 + 16384 = 24576. Times 2 layers.
        self.assertEqual(with_dropout - no_dropout, 24576 * 2)

    def test_checkpointing_saves_fp32_boundaries_not_bf16(self):
        # Same fp32-boundary story as Article 1: the tensor passed BETWEEN layers is
        # 4 bytes/element even under bf16 autocast, because RMSNorm and residual adds
        # run fp32.
        # (L_backward + 3) * bsd_fp32 = (2+3) * 2*32*64*4 = 5 * 16384 = 81920
        # recompute: 6*bsd_bf16 + 2*bsd_kv_bf16 + 4*bsi_bf16 = 6*8192 + 2*2048 + 4*32768
        #          = 49152 + 4096 + 131072 = 184320
        # (6 bsd for RoPE Q fresh tensor -- same as non-checkpointed per-layer count)
        # dropout = 0, live_logits bf16 = 2*32*100*2 = 12800
        # ce_backward_fp32 = 2 * 2*32*100*4 = 51200
        # Total: 81920 + 184320 + 0 + 12800 + 51200 = 330240
        result = predict_activation_bytes(
            2, 32, 64, 2, 4, 1, 256, 100, "amp_bf16", checkpointing=True, dropout_p=0.0,
        )
        self.assertEqual(result, 330240)

    def test_checkpointing_is_smaller_than_non_checkpointed(self):
        # For a wider config where activations dominate, the reduction must be real.
        ckpt = predict_activation_bytes(
            64, 512, 2048, 22, 32, 4, 5632, 32000, "amp_bf16", checkpointing=True,
        )
        no_ckpt = predict_activation_bytes(
            64, 512, 2048, 22, 32, 4, 5632, 32000, "amp_bf16", checkpointing=False,
        )
        self.assertLess(ckpt, no_ckpt / 2)  # at least 2x smaller

    def test_adapter_placement_scales_activations_by_layer_count(self):
        # 'upper-1' at 2-layer model = 1 layer of backward reach vs 'all' = 2.
        full = predict_activation_bytes(
            2, 32, 64, 2, 4, 1, 256, 100, "amp_bf16", False,
        )
        half = predict_activation_bytes(
            2, 32, 64, 1, 4, 1, 256, 100, "amp_bf16", False,
        )
        # once_per_model = 97792; per_layer = 200704; full = 200704*2 + 97792 = 499200;
        # half = 200704*1 + 97792 = 298496. Half saves exactly one per-layer term.
        self.assertEqual(full - half, 200704)


class TestResolveLayersNeedingBackward(unittest.TestCase):
    def test_all_returns_total(self):
        self.assertEqual(resolve_layers_needing_backward(22, "all"), 22)

    def test_upper_n_returns_n(self):
        self.assertEqual(resolve_layers_needing_backward(22, "upper-3"), 3)
        self.assertEqual(resolve_layers_needing_backward(22, "upper-11"), 11)

    def test_upper_n_capped_at_total(self):
        # Requesting more layers than exist just gives you all of them.
        self.assertEqual(resolve_layers_needing_backward(22, "upper-100"), 22)

    def test_invalid_placement_string_raises(self):
        with self.assertRaises(ValueError):
            resolve_layers_needing_backward(22, "middle-3")


class TestLineItemsAndOomPrediction(unittest.TestCase):
    def test_baseline_returns_all_expected_keys(self):
        result = predict_line_items_finetune(**SMALL)
        for key in (
            "frozen_weights", "adapter_weights", "gradients", "optimizer",
            "autocast_weight_cache", "activations", "cublas_workspace",
            "allocated_total", "reserved_total", "predicted_oom",
            "predicted_trainable_param_count", "predicted_frozen_param_count",
            "n_layers_needing_backward",
        ):
            self.assertIn(key, result, f"missing key {key}")

    def test_baseline_allocated_total_composes_from_hand_derived_pieces(self):
        result = predict_line_items_finetune(**SMALL)
        # frozen 2,000,000 + adapters 40,000 + grads 40,000 + optimizer 80,000
        # + autocast_cache 20,000 + activations 499,200
        # + cublas int((4.316 + 1.871*2) * 2*32*64*4) = int(8.058 * 16384) = 132022
        # = 2,811,222 bytes
        self.assertEqual(result["frozen_weights"], 2_000_000)
        self.assertEqual(result["adapter_weights"], 40_000)
        self.assertEqual(result["gradients"], 40_000)
        self.assertEqual(result["optimizer"], 80_000)
        self.assertEqual(result["autocast_weight_cache"], 20_000)
        self.assertEqual(result["activations"], 499_200)
        self.assertEqual(result["cublas_workspace"], 132_022)
        self.assertEqual(result["allocated_total"], 2_811_222)

    def test_checkpointing_zeroes_cublas_workspace_line_item(self):
        # Under gradient checkpointing the allocator repeatedly hits its cap during
        # backward-recompute, forcing cuBLAS to release its cached workspaces between
        # segments. The 2026-08-17 v2 H100 sweep read this as ~zero at peak -- keeping
        # the full non-checkpointed workspace count over-predicted the three checkpointed
        # rows by 36% and 72%. Zeroing it lands them at +2.8%, +8.1%, +8.1%. Empirical,
        # documented in the same paragraph as CUDA_CONTEXT_OVERHEAD_GB.
        args = dict(SMALL)
        args["checkpointing"] = True
        result = predict_line_items_finetune(**args)
        self.assertEqual(result["cublas_workspace"], 0)

    def test_non_checkpointing_keeps_cublas_workspace_line_item(self):
        # Sanity: the zeroing only fires under checkpointing. Non-checkpointed configs
        # keep the fitted K = 4.316 + 1.871 * L_bwd workspaces.
        result = predict_line_items_finetune(**SMALL)
        self.assertEqual(result["cublas_workspace"], 132_022)

    def test_reserved_adds_context_and_fragmentation(self):
        result = predict_line_items_finetune(**SMALL)
        gap = result["reserved_total"] - result["allocated_total"]
        # 0.6 GB context + 1.2 GB fragmentation = 1.8 GB
        self.assertAlmostEqual(gap / GB, 1.8, places=6)

    def test_checkpointed_reserved_adds_larger_fragmentation(self):
        args = dict(SMALL)
        args["checkpointing"] = True
        result = predict_line_items_finetune(**args)
        gap = result["reserved_total"] - result["allocated_total"]
        # 0.6 GB context + 5.2 GB checkpointing fragmentation = 5.8 GB
        self.assertAlmostEqual(gap / GB, 5.8, places=6)

    def test_small_config_does_not_predict_oom(self):
        result = predict_line_items_finetune(**SMALL)
        self.assertFalse(result["predicted_oom"])

    def test_absurd_batch_predicts_oom(self):
        args = dict(SMALL)
        args["batch"] = 1_000_000  # forces activation term past 79 GB
        args["seq_len"] = 2048
        args["d_model"] = 2048
        args["ff_intermediate"] = 5632
        result = predict_line_items_finetune(**args)
        self.assertTrue(result["predicted_oom"])

    def test_upper_3_placement_reduces_activation_line(self):
        args_all = dict(SMALL)
        args_upper = dict(SMALL)
        args_upper["adapter_layers"] = "upper-1"
        result_all = predict_line_items_finetune(**args_all)
        result_upper = predict_line_items_finetune(**args_upper)
        # Fewer backward-reach layers -> smaller activations, everything else equal.
        self.assertLess(result_upper["activations"], result_all["activations"])
        self.assertEqual(result_upper["n_layers_needing_backward"], 1)
        self.assertEqual(result_all["n_layers_needing_backward"], 2)

    def test_int4_base_reduces_frozen_weights_only(self):
        args_bf16 = dict(SMALL)
        args_int4 = dict(SMALL)
        args_int4["base_storage_precision"] = "int4"
        r_bf16 = predict_line_items_finetune(**args_bf16)
        r_int4 = predict_line_items_finetune(**args_int4)
        # int4 base = 500,000 bytes vs bf16 base = 2,000,000 bytes -- 1.5 MB saved.
        self.assertEqual(r_bf16["frozen_weights"] - r_int4["frozen_weights"], 1_500_000)
        # Nothing else moves.
        for k in ("adapter_weights", "gradients", "optimizer", "activations"):
            self.assertEqual(r_bf16[k], r_int4[k])

    def test_rank_flatness_static_columns_grow_linearly_activations_do_not(self):
        # The article's rank-flatness claim, as a formula-level assertion:
        # adapter/gradient/optimizer bytes are linear in n_adapter, activations aren't.
        args_r8 = dict(SMALL)
        args_r64 = dict(SMALL); args_r64["n_adapter"] = 80_000  # rank 8 -> 64 = 8x
        r8 = predict_line_items_finetune(**args_r8)
        r64 = predict_line_items_finetune(**args_r64)
        self.assertEqual(r64["adapter_weights"], 8 * r8["adapter_weights"])
        self.assertEqual(r64["gradients"], 8 * r8["gradients"])
        self.assertEqual(r64["optimizer"], 8 * r8["optimizer"])
        self.assertEqual(r64["activations"], r8["activations"])
        self.assertEqual(r64["frozen_weights"], r8["frozen_weights"])


class TestGpuCapacityConstant(unittest.TestCase):
    def test_matches_h100_and_a100_80gb(self):
        # 79.18 GiB usable on both 80GB SKUs.
        self.assertAlmostEqual(GPU_CAPACITY_GB, 79.18 * (1024 ** 3) / GB, places=2)


if __name__ == "__main__":
    unittest.main()
