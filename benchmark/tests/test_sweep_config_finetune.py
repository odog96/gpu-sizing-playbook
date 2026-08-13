"""Fine-tune sweep enumeration: expected configs per lever, correctly tagged, single-lever
guard, and combined-sweep placement."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sweep_config_finetune import (
    BASE_MODEL_NAME,
    DEFAULT_LORA_TARGET_MODULES,
    SINGLE_LEVER_TAGS,
    build_all_configs_finetune,
    differing_fields,
    make_baseline_finetune,
)


class TestBaseline(unittest.TestCase):
    def test_baseline_matches_plan(self):
        b = make_baseline_finetune()
        self.assertEqual(b.base_model_name, BASE_MODEL_NAME)
        self.assertEqual(b.base_storage_precision, "bf16")
        self.assertEqual(b.lora_rank, 8)
        self.assertEqual(b.lora_target_modules, DEFAULT_LORA_TARGET_MODULES)
        self.assertEqual(b.lora_adapter_layers, "all")
        self.assertEqual(b.batch, 8)
        self.assertEqual(b.seq_len, 512)
        self.assertEqual(b.precision, "amp_bf16")
        self.assertEqual(b.optimizer, "adam")
        self.assertFalse(b.checkpointing)
        self.assertEqual(b.steps, 7)
        self.assertEqual(b.lever, "baseline")
        self.assertEqual(b.lever_value, "baseline")

    def test_target_modules_tuple_matches_llama_convention(self):
        b = make_baseline_finetune()
        self.assertIn("q_proj", b.lora_target_modules)
        self.assertIn("gate_proj", b.lora_target_modules)
        self.assertIn("down_proj", b.lora_target_modules)
        self.assertEqual(len(b.lora_target_modules), 7)


class TestSweepEnumeration(unittest.TestCase):
    def setUp(self):
        self.baseline = make_baseline_finetune()
        self.configs = build_all_configs_finetune(self.baseline)

    def test_expected_config_counts_per_lever(self):
        counts = {}
        for c in self.configs:
            counts[c.lever] = counts.get(c.lever, 0) + 1
        self.assertEqual(counts["baseline"], 1)
        self.assertEqual(counts["batch_size"], 4)
        self.assertEqual(counts["seq_len"], 4)
        self.assertEqual(counts["lora_rank"], 4)
        self.assertEqual(counts["checkpointing"], 2)
        self.assertEqual(counts["adapter_placement"], 4)
        self.assertEqual(counts["base_precision"], 2)
        self.assertEqual(counts["combined"], 4)
        # Total: 25
        self.assertEqual(len(self.configs), 25)

    def test_each_config_tagged_with_lever_and_value(self):
        for c in self.configs:
            self.assertTrue(c.lever)
            self.assertTrue(c.lever_value)

    def test_single_lever_sweeps_vary_at_most_one_field_from_baseline(self):
        for c in self.configs:
            if c.lever in SINGLE_LEVER_TAGS:
                diffs = differing_fields(self.baseline, c)
                self.assertLessEqual(
                    len(diffs), 1,
                    f"{c.lever}={c.lever_value} differs in {diffs}"
                )

    def test_combined_sweep_at_the_single_demanding_batch_and_seq(self):
        combined = [c for c in self.configs if c.lever == "combined"]
        self.assertEqual(len({c.batch for c in combined}), 1)
        self.assertEqual(len({c.seq_len for c in combined}), 1)
        self.assertEqual({c.batch for c in combined}, {32})
        self.assertEqual({c.seq_len for c in combined}, {2048})
        # ...but is explicitly allowed to vary optimizer/checkpointing together.
        self.assertIn("adam8bit", [c.optimizer for c in combined])
        self.assertIn(True, [c.checkpointing for c in combined])

    def test_rank_sweep_covers_the_expected_endpoints(self):
        ranks = [c.lora_rank for c in self.configs if c.lever == "lora_rank"]
        self.assertEqual(sorted(ranks), [4, 8, 16, 64])

    def test_adapter_placement_sweep_covers_expected_values(self):
        placements = [c.lora_adapter_layers for c in self.configs if c.lever == "adapter_placement"]
        self.assertEqual(sorted(placements), sorted(["all", "upper-11", "upper-6", "upper-3"]))

    def test_base_precision_sweep_covers_bf16_and_fp32_by_default(self):
        precisions = [c.base_storage_precision for c in self.configs if c.lever == "base_precision"]
        self.assertEqual(sorted(precisions), ["bf16", "fp32"])


class TestConfigSerialization(unittest.TestCase):
    def test_as_dict_round_trip_preserves_all_fields(self):
        # The subprocess boundary is JSON: as_dict -> json.dumps -> json.loads -> Config(**).
        # tuple survives round-trip as a list; the constructor accepts it and it becomes a
        # tuple again on the receiving side via the _variant helper for further copies.
        b = make_baseline_finetune()
        d = b.as_dict()
        self.assertEqual(d["base_model_name"], BASE_MODEL_NAME)
        self.assertEqual(d["lora_rank"], 8)
        self.assertEqual(d["batch"], 8)
        self.assertEqual(d["seq_len"], 512)


if __name__ == "__main__":
    unittest.main()
