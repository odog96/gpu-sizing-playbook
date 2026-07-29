"""Sweep enumeration: each lever produces the expected configs, correctly tagged, and
each single-lever sweep varies exactly one field from baseline."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sweep_config import SINGLE_LEVER_TAGS, build_all_configs, differing_fields, make_baseline


class TestSweepEnumeration(unittest.TestCase):
    def setUp(self):
        self.baseline = make_baseline()
        self.configs = build_all_configs(self.baseline)

    def test_expected_config_counts_per_lever(self):
        counts = {}
        for c in self.configs:
            counts[c.lever] = counts.get(c.lever, 0) + 1
        self.assertEqual(counts["baseline"], 1)
        self.assertEqual(counts["batch_size"], 3)
        self.assertEqual(counts["seq_len"], 4)
        self.assertEqual(counts["optimizer"], 2)
        self.assertEqual(counts["checkpointing"], 2)
        self.assertEqual(counts["checkpointing_batch"], 3)
        self.assertEqual(counts["checkpointing_seq"], 4)
        self.assertEqual(counts["combined"], 4)
        self.assertEqual(counts["precision"], 1)

    def test_each_config_tagged_with_varied_lever_and_value(self):
        for c in self.configs:
            self.assertTrue(c.lever)
            self.assertTrue(c.lever_value)

    def test_single_lever_sweeps_vary_at_most_one_field_from_baseline(self):
        for c in self.configs:
            if c.lever in SINGLE_LEVER_TAGS:
                diffs = differing_fields(self.baseline, c)
                self.assertLessEqual(len(diffs), 1, f"{c.lever}={c.lever_value} differs in {diffs}")

    def test_checkpointing_crosses_are_checkpointing_true_across_every_batch_and_seq_value(self):
        batch_cross = [c for c in self.configs if c.lever == "checkpointing_batch"]
        seq_cross = [c for c in self.configs if c.lever == "checkpointing_seq"]
        self.assertTrue(all(c.checkpointing for c in batch_cross))
        self.assertTrue(all(c.checkpointing for c in seq_cross))
        self.assertEqual({c.batch for c in batch_cross}, {256, 1024, 4096})
        self.assertEqual({c.seq_len for c in seq_cross}, {100, 512, 1024, 2048})

    def test_combined_sweep_is_at_the_single_demanding_batch_and_seq(self):
        combined = [c for c in self.configs if c.lever == "combined"]
        self.assertEqual(len({c.batch for c in combined}), 1)
        self.assertEqual(len({c.seq_len for c in combined}), 1)
        # ...but is explicitly allowed to vary optimizer/checkpointing together.
        self.assertIn("adam8bit", [c.optimizer for c in combined])
        self.assertIn(True, [c.checkpointing for c in combined])


if __name__ == "__main__":
    unittest.main()
