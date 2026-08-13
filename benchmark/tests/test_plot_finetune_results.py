"""Chart generation: run plot_finetune_results.py against the fixture CSV in a temp dir,
assert every expected PNG/SVG file exists and is non-trivial in size."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "..")
FIXTURE = os.path.join(BENCHMARK_DIR, "fixtures", "results_finetune_sample.csv")
SCRIPT = os.path.join(BENCHMARK_DIR, "plot_finetune_results.py")

EXPECTED_FILES = [
    "adapter_placement.png", "adapter_placement.svg",
    "rank_flatness.png", "rank_flatness.svg",
    "predicted_vs_measured.png", "predicted_vs_measured.svg",
    "autocast_cache_residual.png", "autocast_cache_residual.svg",
    "component_stack.png", "component_stack.svg",
]


class TestChartGeneration(unittest.TestCase):
    def test_charts_generated_from_fixture_csv(self):
        outdir = tempfile.mkdtemp()
        try:
            result = subprocess.run(
                [sys.executable, SCRIPT, "--input", FIXTURE, "--outdir", outdir],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for fname in EXPECTED_FILES:
                path = os.path.join(outdir, fname)
                self.assertTrue(os.path.exists(path), f"missing {fname}")
                self.assertGreater(os.path.getsize(path), 2000,
                                   f"{fname} looks too small to be a real chart")
        finally:
            shutil.rmtree(outdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
