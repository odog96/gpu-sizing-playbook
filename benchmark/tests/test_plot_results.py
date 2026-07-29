"""Chart generation: run plot_results.py against the fixture CSV in a temp dir, assert
the expected PNG and SVG files exist and are non-trivial in size."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "..")
FIXTURE = os.path.join(BENCHMARK_DIR, "fixtures", "results_sample.csv")
SCRIPT = os.path.join(BENCHMARK_DIR, "plot_results.py")

EXPECTED_FILES = [
    "memory_vs_batch.png", "memory_vs_batch.svg",
    "memory_vs_seqlen.png", "memory_vs_seqlen.svg",
    "lever_impact.png", "lever_impact.svg",
    "allocated_vs_reserved.png", "allocated_vs_reserved.svg",
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
                self.assertGreater(os.path.getsize(path), 2000, f"{fname} looks too small to be a real chart")
        finally:
            shutil.rmtree(outdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
