"""Parameter count: catches an architecture bug before it costs GPU hours."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import TinyTransformer, analytical_param_count, count_parameters


class TestParameterCount(unittest.TestCase):
    def test_tiny_config_matches_analytical_formula(self):
        d, layers, heads, ff_mult, vocab = 256, 4, 8, 4, 1000
        model = TinyTransformer(
            vocab_size=vocab, d_model=d, n_layers=layers, n_heads=heads, ff_mult=ff_mult, seq_len=100
        )
        actual = count_parameters(model)
        analytic = analytical_param_count(d, layers, vocab, ff_mult)
        # 12*d^2*L dominates; biases/layernorms make up the small remainder.
        self.assertAlmostEqual(actual / analytic, 1.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
