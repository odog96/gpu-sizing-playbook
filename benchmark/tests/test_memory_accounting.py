"""Direct tests for the measured_*_bytes helpers -- pure tensor accounting, CPU only.

These exist so the numbers fed into the static-vs-measured cross-check in
test_predictions.py are themselves trustworthy: e.g. that grad bytes are 0 before any
backward() call, and that optimizer bytes are 0 before any step().
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from memory_accounting import measured_grad_bytes, measured_optimizer_bytes, measured_param_bytes


class TestMeasuredParamBytes(unittest.TestCase):
    def test_matches_element_size_times_count(self):
        model = nn.Linear(10, 10, bias=False)  # 100 params, fp32 -> 400 bytes
        self.assertEqual(measured_param_bytes(model), 400)


class TestMeasuredGradBytes(unittest.TestCase):
    def test_zero_before_backward(self):
        model = nn.Linear(10, 10, bias=False)
        self.assertEqual(measured_grad_bytes(model), 0)

    def test_matches_param_bytes_after_backward(self):
        model = nn.Linear(10, 10, bias=False)
        loss = model(torch.randn(4, 10)).sum()
        loss.backward()
        # Every param got a gradient of the same shape/dtype, so grad bytes == param bytes.
        self.assertEqual(measured_grad_bytes(model), measured_param_bytes(model))


class TestMeasuredOptimizerBytes(unittest.TestCase):
    def test_zero_before_any_step(self):
        model = nn.Linear(10, 10, bias=False)
        optimizer = torch.optim.Adam(model.parameters())
        self.assertEqual(measured_optimizer_bytes(optimizer), 0)

    def test_adam_holds_two_moments_per_param_after_one_step(self):
        model = nn.Linear(10, 10, bias=False)  # 100 params
        optimizer = torch.optim.Adam(model.parameters())
        loss = model(torch.randn(4, 10)).sum()
        loss.backward()
        optimizer.step()
        # exp_avg + exp_avg_sq, both fp32, same shape as the param (2 * 100 * 4 bytes),
        # plus a tiny 0-dim fp32 "step" counter tensor this torch version also keeps in
        # state -- walking every tensor in state (not just named ones) is the point of
        # measured_optimizer_bytes, so the real total includes it too.
        expected = 2 * 100 * 4 + 4
        self.assertEqual(measured_optimizer_bytes(optimizer), expected)


if __name__ == "__main__":
    unittest.main()
