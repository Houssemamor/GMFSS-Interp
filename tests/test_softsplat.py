"""Tests for the weighted softsplat kernel.

The splat operator is a forward scatter, so its accumulation order is not fixed:
atomicAdd makes two runs of the *same* kernel differ in the last bits. These
tests therefore compare the weighted kernel against the upstream formulation
relative to that noise floor, and they pin the property that matters - the
weighted scatter plus its normalising channel is the concatenated scatter.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

import torch

VENDOR = pathlib.Path(__file__).resolve().parent.parent / "vendor"

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    try:
        from vfi_models.ops import init as ops_init
        from vfi_models.ops.cupy_ops.softsplat import (
            softsplat_func,
            softsplat_weighted,
        )

        ops_init()
    except Exception:  # pragma: no cover - cupy or the CUDA toolchain is absent
        CUDA_AVAILABLE = False

# Two runs of the same kernel differ by roughly one fp32 ulp on values near 1.
# The tolerance is one decade above that, which still catches any change in the
# arithmetic rather than in the accumulation order.
TOLERANCE = 1e-5


def old_formulation(ten_in, ten_flow, ten_metric):
    """What softsplat() computed before the weighted kernel existed."""
    exp_metric = ten_metric.exp()
    stacked = torch.cat([ten_in * exp_metric, exp_metric], 1)
    out = softsplat_func.apply(stacked, ten_flow)
    return out[:, :-1] / (out[:, -1:] + 1e-7)


def new_formulation(ten_in, ten_flow, ten_metric):
    out = softsplat_weighted(ten_in, ten_flow, ten_metric.exp())
    return out[:, :-1] / (out[:, -1:] + 1e-7)


@unittest.skipUnless(CUDA_AVAILABLE, "needs CUDA and the cupy softsplat kernels")
class WeightedSplatTest(unittest.TestCase):
    def scatter(self, channels: int, height: int, width: int, flow_scale: float, batch: int = 1):
        generator = torch.Generator(device="cuda").manual_seed(20261027)
        ten_in = torch.randn(batch, channels, height, width, device="cuda", generator=generator)
        flow = torch.randn(batch, 2, height, width, device="cuda", generator=generator) * flow_scale
        metric = torch.randn(batch, 1, height, width, device="cuda", generator=generator)
        return ten_in, flow, metric

    def assert_close(self, measured, expected, label: str) -> None:
        drift = (measured - expected).abs().max().item()
        self.assertLess(drift, TOLERANCE, f"{label}: max drift {drift:.3e}")

    def test_matches_concatenated_formulation(self) -> None:
        ten_in, flow, metric = self.scatter(8, 32, 40, 1.0)
        with torch.inference_mode():
            expected = old_formulation(ten_in, flow, metric)
            measured = new_formulation(ten_in, flow, metric)
        self.assertEqual(measured.shape, expected.shape)
        self.assertTrue(torch.isfinite(measured).all())
        self.assert_close(measured, expected, "short flow")

    def test_matches_when_flow_leaves_the_frame(self) -> None:
        ten_in, flow, metric = self.scatter(8, 32, 40, 1.0)
        flow[0, 0, 0, 0] = 1e6
        flow[0, 1, -1, -1] = -1e6
        with torch.inference_mode():
            expected = old_formulation(ten_in, flow, metric)
            measured = new_formulation(ten_in, flow, metric)
        self.assert_close(measured, expected, "out of bounds")

    def test_matches_when_flow_is_not_finite(self) -> None:
        ten_in, flow, metric = self.scatter(8, 32, 40, 1.0)
        flow[0, 0, 3, 3] = float("nan")
        flow[0, 1, 7, 7] = float("inf")
        with torch.inference_mode():
            expected = old_formulation(ten_in, flow, metric)
            measured = new_formulation(ten_in, flow, metric)
        self.assert_close(measured, expected, "non-finite flow")

    def test_weight_channel_holds_the_scattered_weight(self) -> None:
        # The last channel is the denominator of the splat, so it must equal the
        # one the concatenated formulation produces for the same inputs.
        ten_in, flow, metric = self.scatter(4, 16, 24, 0.5)
        weight = metric.exp()
        with torch.inference_mode():
            reference = softsplat_func.apply(
                torch.cat([ten_in * weight, weight], 1), flow
            )
            out = softsplat_weighted(ten_in, flow, weight)
        self.assertEqual(out.shape[1], ten_in.shape[1] + 1)
        self.assert_close(out[:, -1], reference[:, -1], "normaliser channel")

    def test_multi_batch_and_smaller_plane(self) -> None:
        ten_in, flow, metric = self.scatter(3, 9, 17, 2.0, batch=2)
        with torch.inference_mode():
            expected = old_formulation(ten_in, flow, metric)
            measured = new_formulation(ten_in, flow, metric)
        self.assert_close(measured, expected, "batch 2, ragged size")


if __name__ == "__main__":
    unittest.main()
