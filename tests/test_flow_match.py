"""Tests for the banded global matching in GMFlow.

Two places in the flow network compute softmax(q k^T / sqrt(d)) v, and both wrote
out the whole quadratic score matrix. Banding the query rows is exact maths - the
softmax runs along the key axis and the final matmul reduces over that same axis -
so these tests compare the banded result against the single-shot expression with a
tight tolerance, and check the bidirectional layout against the two-call version.
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
        import GMFSS_Fortuna_arch as arch
    except Exception:  # pragma: no cover - cupy or the CUDA toolchain is absent
        CUDA_AVAILABLE = False

# Banding only changes matmul tiling, so the results agree to fp32 accumulation
# noise rather than exactly. 1e-4 relative is far tighter than anything the
# network could hide a real error in.
TOLERANCE = 1e-4


@unittest.skipUnless(CUDA_AVAILABLE, "needs the vendored architecture and cupy")
class GlobalSoftmaxAverageTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(4242)
        self.query = torch.randn(2, 40, 16, device="cuda", generator=generator)
        self.key = torch.randn(2, 40, 16, device="cuda", generator=generator)
        self.value = torch.randn(2, 40, 2, device="cuda", generator=generator)

    def reference(self):
        scale = self.query.shape[-1] ** 0.5
        scores = torch.matmul(self.query, self.key.transpose(-1, -2)) / scale
        return torch.matmul(torch.softmax(scores, dim=-1), self.value)

    def test_banded_matches_single_shot(self) -> None:
        expected = self.reference()
        for chunk in (1, 7, 13, 40):
            with self.subTest(chunk=chunk):
                measured = arch.global_softmax_average(
                    self.query, self.key, self.value, chunk
                )
                self.assertEqual(measured.shape, expected.shape)
                drift = (measured - expected).abs().max().item()
                self.assertLess(drift, TOLERANCE, f"chunk {chunk}: drift {drift:.3e}")

    def test_zero_and_oversized_chunk_take_the_single_shot_path(self) -> None:
        expected = self.reference()
        for chunk in (0, -5, 40, 4096):
            with self.subTest(chunk=chunk):
                measured = arch.global_softmax_average(
                    self.query, self.key, self.value, chunk
                )
                drift = (measured - expected).abs().max().item()
                self.assertLess(drift, TOLERANCE, f"chunk {chunk}: drift {drift:.3e}")

    def test_output_carries_the_value_channel_count(self) -> None:
        # The flow-attention call passes a 2-channel value against 16-channel
        # queries, which is the shape the helper has to keep supporting.
        measured = arch.global_softmax_average(self.query, self.key, self.value, 8)
        self.assertEqual(measured.shape, (2, 40, 2))


@unittest.skipUnless(CUDA_AVAILABLE, "needs the vendored architecture and cupy")
class GlobalCorrelationTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(99)
        self.feature0 = torch.randn(1, 16, 6, 8, device="cuda", generator=generator)
        self.feature1 = torch.randn(1, 16, 6, 8, device="cuda", generator=generator)

    def test_single_direction_matches_upstream(self) -> None:
        expected = arch.global_correlation_softmax(self.feature0, self.feature1, False)[0]
        measured = arch.global_correlation_softmax_chunked(
            self.feature0, self.feature1, False, chunk=10
        )
        self.assertEqual(measured.shape, expected.shape)
        drift = (measured - expected).abs().max().item()
        self.assertLess(drift, TOLERANCE, f"drift {drift:.3e}")

    def test_bidirectional_stacks_both_directions(self) -> None:
        forward = arch.global_correlation_softmax(self.feature0, self.feature1, False)[0]
        backward = arch.global_correlation_softmax(self.feature1, self.feature0, False)[0]

        measured = arch.global_correlation_softmax_chunked(
            self.feature0, self.feature1, True, chunk=10
        )
        self.assertEqual(measured.shape[0], 2)
        self.assertEqual(measured.shape[1:], forward.shape[1:])

        # The first half is the forward flow, the second half the backward one.
        drift = (measured[0:1] - forward).abs().max().item()
        self.assertLess(drift, TOLERANCE, f"forward drift {drift:.3e}")
        drift = (measured[1:2] - backward).abs().max().item()
        self.assertLess(drift, TOLERANCE, f"backward drift {drift:.3e}")

    def test_rejects_a_negative_chunk(self) -> None:
        with self.assertRaises(ValueError):
            arch.global_correlation_softmax_chunked(
                self.feature0, self.feature1, False, chunk=-1
            )


if __name__ == "__main__":
    unittest.main()
