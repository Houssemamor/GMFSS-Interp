"""Tests for the frame analysis used by dedup and scene detection.

The block-mean fast path replaced a per-block Python loop, so these tests pin
it against a literal reference implementation, including the degenerate frame
sizes where some grid cells are empty.
"""

from __future__ import annotations

import unittest

import numpy as np

from src import dedup

GRID = dedup.GRID


def reference_grid(frame: np.ndarray) -> np.ndarray:
    """The original per-block loop, kept verbatim as the reference oracle."""
    luma = frame.astype(np.float32) @ dedup._LUMA_WEIGHTS
    height, width = luma.shape
    row_edges = np.linspace(0, height, GRID + 1).astype(np.int64)
    column_edges = np.linspace(0, width, GRID + 1).astype(np.int64)

    grid = np.empty((GRID, GRID), dtype=np.float32)
    for row in range(GRID):
        row_slice = luma[row_edges[row]:row_edges[row + 1]]
        for column in range(GRID):
            block = row_slice[:, column_edges[column]:column_edges[column + 1]]
            grid[row, column] = float(block.mean()) if block.size else 0.0
    return grid


def frame(height: int, width: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


class AnalyseTests(unittest.TestCase):
    def test_matches_reference_on_common_frame_sizes(self) -> None:
        for height, width in ((1080, 1920), (720, 1280), (360, 640), (64, 64), (480, 64)):
            with self.subTest(size=(height, width)):
                sample = frame(height, width)
                expected = reference_grid(sample)
                actual = dedup.analyse(sample).luma_grid
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.dtype, np.float32)
                # float64 accumulation versus float32 pairwise sums: agreement
                # to ~1e-4 levels cannot move a 6-level (diff) or 76-level
                # (scene) threshold.
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3)

    def test_empty_cells_are_zero_not_nan(self) -> None:
        # Fewer pixels than grid cells: some blocks have no pixels at all and
        # must contribute 0.0 rather than a division by zero.
        sample = frame(20, 30)
        actual = dedup.analyse(sample).luma_grid
        expected = reference_grid(sample)
        self.assertFalse(np.isnan(actual).any())
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3)
        self.assertTrue((actual == 0.0).any())

    def test_uniform_frame_has_uniform_grid(self) -> None:
        sample = np.full((120, 160, 3), 128, dtype=np.uint8)
        grid = dedup.analyse(sample).luma_grid
        np.testing.assert_allclose(grid, 128.0, atol=1e-4)


class DecisionTests(unittest.TestCase):
    def test_exact_mode_identifies_identical_frames(self) -> None:
        sample = frame(64, 64)
        stats = dedup.analyse(sample)
        self.assertTrue(
            dedup.is_duplicate(sample, sample.copy(), stats, stats, "exact", 6.0, 0.0005)
        )
        changed = sample.copy()
        changed[0, 0, 0] ^= 0xFF
        self.assertFalse(
            dedup.is_duplicate(sample, changed, stats, stats, "exact", 6.0, 0.0005)
        )

    def test_exact_mode_does_not_need_stats(self) -> None:
        sample = frame(64, 64)
        self.assertTrue(
            dedup.is_duplicate(sample, sample.copy(), None, None, "exact", 6.0, 0.0005)
        )
        self.assertFalse(dedup.is_duplicate(sample, sample, None, None, "off", 6.0, 0.0005))

    def test_diff_mode_requires_stats(self) -> None:
        sample = frame(64, 64)
        stats = dedup.analyse(sample)
        with self.assertRaises(ValueError):
            dedup.is_duplicate(sample, sample.copy(), None, stats, "diff", 6.0, 0.0005)
        with self.assertRaises(ValueError):
            dedup.scene_difference(None, stats)

    def test_scene_difference_scales_to_unit_range(self) -> None:
        black = dedup.analyse(np.zeros((64, 64, 3), dtype=np.uint8))
        white = dedup.analyse(np.full((64, 64, 3), 255, dtype=np.uint8))
        self.assertAlmostEqual(dedup.scene_difference(black, black), 0.0, places=6)
        self.assertAlmostEqual(dedup.scene_difference(black, white), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
