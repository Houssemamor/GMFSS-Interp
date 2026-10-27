"""Per-frame analysis used for duplicate skipping and hard-cut detection.

Anime is drawn on twos or threes, so a large share of consecutive frames are
literally identical. Skipping those pairs saves a lot of GPU time. The default
mode therefore compares decoded frames byte for byte, which is exact: the same
source frame always decodes to the same bytes.

A looser "diff" mode exists for sources that were re-encoded in a way that
perturbs identical frames. It is opt-in because a coarse block comparison can
also hide small motion (a blinking eye, a moving mouth) and freeze it.

Scene detection compares block luma averages, which is stable under motion blur
and compression noise but reacts strongly to an actual cut.
"""

from __future__ import annotations

import dataclasses

import numpy as np

# Luma grid resolution. 64x64 is fine enough to localise a cut and coarse enough
# that computing it costs a fraction of a millisecond.
GRID = 64

# Rec.601 luma weights. Exact coefficients do not matter here: values are only
# compared against each other.
_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


@dataclasses.dataclass(frozen=True)
class FrameStats:
    """Block luma map for one frame, in 0..255."""

    luma_grid: np.ndarray


def analyse(frame: np.ndarray) -> FrameStats:
    """Build the luma map for one RGB24 frame.

    Args:
        frame: uint8 array of shape (height, width, 3).

    Returns:
        FrameStats holding a GRID x GRID block-mean luma map.
    """
    luma = frame.astype(np.float32) @ _LUMA_WEIGHTS
    height, width = luma.shape
    row_edges = np.linspace(0, height, GRID + 1).astype(np.int64)
    column_edges = np.linspace(0, width, GRID + 1).astype(np.int64)

    grid = np.empty((GRID, GRID), dtype=np.float32)
    for row in range(GRID):
        row_slice = luma[row_edges[row]:row_edges[row + 1]]
        for column in range(GRID):
            block = row_slice[:, column_edges[column]:column_edges[column + 1]]
            grid[row, column] = float(block.mean()) if block.size else 0.0
    return FrameStats(luma_grid=grid)


def changed_ratio(previous: FrameStats, current: FrameStats, luma_delta: float) -> float:
    """Fraction of blocks whose luma moved by more than ``luma_delta``.

    Compression noise shifts every block by a small amount, so a low delta with
    a small allowed ratio separates "re-encode noise" from "something moved".
    """
    changed = np.abs(previous.luma_grid - current.luma_grid) > luma_delta
    return float(changed.mean())


def is_duplicate(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    previous: FrameStats,
    current: FrameStats,
    mode: str,
    luma_delta: float,
    max_changed_ratio: float,
) -> bool:
    """Decide whether a pair should skip interpolation.

    Args:
        previous_frame: the earlier RGB24 frame.
        current_frame: the later RGB24 frame.
        previous: stats of the earlier frame.
        current: stats of the later frame.
        mode: "off", "exact" or "diff".
        luma_delta: per-block luma change counted as "changed" for "diff".
        max_changed_ratio: allowed fraction of changed blocks for "diff".

    Returns:
        True when the pair should not be interpolated.
    """
    if mode == "off":
        return False
    if mode == "exact":
        return bool(np.array_equal(previous_frame, current_frame))
    return changed_ratio(previous, current, luma_delta) <= max_changed_ratio


def scene_difference(previous: FrameStats, current: FrameStats) -> float:
    """Mean absolute luma change between two frames, normalised to 0..1."""
    return float(np.abs(previous.luma_grid - current.luma_grid).mean() / 255.0)
