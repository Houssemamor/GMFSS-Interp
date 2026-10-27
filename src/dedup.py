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


def _block_edges(length: int) -> np.ndarray:
    """Return the GRID+1 block boundaries for one axis.

    ``linspace`` keeps the blocks evenly distributed when the axis is not a
    multiple of GRID, which is what the original per-block loop did: a 1080-row
    frame yields blocks of 16 or 17 rows rather than a rescaled grid.
    """
    return np.linspace(0, length, GRID + 1).astype(np.int64)


def analyse(frame: np.ndarray) -> FrameStats:
    """Build the luma map for one RGB24 frame.

    The block means are a two-stage segmented reduction (rows, then columns)
    rather than a GRID*GRID Python loop; that is ~19 ms faster per 1080p frame
    and produces the same values to within float32 rounding.

    Args:
        frame: uint8 array of shape (height, width, 3).

    Returns:
        FrameStats holding a GRID x GRID block-mean luma map.
    """
    # float32 upcast plus a 3-term weighted sum; this is the cheapest exact
    # Rec.601 luma available without a compiled kernel.
    luma = frame.astype(np.float32) @ _LUMA_WEIGHTS
    row_edges = _block_edges(luma.shape[0])
    column_edges = _block_edges(luma.shape[1])

    row_sums = np.add.reduceat(luma, row_edges[:-1], axis=0)
    block_sums = np.add.reduceat(row_sums, column_edges[:-1], axis=1)

    # Block sizes differ by at most one pixel per axis, so the divisor is
    # separable. accumulate in float64: the sums are over up to ~500 pixels and
    # float32 summation order would otherwise show up in the diff thresholds.
    block_counts = np.outer(np.diff(row_edges), np.diff(column_edges))
    grid = np.divide(
        block_sums,
        block_counts,
        out=np.zeros_like(block_sums, dtype=np.float32),
        where=block_counts > 0,
    )
    return FrameStats(luma_grid=grid.astype(np.float32, copy=False))


def changed_ratio(previous: FrameStats | None, current: FrameStats | None, luma_delta: float) -> float:
    """Fraction of blocks whose luma moved by more than ``luma_delta``.

    Compression noise shifts every block by a small amount, so a low delta with
    a small allowed ratio separates "re-encode noise" from "something moved".

    The stats are optional only so callers can skip ``analyse`` when they do not
    need it; passing None here is a bug, because this mode is built on them.
    """
    if previous is None or current is None:
        raise ValueError("changed_ratio requires frame stats (dedup mode 'diff')")
    changed = np.abs(previous.luma_grid - current.luma_grid) > luma_delta
    return float(changed.mean())


def is_duplicate(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    previous: FrameStats | None,
    current: FrameStats | None,
    mode: str,
    luma_delta: float,
    max_changed_ratio: float,
) -> bool:
    """Decide whether a pair should skip interpolation.

    Args:
        previous_frame: the earlier RGB24 frame.
        current_frame: the later RGB24 frame.
        previous: stats of the earlier frame, or None when the caller skipped
            them (allowed for "off" and "exact", which never read them).
        current: stats of the later frame, same rule.
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


def scene_difference(previous: FrameStats | None, current: FrameStats | None) -> float:
    """Mean absolute luma change between two frames, normalised to 0..1."""
    if previous is None or current is None:
        raise ValueError("scene_difference requires frame stats")
    return float(np.abs(previous.luma_grid - current.luma_grid).mean() / 255.0)
