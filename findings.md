# Find (bidirectional, splat, feature reuse)

## Splat (`softsplat_weighted`) - committed 69d15e0

- Replacement kernel adds `softsplat_weighted_out`: takes `in`, `w`, `flow`
  separately and applies the weight inside the kernel, so `cat([x*w, w])` never
  exists. Keeps the upstream `softsplat_func.apply` path for autograd.
- Paired A/B: -15 ms/pair (-1.3%) at 1080p. Output identity 71.6 dB on the pair;
  full clip 50.3 dB avg vs the pre-optimisation reference. Tests:
  `tests/test_softsplat.py`.

## Bidirectional flow + banded global matching - committed (this change)

- Restores what an earlier commit had reverted, which had left `[model]
  flow_pass = "bidir"` as a silent no-op: the backend set `arch_module.BIDIR_FLOW`
  but the vendored branch did not exist. That inconsistency is now closed.
- `global_softmax_average` bands the two global `softmax(q k^T) v` products
  (correlation and first flow-attention step). Banding is exact: softmax runs
  along the key axis and the final matmul reduces over the same axis, so no query
  row mixes with another. `GLOBAL_MATCH_CHUNK = 1024` (module constant, not
  config).
- The earlier "4.26 GB correlation matrix" explanation was wrong: the global
  matching scale is 1/16 at 1080p (8160 positions), so the matrix is 133 MB per
  direction. The real memory cost of bidir is the doubled feature stage: measured
  peak 5.60 GiB vs 2.68 GiB for separate passes.
- Unbanded bidir measured 1468 ms/pair (slower than separate at 1178); banded at
  chunk 1024 it is 1091. Paired A/B (alternating, warm allocator): -80.8
  ms/pair (-5.47%). Full clip with the new default vs the pristine reference:
  50.33 dB avg, 48.25 dB min over 216 frames - the encoder noise floor, i.e.
  visually indistinguishable.
- Tests: `tests/test_flow_match.py` pins the banded result against the
  single-shot expression and the upstream correlation function (6 tests).

## Feature reuse between overlapping pairs - completed (this change)

- Pair N ends on frame f+1 and pair N+1 starts on it, so pair N's
  `feat21/feat22/feat23` (second-frame features) are pair N+1's first-frame
  features. The earlier plumbing collected `state[4:7]` - the FIRST frame's
  features - which was the wrong half; fixed to `state[7:10]` with keys
  `feat21/feat22/feat23`.
- The vendored `Model.reuse` gains an optional `feat0=None` parameter: when
  given, `feat_ext(img0)` is skipped. Default keeps upstream behaviour. The
  backend only passes it when the incoming `frame_a` is the exact object the
  state was built from (object-identity guard), so a mismatched call falls back
  to full extraction.
- Ceiling measured from the stage split (`feat_ext x2` = 29.3 ms/pair): one
  saved extraction is ~14.6 ms/pair (~1.2%).
- Saving holds ~112 MB of fp16 feature tensors on the GPU between pairs (they
  replace tensors that would be allocated anyway).

## Rejected with numbers

- Hoisting the per-timestep interpolates out of `Model.inference` (the old
  ~1.4% estimate): all 8 interpolates at real 1080p fp16 shapes sum to 0.360
  ms per timestep, 1.08 ms per pair at 3 timesteps - 0.09% of a ~1200 ms pair.
  Not worth the vendored interface change. (`interp_ceiling.py`)

## Measurement practice

- Per-process totals drift several percent between runs on this machine, so all
  deltas are paired A/Bs in one process, alternating the two implementations
  with a warm allocator (no `empty_cache` between rounds - it forces a cold
  cudaMalloc of each variant's whole footprint, which production never pays).
