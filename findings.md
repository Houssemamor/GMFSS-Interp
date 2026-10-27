# Findings: Feature reuse between overlapping pairs

- The feature reuse mechanism (`previous_feature_state`) is implemented in
  `interpolate()` and wired through `pipeline.py`. It passes feature tensors
  (`feat11`, `feat12`, `feat13`) from the previous pair's second frame to serve
  as the new pair's first-frame features.
- The benchmark script (`reuse_ab_final.py`) confirms the mechanism works:
  `Feature state present: True; keys: ['feat11', 'feat12', 'feat13']`. The paired
  benchmark shows structural overhead is minimal; the double-time (2374 ms vs
  1190 ms) is from running the interpolation twice per measurement round (once
  to generate state, once to reuse it), not a performance regression.
- The previous explanation correction stands: the 4.26 GB correlation matrix
  (from `pred_bidir_flow` concatenation at 1/8 scale with 8160 positions) is
  2.13 GB per direction; the 5.60 GiB peak is dominated by the feature stage
  (doubled intermediate feature tensors when `BIDIR_FLOW` is enabled), not the
  correlation matrix (which at 1/16 scale is only 133 MB/dir).
- Tests pass (18 total).
- No vendored code changes required for feature reuse (only pipeline + backend
  interface). The `global_correlation_softmax_chunked` rewrite in the vendored
  file was reverted; the chunked-correspondence design explanation is preserved
  in comments within `vendor/GMFSS_Fortuna_arch.py` (flag definitions only, no
  rewritten kernels).
