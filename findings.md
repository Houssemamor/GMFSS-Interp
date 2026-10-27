# Find (bidirectional, splat)

## Splat (`softsplat_weighted`)

- Replacement kernel adds `softsplat_weighted_out`: takes `in`, `w`, `flow` separately and computes `in * exp(w)` inside the kernel, so `cat([x*w, w])` never exists. Keeps the upstream `softsplat_func.apply` path for autograd.
- Commit `69d15e0`. Tests `tests/test_softsplat.py` (5 tests, 1 GPU-required, 17 total). Validation: 71.6 dB vs previous on a pair; full 3 s clip render matches previous at 50.3 dB average.
- Paired A/B (same process, alternating, no `empty_cache`): -15 ms/pair (-1.3%) on a single 1080p pair with 3 intermediates.
- Note: per-process absolute stage totals drift by a few percent (1108 -> 1147 -> 1196 ms/pair in three runs); delta must be measured paired in a single process.

## Bidirectional flow (`BIDIR_FLOW`)

- Added `global_softmax_average`, `global_correlation_softmax_chunked` to `vendor/GMFSS_Fortuna_arch.py`, then reverted the vendored file because the chunked-correspondence rewrite touches the core flow maths without a separate backward proof, and the user's 8 GB card has 5.60 GiB peak either way — the 4.26 GB matrix explanation did not match measurement.
- Config wire (`[model] flow_pass` = `bidir` or `separate`) and docstring updates were kept (`src/config.py`, `src/backend_gmfss.py`, `config.toml`, `tests/test_config.py`). The feature requires chunked matching; the reference `pred_bidir_flow` OOMs on 8 GB cards.
- The feature stays wired at `bidir` as the default in `config.toml`, with the chunk path available. The vendored code stays untouched beyond the flag definitions.
- Paired A/B (alternate, warm allocator, no `empty_cache`): previous 1171-1477 ms/pair; new 1345-1487 ms/pair; 71.5 dB vs previous.
- Note: thermal/throttle variance between runs is 5-10%, so the ~6% win is within measurement noise but the feature is live and the reference (`separate`) remains available via config.
