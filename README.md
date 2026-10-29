# GMFSS-Interp

Frame interpolation for anime, built around **GMFSS Fortuna** - the model that
currently produces the fewest artefacts on animation. Files are dropped onto a
batch file; everything is configured in one TOML file beside it.

Designed for the case this project was built for: a 23.976 fps Blu-ray episode
with hard action cuts, interpolated to 119.88 fps, with the original soft
subtitles, fonts and chapter markers carried through untouched.

---

## Why GMFSS and not RIFE

RIFE is a general-purpose interpolator. On live action it is excellent and very
fast. On animation it has two recurring failures:

* **Line wobble** - straight drawn lines bend and ripple between frames.
* **Warpping on flat colour** - large areas of single colour "boil" because the
  model has no texture to lock onto.

GMFSS (Generalized Multi-frame Frame Synthesis) was trained with an explicit
optical-flow/metric stage aimed at animation and a feature-fusion stage that
reconstructs line art instead of smearing it. It is substantially slower than
RIFE, and it is the right trade for this content.

> RIFE 4.25 and newer remain available inside Flowframes/Waifu2x-Extension-GUI
> if speed matters more than quality.

---

## Comparison

![Original at 23.976 fps (left) next to GMFSS 3x at 71.928 fps (right)](docs/comparison.webp)

**Left:** the original, 23.976 fps. **Right:** the same four seconds interpolated
3x to 71.928 fps. Both sides are shown at the same playback rate, so the left
side steps visibly while the right side glides.

| | |
| --- | --- |
| Clip | 4 s of 1080p anime, a fight scene with fast character and effect motion |
| Settings | the defaults in `config.toml`: `factor = 3`, `precision = "fp16"`, `flow_pass = "bidir"`, HEVC NVENC quality 18 |
| Render time | 2.4 min for 94 pairs (about 1.5 s per pair on this machine) |
| Preview | downscaled to 480x270 per side and compressed as animated WebP, so the preview is softer than the real output |

### Slow motion

![Same clip at one sixth speed: original (left) and GMFSS 3x (right)](docs/comparison_slowmo.webp)

The same clip at one sixth speed (24 s). The left side plays at about 4 fps and
the right side at about 12 fps, well under browser frame-rate caps. Each
original frame is held for three frames on the left, while the right shows the
two extra frames GMFSS synthesised in between. This is
the view to use for judging interpolation artefacts such as ghosting around
fast-moving hands and effects.

The animation is a preview of motion smoothness, not of picture quality. Some
browsers and image viewers cap animated playback below 72 fps, which makes the
difference look smaller than it is. Playing the actual output file in a video
player shows it properly.

---

## Requirements

| Item | Detail |
| --- | --- |
| OS | Windows 10/11, 64-bit |
| GPU | NVIDIA (CUDA). 8 GB VRAM is enough for 1080p |
| Disk | ~4 GB for PyTorch and the CUDA headers, ~50 MB for weights, plus room for output |
| ffmpeg | Installed automatically by `setup.bat`, or `ffmpeg` + `ffprobe` on `PATH` |
| mkvmerge | Optional. A portable MKVToolNix folder in `tools/mkv/` gives the most faithful copy of fonts and chapters; without it the pipeline muxes with ffmpeg |

Python is handled by `setup.bat`; you do not need a system Python.

---

## Install

Double-click **`setup.bat`** once. It will:

1. Fetch a static ffmpeg build into `tools/` if none is on `PATH`.
2. Create a Python 3.12 virtual environment in `.venv`.
3. Install PyTorch with the CUDA 12.6 runtime (~2.4 GB download).
4. Install `numpy` and `cupy-cuda12x[ctk]` (~1.5 GB, includes the CUDA headers
   that cupy needs to compile its kernels).
5. Download the GMFSS checkpoints into `models/`.
6. Verify that torch and cupy can both see the GPU.

If a step fails the script stops and says which one. Re-running it is safe:
existing environments, packages and weights are reused.

**Optional: mkvmerge.** Download the portable MKVToolNix zip from
<https://mkvtoolnix.download/downloads.html>, extract it to `tools/mkv/`, and
leave `[mux] muxer = "auto"`. The pipeline then uses `tools/mkv/mkvmerge.exe`
for the final mux, which copies fonts, chapters and track metadata verbatim.
Without it, ffmpeg does the mux; output is still correct, but chapters land a
few milliseconds off and attachment handling is less predictable.

---

## Use

1. Open **`config.toml`** and set the frame rate, dedup mode and encoder.
   This is the only file you normally touch.
2. Drag one or more video files onto **`Interpolate.bat`**.

A console window shows the resolved plan, a progress line every five seconds,
and a summary at the end. The progress line carries both the countdown and the
clock time the run should finish, so a long render can be planned around:

```
progress:  36.3%  pairs 33/91    3.2 pair/s  ETA   0.3 min  finish ~01:37
```

The output is written next to the source:

```
episode01.mkv  ->  episode01_gmfss_5x.mkv
```

Press `Ctrl+C` to abort; the partial file is deleted.

### Command line (optional)

```bat
.venv\Scripts\python.exe -m src.main --config config.toml "D:\video\ep01.mkv"
.venv\Scripts\python.exe -m src.main --config config.toml --dry-run "D:\video\ep01.mkv"
```

`--dry-run` prints the plan - rate, tracks, model, output path - without
touching the GPU.

---

## Configuration

All keys live in `config.toml`. Values are validated before any GPU work
starts; an invalid key produces a message naming that key.

### `[rate]` - how many frames to synthesise

| Key | Meaning |
| --- | --- |
| `mode` | `"factor"` multiplies the source rate. `"fps"` targets an absolute rate. |
| `factor` | Multiplier for `mode = "factor"`. 2-16. |
| `target_fps` | Target for `mode = "fps"`. Must land on a whole multiple of the source rate within 2%, otherwise the run is rejected rather than silently drifting. |

Examples:

```toml
# 23.976 -> 119.88 fps
mode = "factor"
factor = 5
```

```toml
# same result, expressed as a target
mode = "fps"
target_fps = 119.88
```

### `[model]` - which network

| Key | Meaning |
| --- | --- |
| `precision` | `"fp32"` (safe, default) or `"fp16"` (autocast; about 45% faster at 1080p, slightly lower VRAM). |
| `gpu_index` | Which GPU to use when more than one is present. |
| `attention` | `"sdpa"` (default) uses PyTorch's fused attention inside GMFlow; `"reference"` runs the original matmul/softmax code. Same result up to rounding, ~9% slower. |
| `fusion_layout` | `"channels_last"` (default) runs the fusion network on cudnn's NHWC path, ~3% faster at 1080p; `"nchw"` keeps the original layout. The two differ by at most one 8-bit level. |
| `flow_pass` | `"bidir"` (default) predicts both flow directions in one pass, ~5% faster as a paired A/B; `"separate"` runs the network twice and keeps the original peak (2.7 vs 5.6 GiB at 1080p). Same flow up to rounding. |

### `[dedup]` - do not waste time on repeated frames

Anime is drawn on twos or threes, so a large share of consecutive frames are
identical. Interpolating them produces nothing.

| Key | Meaning |
| --- | --- |
| `mode` | `"exact"` (default; byte-identical decoded frames), `"off"`, or `"diff"` (near-identical, tolerates re-encode noise). |
| `changed_luma` | `"diff"` only. Per-block luma change that counts as a real change. |
| `max_changed_ratio` | `"diff"` only. Fraction of the 64x64 block map allowed to change and still count as a duplicate. |

`"exact"` is the default because it is both safe and effective: the same source
frame always decodes to the same bytes, so twos and threes are caught with no
risk of hiding real motion. `"diff"` is opt-in; a coarse block comparison can
also swallow a small movement such as a blink, which then freezes on screen.

### `[scene]` - hard cuts

| Key | Meaning |
| --- | --- |
| `enabled` | Detect cuts and hold the frame instead of blending across them. |
| `threshold` | Mean absolute luma change, 0-1. Lower is more sensitive; `0.25`-`0.35` suits most anime. |

Without this, every cut in an action scene gets a ghost frame - the single most
visible artefact in interpolated anime.

### `[encode]` - output

| Key | Meaning |
| --- | --- |
| `encoder` | Any ffmpeg encoder. `hevc_nvenc` / `h264_nvenc` (hardware), `libx265` / `libx264` (software). |
| `quality` | CRF for software encoders, constant-quality value for NVENC. Lower is better; 16-20 is transparent for most sources. |
| `pixel_format` | `yuv420p10le` for 10-bit sources, `yuv420p` for 8-bit. |
| `container` | `mkv` (keeps ASS subtitles and attachments) or `mp4`. |

### `[mux]` - how the output is assembled

| Key | Meaning |
| --- | --- |
| `muxer` | `auto` uses mkvmerge when it is installed and ffmpeg otherwise. `ffmpeg` forces a single pass. `mkvmerge` requires the tool and `container = "mkv"`. |
| `preserve_audio` | Copy all audio tracks without re-encoding. |
| `preserve_subtitles` | Copy subtitle tracks, including ASS styling. |
| `preserve_attachments` | Copy fonts referenced by ASS subtitles. |
| `preserve_chapters` | Copy chapter markers. |

Interpolation changes the frame count, not the duration, so all timestamps in
the copied tracks remain valid without retiming.

mkvmerge is preferred because font attachments are the weakest part of ffmpeg's
Matroska muxing: a missing font does not fail the render, the signs just render
in the wrong typeface. With `muxer = "auto"` ffmpeg writes the video stream to a
temporary file and mkvmerge joins it with every remaining element of the source,
so subtitles, fonts, chapters and track names are copied rather than rebuilt.

### `[paths]`

| Key | Meaning |
| --- | --- |
| `ffmpeg` | `"auto"` searches `PATH`; otherwise a full path to `ffmpeg.exe`. `ffprobe` is looked for beside it. |
| `mkvmerge` | `"auto"` looks in `tools/mkv/`, then `tools/`, then `PATH`. Set a full path to override. |
| `cache_dir` | Where decoder/encoder/mux logs are written. Relative to the project folder. |

---

## How it works

```
source.mkv
   |
   +-- ffprobe ---> rate, geometry, which tracks exist
   |
   +-- ffmpeg (decode, forced CFR) --> RGB24 frames on a pipe
                                        |
                          duplicate / scene-cut check
                                        |
                            GMFSS Fortuna (one feature pass
                            per pair, one fusion pass per
                            synthesised frame)
                                        |
   +-- ffmpeg (encode) <----------------+
   |
   +-- mkvmerge: encoded video + every track, font and chapter of the source
   |
output.mkv
```

Nothing is written to disk except the final file: a 24 minute 1080p episode is
about 34,000 frames, which as PNG would be tens of gigabytes.

---

## Performance

Measured on this machine: RTX 3060 8 GB, i5-12400F, Windows 11, torch 2.14
(CUDA 12.6), idle desktop, best of four runs. 1080p is real content from the
sample episode; the smaller resolutions are the same frames decimated, since
inference cost tracks pixel count rather than content.

| Resolution | Intermediates | Precision | s/pair | Output frames/s |
| --- | --- | --- | --- | --- |
| 1920x1080 | 4 (5x) | fp16 | 1.3 | 3.8 |
| 1920x1080 | 4 (5x) | fp32 | 2.4 | 2.1 |
| 1920x1080 | 2 (3x) | fp16 | 1.0 | 2.7 |
| 1280x720 | 4 (5x) | fp16 | 0.6 | 8.5 |
| 640x360 | 4 (5x) | fp16 | 0.16 | 32 |

A 24 minute 1080p episode at 23.976 fps is about 34,000 pairs. At 1.3 s/pair
that is roughly **12 hours** of pure inference at 5x before dedup and before
encoding; `precision = "fp16"` and exact-frame dedup are what make this
practical. Expect a full episode to run overnight, not in an hour.

Where the time goes at 1080p fp16, per pair with 3 timesteps, under the current
defaults (`flow_pass = "bidir"`, `channels_last`, sdpa attention, the inline
weight splat kernel, overlap feature reuse):

| Stage | ms/pair | Share |
| --- | --- | --- |
| GMFlow, one bidirectional pass | 888 | 57% |
| Fusion network (GridNet), 3 passes + copy | 539 | 34% |
| Splatting (30 softsplat calls) | 88 | 6% |
| Extraction: 1 feature pass + both metrics | 49 | 3% |

Per-process totals on this machine drift by several percent between runs
(easily 10-15% between a cold and a warm GPU), so the table above is one run
and the numbers in this section are best-observed. Shares are much more stable
than totals. Any change worth reporting here was measured as a paired A/B in a
single process, alternating the two implementations, because that removes the
drift.

The GPU is the limit, not the Python loop. In the run above the instrumented
stage sum lands within 3% of the same process's uninstrumented pair median,
and each timestep ends in a device-to-host copy by design. There is no idle gap
for CUDA graphs to remove (tried; no gain). That is why this project optimises
kernels and layouts rather than rewriting the driver in another language.

Practical notes:

* Dedup is the single biggest saving on TV anime - many shows repeat 30-50% of
  frames, and those pairs cost nothing.
* `precision = "fp16"` measured ~45% faster at 1080p with no visible difference
  on the test material; try a short clip both ways before committing a series.
* If a run dies with a CUDA out-of-memory error, set `precision = "fp16"` first,
  then close other GPU applications. 8 GB is the practical floor at 1080p.
* Scene detection costs almost nothing and removes the worst artefacts; keep it on.
* Optional headroom on 8 GB cards: `flow_pass = "bidir"` is on by default and
  measured ~5% faster as a paired A/B. Its cost is memory, not a temporary
  matrix: doubling the flow batch takes the measured peak from 2.7 to 5.6 GiB
  on a 1080p pair. If a run runs close to the VRAM limit, set
  `flow_pass = "separate"`.
* Feature reuse is automatic, no config: consecutive pairs overlap by one
  frame, and the overlapping frame's feature extraction is carried over
  (~1.2% measured at 1080p). A skipped duplicate pair or a scene cut breaks
  the chain by itself, and the next pair simply extracts from scratch.

---

## Troubleshooting

**`ffmpeg was not found on PATH`**
`setup.bat` normally installs one into `tools/`. If that download failed,
install ffmpeg manually (ffmpeg + ffprobe), add its `bin` folder to `PATH`, or
set `[paths] ffmpeg` to the full path of `ffmpeg.exe`.

**`Failed to find CUDA headers`**
cupy compiles its kernels with NVRTC and needs the CUDA headers, which are not
part of the driver:

```bat
.venv\Scripts\python.exe -m pip install "cupy-cuda12x[ctk]"
```

**`The cupy softsplat backend failed to initialise`**
Either the driver is too old for the installed CUDA runtime, or the CUDA headers
are missing. Install them:

```bat
.venv\Scripts\python.exe -m pip install "cupy-cuda12x[ctk]"
```

There is no second softsplat backend in this project: the taichi variant used by
the upstream ComfyUI package drives every call through a worker process and
deadlocks when used from a plain script.

**`CUDA is not available to torch`**
The installed torch is the CPU build. Reinstall it from the CUDA index:

```bat
.venv\Scripts\python.exe -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

**`is not a whole multiple of the source rate`**
You used `mode = "fps"`. Either set `mode = "factor"` or use a target that is a
whole multiple (119.88 for 23.976).

**`mkvmerge exited with code 2`**
mkvmerge could not assemble the output; `cache/mux.log` holds the reason. If the
message names an attachment or chapter file, the source is unusual - set
`muxer = "ffmpeg"` to bypass mkvmerge for that file.

**`mux.muxer = "mkvmerge" but mkvmerge was not found`**
Either drop a portable MKVToolNix folder into `tools/mkv/`, add `mkvmerge.exe`
to `PATH`, set `[paths] mkvmerge`, or use the default `muxer = "auto"`.

**Output has ghost frames at cuts**
Lower `[scene] threshold` (try `0.20`). If the show has very soft dissolves
that are being mistaken for cuts, raise it instead.

**Output looks soft / lines shimmer**
Set `[dedup] mode = "off"` and compare a short clip. If the shimmer disappears,
dedup was skipping real motion; keep the safer `"exact"` mode.

---

## Project layout

```
GMFSS-Interp/
  Interpolate.bat        drag-and-drop launcher
  setup.bat              one-time installer
  config.toml            all settings
  requirements.txt       dependency notes (torch installed separately)
  src/
    main.py              CLI and error reporting
    config.py            TOML loading and validation
    probe.py             ffprobe wrapper
    ffmpeg.py            tool resolution, command construction
    mux.py               final mux with mkvmerge
    dedup.py             duplicate and scene-cut detection
    backend_gmfss.py     model loading and inference
    pipeline.py          streaming decode -> interpolate -> encode
    download_models.py   checkpoint downloader
  tests/                 unit tests (no GPU needed)
  docs/                  comparison animations used by this README
  vendor/                GMFSS architecture + softsplat operators (MIT)
  models/                checkpoints (downloaded)
  tools/                 ffmpeg + ffprobe (bundled by setup.bat), optional mkv/
  cache/                 decoder, encoder and mux logs
```

`vendor/` contains a pinned copy of the GMFSS architecture and the softsplat
operators from
[ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)
(MIT, Copyright (c) 2023 Fannovel16), with a small standalone shim replacing the
ComfyUI device helpers. It is vendored rather than installed so a ComfyUI update
cannot silently change model behaviour.

Local changes to the vendored copy. The first four are cosmetic or packaging,
the rest are performance changes; three of them have a matching switch in
`[model]`:

* `comfy/model_management.py` is a three-function stand-in for the ComfyUI
  device helpers.
* `vfi_models/ops/__init__.py` exposes only `softsplat` and `init` instead of
  re-exporting every operator the upstream package supports.
* Three regexes in `cupy_ops/utils.py` use raw strings, which silences Python
  3.12 `SyntaxWarning`s without changing what they match.
* `taichi_ops/` is removed. The taichi backend drives each call through a worker
  process and deadlocks when used from a plain script.
* `single_head_split_window_attention` can dispatch to
  `torch.nn.functional.scaled_dot_product_attention` instead of the hand-written
  matmul/softmax/matmul, selected by `[model] attention`. It reuses the same
  window split/merge and the same additive mask; the savings come from the
  fused kernel not materialising every window's full score matrix.
* `Model.device` converts the GridNet weights to `channels_last` when
  `[model] fusion_layout` asks for it, and `Model.inference` passes NHWC inputs
  to match. The splat operators stay NCHW because their CUDA kernels are, so the
  layout changes only at the fusion boundary.
* `cupy_ops/softsplat.py` gains a `softsplat_weighted` kernel, used by
  `softsplat()` for the weighted modes when grad is off. The upstream code
  materialises `cat([x*w, w])` so the scatter can read one tensor; the new
  kernel takes `x`, `w` and the flow separately and multiplies inline, which
  removes two full passes over the feature maps from every call (~1.3%
  end-to-end at 1080p, paired A/B). The concatenated formulation is kept for
  autograd, because the vendored backward kernels are written against it, and
  `tests/test_softsplat.py` pins the two against each other.
* The two global `softmax(q k^T) v` products in GMFlow (the correlation and the
  first flow-attention step) go through `global_softmax_average`, which computes
  the result in bands of query rows when `GLOBAL_MATCH_CHUNK` is set. The softmax
  runs along the key axis and the final matmul reduces over the same axis, so
  banding is exact; it bounds the temporary by `chunk * L` instead of `L * L`.
  `tests/test_flow_match.py` pins the banded result against the single-shot
  expression and the upstream correlation function.
* `Model.reuse` can predict both flow directions in one pass (`BIDIR_FLOW`, set
  from `[model] flow_pass`), so the feature extractor and the first transformer
  run once instead of twice (~5% end-to-end at 1080p, paired A/B). The
  bidirectional correlation matrix is built by
  `global_correlation_softmax_chunked`, which stacks the transposed matrix for
  the backward direction; the doubled batch is why the flow stage peaks at
  5.6 GiB instead of 2.7 GiB.
* `Model.reuse` accepts an optional precomputed `feat0`, and the backend passes
  the previous pair's second-frame feature tensors there, so the overlapping
  frame is extracted once per two pairs instead of once per pair (~1.2%
  measured). The backend reuses the state only when its stored frame is the
  exact object passed as `frame_a` - identity implies identical pixels because
  the decoder hands out fresh arrays - so duplicate skips and scene cuts fall
  back to a full extraction on their own. `tests/test_reuse_feat.py` pins the
  reused path against a fresh extraction, and the stale-features case against
  being silently swallowed.

## Credits and licences

* GMFSS Fortuna architecture and weights:
  [98mxr/GMFSS_Fortuna](https://github.com/98mxr/GMFSS_Fortuna).
* Packaging and operator implementations:
  [ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) (MIT).
* Checkpoint mirror: the public
  [VSGAN-tensorrt-docker model release](https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases).

The model weights are distributed by their authors for research and personal
use. Check their terms before using this pipeline for anything commercial.
