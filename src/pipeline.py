"""Pipeline orchestration: decode, interpolate, encode, mux.

Frames are streamed through pipes rather than written to disk. A 24 minute
1080p episode is roughly 34,000 frames; as PNG that would be tens of gigabytes,
so the decoder, the GPU loop and the encoder all run at the same time and only
two source frames are ever held in memory.

The encode writes to a temporary file that is renamed only after ffmpeg exits
cleanly, so an interrupted run never leaves a half-written file in place of a
valid one.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import pathlib
import subprocess
import time
from collections.abc import Callable
from fractions import Fraction

import numpy as np

from . import backend_gmfss
from . import config as config_module
from . import dedup as dedup_module
from . import ffmpeg as ffmpeg_module
from . import mux as mux_module
from . import probe as probe_module

# How often the progress line is refreshed, in seconds.
PROGRESS_INTERVAL = 5.0

# Read size for the decode/encode pipes. Large enough that the FFmpeg processes
# are not woken for every few hundred kilobytes.
PIPE_BUFFER_BYTES = 8 * 1024 * 1024


class PipelineError(RuntimeError):
    """Raised when a stage of the pipeline fails."""


class EncoderClosedError(PipelineError):
    """Raised when the encoder's stdin closes while frames are still being sent.

    The real cause (bad pixel format, missing encoder, no GPU session) is only
    in the encoder log, so ``run`` replaces this with a message quoting it.
    """


@dataclasses.dataclass(frozen=True)
class Plan:
    """Everything decided before the first frame is touched."""

    info: probe_module.MediaInfo
    tools: ffmpeg_module.Tools
    factor: int
    output_rate: Fraction
    output_path: pathlib.Path
    timesteps: tuple[float, ...]
    mux_via: str

    @property
    def source_rate(self) -> Fraction:
        return self.info.video.frame_rate


@dataclasses.dataclass
class RunStats:
    """Counters reported at the end of a run."""

    pairs: int = 0
    duplicate_pairs: int = 0
    scene_cuts: int = 0
    frames_written: int = 0


def _output_path(config: config_module.Config, info: probe_module.MediaInfo, factor: int) -> pathlib.Path:
    """Choose the output name next to the source file."""
    return info.path.with_name(f"{info.path.stem}_gmfss_{factor}x.{config.encode.container}")


def build_plan(config: config_module.Config, input_path: pathlib.Path) -> Plan:
    """Probe the source and resolve every derived setting.

    Raises:
        PipelineError: if the input is missing or would be overwritten.
    """
    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise PipelineError(f"Input file not found: {input_path}")

    tools = ffmpeg_module.resolve_tools(config.paths, config.root)
    info = probe_module.probe(tools.ffprobe, input_path)

    factor = config.resolve_factor(info.video.frame_rate)

    # Compose rates as fractions so 24000/1001 * 5 stays exact instead of
    # drifting to 119.8801... through float arithmetic.
    output_rate = info.video.frame_rate * factor
    output = _output_path(config, info, factor)
    if output == input_path:
        raise PipelineError("Output path would overwrite the input file")

    timesteps = tuple(index / factor for index in range(1, factor))
    return Plan(
        info=info,
        tools=tools,
        factor=factor,
        output_rate=output_rate,
        output_path=output,
        timesteps=timesteps,
        mux_via=_resolve_muxer(config, tools),
    )


def _resolve_muxer(config: config_module.Config, tools: ffmpeg_module.Tools) -> str:
    """Decide which tool writes the final container.

    mkvmerge is preferred whenever it is installed because it copies font
    attachments more reliably than ffmpeg. An explicit request for it fails
    fast here, before the model is loaded, instead of after the render.

    mkvmerge only writes Matroska, so "auto" falls back to ffmpeg for any other
    container; otherwise the output would be MKV data under a .mp4 name.
    """
    if config.mux.muxer == "ffmpeg":
        return "ffmpeg"
    if config.mux.muxer == "auto" and config.encode.container != "mkv":
        return "ffmpeg"
    if tools.mkvmerge is not None:
        return "mkvmerge"
    if config.mux.muxer == "mkvmerge":
        raise PipelineError(
            'mux.muxer = "mkvmerge" but mkvmerge was not found. Put mkvmerge.exe in '
            f"{config.root / 'tools' / 'mkv'}, add it to PATH, or set [paths] mkvmerge."
        )
    return "ffmpeg"


def describe_plan(plan: Plan, config: config_module.Config) -> str:
    """Render a human-readable summary of the plan."""
    video = plan.info.video
    subtitles = ", ".join(plan.info.subtitle_codecs) if plan.info.subtitle_codecs else "none"
    model = "GMFSS Fortuna"
    if config.dedup.mode == "off":
        dedup_line = "Dedup     : off"
    elif config.dedup.mode == "exact":
        dedup_line = "Dedup     : exact (byte-identical frames)"
    else:
        dedup_line = (
            f"Dedup     : diff (delta {config.dedup.luma_delta}, "
            f"ratio {config.dedup.max_changed_ratio})"
        )
    if plan.mux_via == "mkvmerge":
        mux_line = "Mux       : mkvmerge (video from the encode, all other tracks from the source)"
    else:
        mux_line = "Mux       : ffmpeg (single pass)"
    return "\n".join([
        f"Source    : {plan.info.path.name}",
        f"Video     : {video.width}x{video.height} @ {float(video.frame_rate):.3f} fps "
        f"({'VFR' if video.is_variable_rate else 'CFR'}, {video.pixel_format})",
        f"Tracks    : audio={'yes' if plan.info.has_audio else 'no'}, "
        f"subtitles={subtitles}, chapters={'yes' if plan.info.has_chapters else 'no'}, "
        f"attachments={'yes' if plan.info.has_attachments else 'no'}",
        f"Rate      : {float(plan.source_rate):.3f} fps x {plan.factor} = "
        f"{float(plan.output_rate):.3f} fps (mode={config.rate.mode})",
        f"Model     : {model}, {config.model.precision}",
        dedup_line,
        f"Scenes    : {'on' if config.scene.enabled else 'off'} "
        f"(threshold {config.scene.threshold})",
        f"Encoder   : {config.encode.encoder}, quality {config.encode.quality}, "
        f"{config.encode.pixel_format}",
        mux_line,
        f"Output    : {plan.output_path}",
    ])


class _FrameSource:
    """Reads fixed-size RGB24 frames from the decoder's stdout."""

    def __init__(self, process: subprocess.Popen, width: int, height: int) -> None:
        self._process = process
        self._width = width
        self._height = height
        self._frame_bytes = width * height * 3
        self._stream = process.stdout

    def read(self) -> np.ndarray | None:
        """Return the next frame, or None at end of stream."""
        if self._stream is None:
            return None
        buffer = self._stream.read(self._frame_bytes)
        if buffer is None or len(buffer) != self._frame_bytes:
            return None
        return np.frombuffer(buffer, dtype=np.uint8).reshape((self._height, self._width, 3))

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


class _FrameSink:
    """Writes RGB24 frames to the encoder's stdin."""

    def __init__(self, process: subprocess.Popen) -> None:
        self._process = process
        self._stream = process.stdin

    def write(self, frame: np.ndarray) -> None:
        if self._stream is None:
            raise PipelineError("Encoder input is closed")
        # Hand the buffer over directly instead of frame.tobytes(): the pipe
        # write copies into its own buffer anyway, so tobytes() only added a
        # second 6 MB copy per frame. cast("B") flattens the shape and asserts
        # C-contiguity, which every frame this sink receives satisfies.
        try:
            self._stream.write(memoryview(frame).cast("B"))
        except OSError as error:
            # BrokenPipeError (and Windows' EINVAL variant) means ffmpeg exited.
            raise EncoderClosedError(f"Encoder input closed unexpectedly: {error}") from error

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None


def _tail(path: pathlib.Path, lines: int = 25) -> str:
    """Return the last lines of a log file, or an empty string."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(content[-lines:])


def _progress_line(done: int, total: int, elapsed: float) -> str:
    """Build the periodic status line.

    The countdown is useful for judging speed, the wall-clock finish time is
    what actually decides whether a run fits in the evening, so both are shown.
    """
    rate = done / elapsed if elapsed > 0 else 0.0
    if total > 0:
        percent = 100.0 * done / total
        remaining = (total - done) / rate if rate > 0 else 0.0
        now = datetime.datetime.now()
        finish = now + datetime.timedelta(seconds=remaining)
        # A 1080p episode runs for many hours, so an end time on a later day
        # needs the date; a bare "23:40" would be read as tonight.
        stamp = "%H:%M" if finish.date() == now.date() else "%a %d %b %H:%M"
        return (f"progress: {percent:5.1f}%  pairs {done}/{total}  "
                f"{rate:5.1f} pair/s  ETA {remaining / 60:5.1f} min  "
                f"finish ~{finish.strftime(stamp)}")
    return f"progress: pairs {done}  {rate:5.1f} pair/s"


def run(
    config: config_module.Config,
    input_path: pathlib.Path,
    log: Callable[[str], None] = print,
) -> tuple[pathlib.Path, RunStats]:
    """Execute the full pipeline for one input file.

    Args:
        config: validated configuration.
        input_path: source video.
        log: sink for status messages.

    Returns:
        The finished output path and the run statistics.

    Raises:
        PipelineError: when any stage fails.
    """
    plan = build_plan(config, input_path)
    log(describe_plan(plan, config))

    cache_dir = pathlib.Path(config.paths.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    decoder_log = cache_dir / "decoder.log"
    encoder_log = cache_dir / "encoder.log"
    mux_log = cache_dir / "mux.log"

    # Both targets are temporary and only the final name survives a clean run.
    temp_output = plan.output_path.with_name(
        f"{plan.output_path.stem}.part{plan.output_path.suffix}"
    )
    # When mkvmerge muxes, ffmpeg writes the video stream alone and mkvmerge
    # joins it with the source tracks afterwards.
    video_only = plan.mux_via == "mkvmerge"
    temp_video = plan.output_path.with_name(
        f"{plan.output_path.stem}.video.part{plan.output_path.suffix}"
    )
    encode_target = temp_video if video_only else temp_output

    decode_command = ffmpeg_module.build_decode_command(
        plan.tools, plan.info.path, plan.source_rate
    )
    encode_command = ffmpeg_module.build_encode_command(
        plan.tools,
        plan.info,
        encode_target,
        plan.output_rate,
        config.encode,
        config.mux,
        video_only=video_only,
    )

    width = plan.info.video.width
    height = plan.info.video.height
    expected_pairs = max(1, int(round(plan.info.duration_seconds * float(plan.source_rate))))

    backend = backend_gmfss.GmfssBackend.load(config.model, config.root)
    log(f"Model loaded on {backend.device}")

    stats = RunStats()
    started = time.monotonic()

    with decoder_log.open("wb") as decoder_errors, encoder_log.open("wb") as encoder_errors:
        decoder = subprocess.Popen(
            decode_command,
            stdout=subprocess.PIPE,
            stderr=decoder_errors,
            bufsize=PIPE_BUFFER_BYTES,
        )
        try:
            encoder = subprocess.Popen(
                encode_command,
                stdin=subprocess.PIPE,
                stderr=encoder_errors,
                bufsize=PIPE_BUFFER_BYTES,
            )
        except OSError:
            decoder.kill()
            decoder.wait()
            raise

        source = _FrameSource(decoder, width, height)
        sink = _FrameSink(encoder)

        try:
            first = source.read()
            if first is None:
                raise PipelineError(
                    "The decoder produced no frames. Check that the file is a valid video "
                    f"and see {decoder_log}."
                )

            sink.write(first)
            stats.frames_written += 1
            previous_frame = first
            # The block map is only needed by "diff" dedup and by scene
            # detection. In the common defaults (exact dedup, scenes on) that
            # still means every frame, but a run with scenes off and exact
            # dedup now costs no analysis at all.
            needs_luma = config.dedup.mode == "diff" or config.scene.enabled
            previous_stats = dedup_module.analyse(first) if needs_luma else None
            previous_feature_state = None  # reused between overlapping pairs

            last_report = time.monotonic()
            while True:
                frame = source.read()
                if frame is None:
                    break

                stats.pairs += 1
                current_stats = dedup_module.analyse(frame) if needs_luma else None

                duplicate = dedup_module.is_duplicate(
                    previous_frame,
                    frame,
                    previous_stats,
                    current_stats,
                    config.dedup.mode,
                    config.dedup.luma_delta,
                    config.dedup.max_changed_ratio,
                )
                # A cut and a duplicate are mutually exclusive: identical frames
                # cannot also be a scene change.
                cut = (
                    not duplicate
                    and config.scene.enabled
                    and dedup_module.scene_difference(previous_stats, current_stats)
                    > config.scene.threshold
                )

                if duplicate:
                    stats.duplicate_pairs += 1
                if cut:
                    stats.scene_cuts += 1

                if duplicate or cut:
                    # Hold the outgoing frame instead of blending across the
                    # pair. Interpolating a cut would mix two scenes; for an
                    # identical pair there is simply nothing to invent.
                    for _ in range(plan.factor - 1):
                        sink.write(previous_frame)
                        stats.frames_written += 1
                else:
                    interpolate_result, feature_state = backend.interpolate(
                        previous_frame, frame, plan.timesteps, previous_feature_state
                    )
                    for intermediate in interpolate_result:
                        sink.write(intermediate)
                        stats.frames_written += 1
                    # Carry feature state forward for overlapping pair reuse.
                    previous_feature_state = feature_state

                sink.write(frame)
                stats.frames_written += 1
                previous_frame = frame
                previous_stats = current_stats

                now = time.monotonic()
                if now - last_report >= PROGRESS_INTERVAL:
                    log(_progress_line(stats.pairs, expected_pairs, now - started))
                    last_report = now

            sink.close()
            decoder_errors.flush()
            encoder_errors.flush()

            decoder_code = decoder.wait()
            encoder_code = encoder.wait()

        except BaseException as failure:
            # Any failure (including Ctrl+C) must not leave orphan processes or
            # a partial file behind.
            for process in (decoder, encoder):
                if process.poll() is None:
                    process.kill()
                process.wait()
            temp_output.unlink(missing_ok=True)
            temp_video.unlink(missing_ok=True)
            if isinstance(failure, EncoderClosedError):
                raise PipelineError(
                    f"The encoder stopped early. Last messages:\n{_tail(encoder_log)}"
                ) from failure
            raise
        finally:
            source.close()
            sink.close()

    if decoder_code != 0:
        temp_output.unlink(missing_ok=True)
        temp_video.unlink(missing_ok=True)
        raise PipelineError(
            f"The decoder exited with code {decoder_code}. Last messages:\n{_tail(decoder_log)}"
        )
    if encoder_code != 0:
        temp_output.unlink(missing_ok=True)
        temp_video.unlink(missing_ok=True)
        raise PipelineError(
            f"The encoder exited with code {encoder_code}. Last messages:\n{_tail(encoder_log)}"
        )

    if video_only:
        mkvmerge = plan.tools.mkvmerge
        if mkvmerge is None:
            # _resolve_muxer guarantees a path; guard so a future change cannot
            # quietly turn this into a video-only output.
            temp_video.unlink(missing_ok=True)
            raise PipelineError("mkvmerge muxing was selected but no mkvmerge executable resolved")
        try:
            warnings_text = mux_module.run(
                mkvmerge, temp_video, plan.info.path, temp_output, config.mux, mux_log
            )
        except mux_module.MuxError:
            temp_output.unlink(missing_ok=True)
            temp_video.unlink(missing_ok=True)
            raise
        temp_video.unlink(missing_ok=True)
        if warnings_text:
            log(f"mkvmerge finished with warnings; see {mux_log}")

    if not temp_output.is_file() or temp_output.stat().st_size == 0:
        temp_output.unlink(missing_ok=True)
        failure_log = mux_log if video_only else encoder_log
        raise PipelineError(
            f"Muxing finished but produced no output. Last messages:\n{_tail(failure_log)}"
        )

    os.replace(temp_output, plan.output_path)

    elapsed = time.monotonic() - started
    log(
        f"Frames written: {stats.frames_written} "
        f"({stats.pairs} pairs, {stats.duplicate_pairs} duplicate pairs skipped, "
        f"{stats.scene_cuts} scene cuts)"
    )
    log(f"Elapsed: {elapsed / 60:.1f} min")
    return plan.output_path, stats
