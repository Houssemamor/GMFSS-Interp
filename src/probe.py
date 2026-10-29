"""Media probing via ffprobe.

Everything the pipeline needs to know about the source is gathered once here:
rate, geometry, which tracks exist, and whether the source is variable frame
rate (which must be normalised before interpolation).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
from fractions import Fraction


class ProbeError(RuntimeError):
    """Raised when ffprobe cannot describe the input."""


@dataclasses.dataclass(frozen=True)
class VideoTrack:
    """The single video stream the pipeline will read."""

    width: int
    height: int
    frame_rate: Fraction
    average_frame_rate: Fraction
    pixel_format: str
    color_primaries: str | None
    color_transfer: str | None
    color_space: str | None
    # Container start time of the video stream in seconds. Zero when unknown.
    start_seconds: float = 0.0

    @property
    def is_variable_rate(self) -> bool:
        """True when the nominal and average rates disagree.

        A variable-rate source makes pair-based interpolation meaningless: the
        time between frames is not constant, so a fixed timestep is wrong.
        """
        if self.frame_rate <= 0 or self.average_frame_rate <= 0:
            return False
        return abs(float(self.frame_rate) - float(self.average_frame_rate)) > 0.01


@dataclasses.dataclass(frozen=True)
class MediaInfo:
    """Everything the pipeline needs from a source file."""

    path: pathlib.Path
    video: VideoTrack
    has_audio: bool
    subtitle_codecs: tuple[str, ...]
    has_chapters: bool
    has_attachments: bool
    duration_seconds: float

    @property
    def has_subtitles(self) -> bool:
        return bool(self.subtitle_codecs)

    @property
    def has_text_subtitles(self) -> bool:
        """True when subtitles are text (ASS/SRT), not bitmap (PGS/VobSub)."""
        text_codecs = {"ass", "ssa", "subrip", "srt", "webvtt", "mov_text", "text"}
        return any(codec.lower() in text_codecs for codec in self.subtitle_codecs)


def _fraction(value: str | None) -> Fraction:
    """Parse ffprobe's "num/den" rate strings into a Fraction."""
    if not value or value in ("0/0", "N/A"):
        return Fraction(0)
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


def _start_seconds(value: object) -> float:
    """Parse ffprobe's start_time (a decimal string or N/A) into seconds."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    # float() accepts "nan" and "inf"; neither is a usable offset.
    return number if number == number and abs(number) != float("inf") else 0.0


def probe(ffprobe_path: pathlib.Path, media_path: pathlib.Path) -> MediaInfo:
    """Run ffprobe and build a MediaInfo.

    Args:
        ffprobe_path: resolved ffprobe executable.
        media_path: source video.

    Raises:
        ProbeError: if ffprobe fails or the file has no usable video stream.
    """
    command = [
        str(ffprobe_path),
        "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        "-show_chapters",
        str(media_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ProbeError(f"ffprobe failed for {media_path.name}: {result.stderr.strip()}")

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ProbeError(f"ffprobe returned unreadable JSON: {error}") from error

    streams = payload.get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    if not video_streams:
        raise ProbeError(f"No video stream found in {media_path.name}")

    # Cover art in audio files also reports as video; pick the first stream that
    # carries real dimensions and a plausible frame rate.
    video = None
    for stream in video_streams:
        if int(stream.get("width") or 0) > 0 and int(stream.get("height") or 0) > 0:
            video = stream
            break
    if video is None:
        raise ProbeError(f"Video stream in {media_path.name} has no usable dimensions")

    frame_rate = _fraction(video.get("r_frame_rate"))
    average_rate = _fraction(video.get("avg_frame_rate"))
    if frame_rate <= 0:
        frame_rate = average_rate
    if frame_rate <= 0:
        raise ProbeError(f"Could not determine a frame rate for {media_path.name}")

    track = VideoTrack(
        width=int(video["width"]),
        height=int(video["height"]),
        frame_rate=frame_rate,
        average_frame_rate=average_rate,
        pixel_format=str(video.get("pix_fmt") or "unknown"),
        color_primaries=video.get("color_primaries"),
        color_transfer=video.get("color_transfer"),
        color_space=video.get("color_space"),
        start_seconds=_start_seconds(video.get("start_time")),
    )

    subtitle_codecs = tuple(
        str(stream.get("codec_name") or "unknown")
        for stream in streams
        if stream.get("codec_type") == "subtitle"
    )

    format_info = payload.get("format", {})
    try:
        duration = float(format_info.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0

    return MediaInfo(
        path=media_path,
        video=track,
        has_audio=any(stream.get("codec_type") == "audio" for stream in streams),
        subtitle_codecs=subtitle_codecs,
        has_chapters=bool(payload.get("chapters")),
        has_attachments=any(stream.get("codec_type") == "attachment" for stream in streams),
        duration_seconds=duration,
    )
