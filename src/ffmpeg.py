"""External tool resolution and ffmpeg command construction.

Kept separate from the pipeline so command shape can be inspected and tested
without touching a GPU or a real media file.
"""

from __future__ import annotations

import dataclasses
import pathlib
import shutil
from fractions import Fraction

from . import config as config_module
from . import probe as probe_module


class ToolError(RuntimeError):
    """Raised when a required external executable cannot be located."""


@dataclasses.dataclass(frozen=True)
class Tools:
    """Resolved executable paths. mkvmerge is optional."""

    ffmpeg: pathlib.Path
    ffprobe: pathlib.Path
    mkvmerge: pathlib.Path | None


def _find_ffprobe(ffmpeg_path: pathlib.Path) -> pathlib.Path | None:
    """Look for ffprobe next to ffmpeg, then on PATH.

    Some distributions rename the pair (``ffmpeg_foo.exe`` / ``ffprobe_foo.exe``),
    so the sibling is derived by replacing the executable's own stem rather than
    assuming it is literally called ``ffmpeg.exe``.
    """
    name = ffmpeg_path.name
    lowered = name.lower()
    index = lowered.find("ffmpeg")
    candidates: list[pathlib.Path] = []
    if index >= 0:
        candidates.append(ffmpeg_path.with_name(name[:index] + "ffprobe" + name[index + len("ffmpeg"):]))
    candidates.append(ffmpeg_path.with_name("ffprobe" + ffmpeg_path.suffix))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    discovered = shutil.which("ffprobe")
    return pathlib.Path(discovered) if discovered else None


def resolve_mkvmerge(paths: config_module.PathConfig, root: pathlib.Path) -> pathlib.Path | None:
    """Locate mkvmerge, or return None when it is not installed.

    mkvmerge is optional: ffmpeg can mux on its own, but it is less reliable at
    carrying font attachments, which anime releases depend on.

    Raises:
        ToolError: when an explicit [paths] mkvmerge points at a missing file.
    """
    configured = paths.mkvmerge.strip()
    if configured.lower() != "auto":
        candidate = pathlib.Path(configured)
        if not candidate.is_file():
            raise ToolError(f"[paths] mkvmerge points at a file that does not exist: {candidate}")
        return candidate

    discovered = shutil.which("mkvmerge")
    if discovered:
        return pathlib.Path(discovered)

    # Portable MKVToolNix copies dropped into tools/, as this project expects.
    candidates = [
        root / "tools" / "mkv" / "mkvmerge.exe",
        root / "tools" / "mkvmerge.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tools_dir = root / "tools"
    if tools_dir.is_dir():
        for candidate in sorted(tools_dir.glob("*/mkvmerge.exe")):
            return candidate

    return None


def resolve_tools(paths: config_module.PathConfig, root: pathlib.Path) -> Tools:
    """Resolve ffmpeg and ffprobe from config, PATH, or the bundled tools folder.

    Args:
        paths: validated [paths] section.
        root: project root, where ``tools/ffmpeg.exe`` may live.

    Raises:
        ToolError: if either executable is missing, with an actionable message.
    """
    if paths.ffmpeg.strip().lower() == "auto":
        discovered = shutil.which("ffmpeg")
        if discovered:
            ffmpeg_path = pathlib.Path(discovered)
        else:
            # setup.bat drops a build here when nothing is on PATH.
            bundled = root / "tools" / "ffmpeg.exe"
            if not bundled.is_file():
                raise ToolError(
                    "ffmpeg was not found on PATH or in tools/. Either run setup.bat, "
                    "add ffmpeg to PATH, or set [paths] ffmpeg in config.toml to the "
                    "full path of ffmpeg.exe."
                )
            ffmpeg_path = bundled
    else:
        ffmpeg_path = pathlib.Path(paths.ffmpeg)
        if not ffmpeg_path.is_file():
            raise ToolError(f"[paths] ffmpeg points at a file that does not exist: {ffmpeg_path}")

    ffprobe_path = _find_ffprobe(ffmpeg_path)
    if ffprobe_path is None:
        raise ToolError(
            f"ffprobe was not found next to {ffmpeg_path.name} or on PATH. "
            "Install a full ffmpeg build (ffmpeg + ffprobe) or set [paths] ffmpeg accordingly."
        )
    return Tools(
        ffmpeg=ffmpeg_path,
        ffprobe=ffprobe_path,
        mkvmerge=resolve_mkvmerge(paths, root),
    )


# ffprobe colour_space name -> swscale out_color_matrix name. Sources with any
# other value (including untagged) get no explicit matrix: decode and encode
# then both use swscale's default, so the round trip stays self-consistent and
# the output stays untagged, exactly like the source.
_SCALE_MATRIX = {
    "bt709": "bt709",
    "bt470bg": "bt470",
    "smpte170m": "smpte170m",
    "smpte240m": "smpte240m",
    "fcc": "fcc",
    "bt2020nc": "bt2020",
}


def rate_string(rate: Fraction) -> str:
    """Format a Fraction the way ffmpeg expects (e.g. 120000/1001)."""
    return f"{rate.numerator}/{rate.denominator}"


def build_decode_command(
    tools: Tools,
    source: pathlib.Path,
    source_rate: Fraction,
) -> list[str]:
    """Build the ffmpeg command that decodes the source to raw RGB24 on stdout.

    The source is forced to constant frame rate first: pair-based interpolation
    assumes evenly spaced frames, and many anime encodes are variable rate.
    """
    return [
        str(tools.ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-v", "error",
        "-i", str(source),
        "-map", "0:v:0",
        "-an", "-sn", "-dn",
        # -fps_mode replaces the deprecated -vsync (ffmpeg 5.1+).
        "-fps_mode", "cfr",
        "-r", rate_string(source_rate),
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]


def _video_encoder_args(encode: config_module.EncodeConfig) -> list[str]:
    """Return encoder-specific quality arguments."""
    name = encode.encoder.lower()
    if name.endswith("_nvenc"):
        # NVENC constant-quality mode: -b:v 0 is required for -cq to be honoured.
        return ["-c:v", encode.encoder, "-preset", "p5", "-rc", "vbr",
                "-cq", str(encode.quality), "-b:v", "0", "-pix_fmt", encode.pixel_format]
    if name in ("libx264", "libx265"):
        return ["-c:v", encode.encoder, "-preset", "slow",
                "-crf", str(encode.quality), "-pix_fmt", encode.pixel_format]
    if name in ("libsvtav1", "libaom-av1"):
        return ["-c:v", encode.encoder, "-crf", str(encode.quality), "-pix_fmt", encode.pixel_format]
    # Unknown encoder: pass quality through and let ffmpeg complain if invalid.
    return ["-c:v", encode.encoder, "-crf", str(encode.quality), "-pix_fmt", encode.pixel_format]


def build_encode_command(
    tools: Tools,
    info: probe_module.MediaInfo,
    output: pathlib.Path,
    output_rate: Fraction,
    encode: config_module.EncodeConfig,
    mux: config_module.MuxConfig,
    video_only: bool = False,
) -> list[str]:
    """Build the ffmpeg command that encodes frames from stdin.

    Args:
        tools: resolved executables.
        info: probed source.
        output: encode target.
        output_rate: exact output frame rate.
        encode: [encode] section.
        mux: [mux] section; ignored when ``video_only`` is set.
        video_only: when True, only the video stream is written and mkvmerge
            does the final mux. Audio, subtitles, fonts and chapters are then
            copied by mkvmerge instead of ffmpeg.

    Video comes from the pipe. Interpolation leaves duration unchanged, so any
    tracks copied from the source keep valid timestamps without retiming.
    """
    command = [
        str(tools.ffmpeg),
        "-hide_banner",
        "-nostdin",
        "-v", "error",
        "-y",
        # Input 0: raw frames produced by the interpolation loop.
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{info.video.width}x{info.video.height}",
        "-framerate", rate_string(output_rate),
        "-i", "pipe:0",
        # Input 1: the original file, used only as a track source.
        "-i", str(info.path),
        "-map", "0:v:0",
    ]

    # The pipe carries RGB, so the encoder's RGB->YUV conversion must use the
    # same matrix the decoder used (taken from the source's tags) and match the
    # colourspace tag written below. Without this swscale falls back to BT.601
    # while the output is tagged BT.709, shifting saturated colours.
    matrix = _SCALE_MATRIX.get((info.video.color_space or "").lower())
    if matrix is not None:
        command += ["-vf", f"scale=out_color_matrix={matrix}"]

    if not video_only:
        if mux.preserve_audio and info.has_audio:
            command += ["-map", "1:a?"]
        if mux.preserve_subtitles and info.has_subtitles:
            command += ["-map", "1:s?"]
        if mux.preserve_attachments and info.has_attachments:
            command += ["-map", "1:t?"]
        if mux.preserve_chapters and info.has_chapters:
            command += ["-map_chapters", "1"]

    command += _video_encoder_args(encode)

    # Set the output rate explicitly. Without this, ffmpeg derives r_frame_rate
    # from millisecond container timestamps and reports 120/1 instead of
    # 120000/1001, which makes some tools think the video plays fast.
    command += ["-r", rate_string(output_rate)]

    # Copy everything that is not the re-encoded video stream. When mkvmerge
    # performs the final mux these switches are pointless: only video reaches
    # this file, and mkvmerge supplies the rest from the source.
    if not video_only:
        command += ["-c:a", "copy", "-c:s", "copy"]
        if mux.preserve_attachments and info.has_attachments:
            command += ["-c:t", "copy"]

    # Carry over colour metadata so players do not guess wrong after encode.
    if info.video.color_primaries and info.video.color_primaries != "unknown":
        command += ["-color_primaries", info.video.color_primaries]
    if info.video.color_transfer and info.video.color_transfer != "unknown":
        command += ["-color_trc", info.video.color_transfer]
    if info.video.color_space and info.video.color_space != "unknown":
        command += ["-colorspace", info.video.color_space]

    # MP4 cannot carry ASS styling or attachments; text subs are converted.
    if not video_only and encode.container == "mp4" and info.has_text_subtitles:
        command += ["-c:s", "mov_text"]

    if not video_only and mux.preserve_chapters and info.has_chapters:
        command += ["-map_metadata", "1"]

    command += ["-max_muxing_queue_size", "1024", str(output)]
    return command
