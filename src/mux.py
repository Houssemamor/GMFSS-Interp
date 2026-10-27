"""Final muxing with mkvmerge.

ffmpeg can copy audio and subtitle tracks, but attachments are where anime
releases break: when a font is missing, ASS signs silently fall back to a
default face and the typesetting stops matching the release. mkvmerge copies
attachments, chapters and tags verbatim, so it produces the final file whenever
it is available and ffmpeg covers the case where it is not.

Only the container plumbing lives here; the video encode stays in ffmpeg.py.
"""

from __future__ import annotations

import pathlib
import subprocess

from . import config as config_module

# mkvmerge exit codes: 0 = finished cleanly, 1 = finished with warnings,
# 2 = error, 3 = fatal error. Anything at or above 2 means no usable output.
MKVMERGE_WARNINGS = 1
MKVMERGE_ERROR = 2

# How many trailing log lines to quote back at the user on failure.
ERROR_TAIL_LINES = 25


class MuxError(RuntimeError):
    """Raised when mkvmerge fails or cannot be started."""


def build_command(
    mkvmerge: pathlib.Path,
    video: pathlib.Path,
    source: pathlib.Path,
    output: pathlib.Path,
    mux: config_module.MuxConfig,
) -> list[str]:
    """Build the mkvmerge command that grafts source tracks onto new video.

    mkvmerge applies per-file options to the input file that follows them, so
    the switches placed before ``video`` reduce that file to its video track
    and the switch before ``source`` only drops the outdated video stream.

    Chapters, attachments and global tags are container-level in Matroska:
    mkvmerge takes them from the first input that supplies any. Stripping them
    from the encoder output is what makes the source's versions win.

    Args:
        mkvmerge: resolved mkvmerge executable.
        video: encoded video-only file.
        source: original file, used for every non-video track.
        output: destination file.
        mux: [mux] switches; each preserve_* flag maps to one --no-* option on
            the source, so disabling a track type behaves as it does with
            ffmpeg muxing.
    """
    command = [
        str(mkvmerge),
        "--output", str(output),
        # Keep only the re-encoded video from the encoder's output file.
        "--no-audio",
        "--no-subtitles",
        "--no-buttons",
        "--no-chapters",
        "--no-attachments",
        "--no-global-tags",
        "--no-track-tags",
        str(video),
        # Everything except the old video stream from the original file.
        "--no-video",
    ]
    if not mux.preserve_audio:
        command.append("--no-audio")
    if not mux.preserve_subtitles:
        command.append("--no-subtitles")
    if not mux.preserve_attachments:
        command.append("--no-attachments")
    if not mux.preserve_chapters:
        command.append("--no-chapters")
    command.append(str(source))
    return command


def run(
    mkvmerge: pathlib.Path,
    video: pathlib.Path,
    source: pathlib.Path,
    output: pathlib.Path,
    mux: config_module.MuxConfig,
    log_path: pathlib.Path,
) -> str:
    """Mux the encoded video with the selected tracks of the source.

    Args:
        mkvmerge: resolved mkvmerge executable.
        video: encoded video-only file.
        source: original file, used for audio, subtitles, fonts and chapters.
        output: destination file, written directly by mkvmerge.
        mux: [mux] switches controlling which track types are copied.
        log_path: file that receives mkvmerge's combined stdout and stderr.

    Returns:
        mkvmerge's warning text, empty when the run was clean.

    Raises:
        MuxError: when mkvmerge could not run or reported an error.
    """
    command = build_command(mkvmerge, video, source, output, mux)
    try:
        # mkvmerge writes UTF-8 regardless of the console code page, so decode
        # explicitly instead of relying on the system default.
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except OSError as error:
        raise MuxError(f"Could not start mkvmerge at {mkvmerge}: {error}") from error

    combined = result.stdout + result.stderr
    log_path.write_text(combined, encoding="utf-8", errors="replace")

    if result.returncode >= MKVMERGE_ERROR:
        tail = "\n".join(combined.splitlines()[-ERROR_TAIL_LINES:])
        raise MuxError(
            f"mkvmerge exited with code {result.returncode}. Last messages:\n{tail}"
        )
    if result.returncode == MKVMERGE_WARNINGS:
        return combined.strip()
    return ""
