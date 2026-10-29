"""Tests for ffmpeg command construction (no ffmpeg binary required)."""

from __future__ import annotations

import pathlib
import unittest
from fractions import Fraction

from src import config as config_module
from src import ffmpeg as ffmpeg_module
from src import probe as probe_module

TOOLS = ffmpeg_module.Tools(
    ffmpeg=pathlib.Path("ffmpeg.exe"),
    ffprobe=pathlib.Path("ffprobe.exe"),
    mkvmerge=None,
)
ENCODE = config_module.EncodeConfig(
    encoder="hevc_nvenc", quality=18, pixel_format="yuv420p10le", container="mkv",
)
MUX = config_module.MuxConfig(
    muxer="ffmpeg",
    preserve_subtitles=True,
    preserve_chapters=True,
    preserve_attachments=True,
    preserve_audio=True,
)


def _info(color_space: str | None) -> probe_module.MediaInfo:
    video = probe_module.VideoTrack(
        width=1920,
        height=1080,
        frame_rate=Fraction(24000, 1001),
        average_frame_rate=Fraction(24000, 1001),
        pixel_format="yuv420p",
        color_primaries=None,
        color_transfer=None,
        color_space=color_space,
    )
    return probe_module.MediaInfo(
        path=pathlib.Path("source.mkv"),
        video=video,
        has_audio=True,
        subtitle_codecs=(),
        has_chapters=False,
        has_attachments=False,
        duration_seconds=10.0,
    )


def _encode(color_space: str | None) -> list[str]:
    return ffmpeg_module.build_encode_command(
        TOOLS,
        _info(color_space),
        pathlib.Path("out.mkv"),
        Fraction(72000, 1001),
        ENCODE,
        MUX,
    )


class DecodeCommandTests(unittest.TestCase):
    def test_uses_fps_mode_not_the_deprecated_vsync(self) -> None:
        command = ffmpeg_module.build_decode_command(
            TOOLS, pathlib.Path("source.mkv"), Fraction(24000, 1001)
        )
        self.assertIn("-fps_mode", command)
        self.assertNotIn("-vsync", command)
        self.assertEqual(command[command.index("-fps_mode") + 1], "cfr")

    def test_source_path_with_spaces_is_one_argument(self) -> None:
        command = ffmpeg_module.build_decode_command(
            TOOLS, pathlib.Path("my episode 01.mkv"), Fraction(24)
        )
        self.assertIn("my episode 01.mkv", command)


class DecodeTimestampTests(unittest.TestCase):
    def decode(self, start_seconds: float) -> list[str]:
        return ffmpeg_module.build_decode_command(
            TOOLS, pathlib.Path("source.mkv"), Fraction(24000, 1001), start_seconds
        )

    def test_zero_start_keeps_source_timestamps(self) -> None:
        command = self.decode(0.0)
        self.assertIn("-copyts", command)
        # -copyts is an input option, so it must precede the input it applies to.
        self.assertLess(command.index("-copyts"), command.index("-i"))

    def test_small_lead_in_keeps_source_timestamps(self) -> None:
        self.assertIn("-copyts", self.decode(0.959))

    def test_limit_is_inclusive(self) -> None:
        limit = ffmpeg_module.MAX_PRESERVED_START_SECONDS
        self.assertIn("-copyts", self.decode(limit))
        self.assertNotIn("-copyts", self.decode(limit + 0.001))

    def test_large_start_offset_falls_back(self) -> None:
        # A capture starting at hour 1 would otherwise be padded with an hour of
        # duplicate frames.
        self.assertNotIn("-copyts", self.decode(3600.0))

    def test_start_defaults_to_zero_for_old_callers(self) -> None:
        command = ffmpeg_module.build_decode_command(
            TOOLS, pathlib.Path("source.mkv"), Fraction(24)
        )
        self.assertIn("-copyts", command)


class EncodeColourTests(unittest.TestCase):
    def test_bt709_source_converts_rgb_with_the_bt709_matrix(self) -> None:
        command = _encode("bt709")
        self.assertEqual(command[command.index("-vf") + 1], "scale=out_color_matrix=bt709")
        self.assertEqual(command[command.index("-colorspace") + 1], "bt709")

    def test_matrix_filter_comes_after_both_inputs(self) -> None:
        command = _encode("bt709")
        self.assertGreater(command.index("-vf"), max(i for i, a in enumerate(command) if a == "-i"))

    def test_untagged_source_adds_no_matrix_filter(self) -> None:
        self.assertNotIn("-vf", _encode(None))
        self.assertNotIn("-vf", _encode("unknown"))

    def test_unmapped_colourspace_adds_no_matrix_filter(self) -> None:
        # A value we have no swscale name for must not produce an invalid filter.
        self.assertNotIn("-vf", _encode("ictcp"))

    def test_every_mapped_name_is_accepted_case_insensitively(self) -> None:
        for name, matrix in ffmpeg_module._SCALE_MATRIX.items():
            with self.subTest(name=name):
                command = _encode(name.upper())
                self.assertEqual(command[command.index("-vf") + 1], f"scale=out_color_matrix={matrix}")


if __name__ == "__main__":
    unittest.main()
