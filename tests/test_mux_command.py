"""Tests for the mkvmerge command shape (no mkvmerge binary required)."""

from __future__ import annotations

import pathlib
import unittest

from src import config as config_module
from src import mux as mux_module


def _mux(**overrides: bool) -> config_module.MuxConfig:
    values = dict(
        muxer="mkvmerge",
        preserve_subtitles=True,
        preserve_chapters=True,
        preserve_attachments=True,
        preserve_audio=True,
    )
    values.update(overrides)
    return config_module.MuxConfig(**values)


def _command(mux: config_module.MuxConfig) -> list[str]:
    return mux_module.build_command(
        pathlib.Path("mkvmerge.exe"),
        pathlib.Path("video.mkv"),
        pathlib.Path("source.mkv"),
        pathlib.Path("out.mkv"),
        mux,
    )


class BuildCommandTests(unittest.TestCase):
    def test_source_options_come_after_the_video_input(self) -> None:
        command = _command(_mux())
        video_index = command.index("video.mkv")
        source_index = command.index("source.mkv")
        self.assertLess(video_index, command.index("--no-video"))
        self.assertLess(command.index("--no-video"), source_index)

    def test_defaults_keep_every_source_track_type(self) -> None:
        command = _command(_mux())
        after_video = command[command.index("video.mkv") + 1:]
        for switch in ("--no-audio", "--no-subtitles", "--no-attachments", "--no-chapters"):
            self.assertNotIn(switch, after_video)

    def test_each_preserve_flag_maps_to_one_switch(self) -> None:
        cases = {
            "preserve_audio": "--no-audio",
            "preserve_subtitles": "--no-subtitles",
            "preserve_attachments": "--no-attachments",
            "preserve_chapters": "--no-chapters",
        }
        for flag, switch in cases.items():
            with self.subTest(flag=flag):
                command = _command(_mux(**{flag: False}))
                after_video = command[command.index("video.mkv") + 1:]
                self.assertIn(switch, after_video)

    def test_paths_with_spaces_stay_single_arguments(self) -> None:
        command = mux_module.build_command(
            pathlib.Path("C:/Program Files/mkv/mkvmerge.exe"),
            pathlib.Path("a b.video.part.mkv"),
            pathlib.Path("my episode.mkv"),
            pathlib.Path("my episode_gmfss_3x.part.mkv"),
            _mux(),
        )
        self.assertIn("my episode.mkv", command)
        self.assertIn("a b.video.part.mkv", command)


if __name__ == "__main__":
    unittest.main()
