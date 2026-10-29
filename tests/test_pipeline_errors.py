"""Tests for muxer selection and encoder-pipe failure handling.

Regression coverage for two defects:
  * muxer = "auto" picked mkvmerge for an MP4 container, producing Matroska
    data under a .mp4 name.
  * a dead encoder surfaced as a raw BrokenPipeError instead of a message that
    quotes the encoder log.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from src import config as config_module
from src import ffmpeg as ffmpeg_module
from src import pipeline as pipeline_module

SECTIONS = "[rate]\n[model]\n[dedup]\n[scene]\n[encode]\n{encode}\n[mux]\n{mux}\n[paths]\n"


def _tools(with_mkvmerge: bool) -> ffmpeg_module.Tools:
    return ffmpeg_module.Tools(
        ffmpeg=pathlib.Path("ffmpeg.exe"),
        ffprobe=pathlib.Path("ffprobe.exe"),
        mkvmerge=pathlib.Path("mkvmerge.exe") if with_mkvmerge else None,
    )


class ResolveMuxerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def config(self, muxer: str, container: str) -> config_module.Config:
        body = SECTIONS.format(
            encode=f'container = "{container}"',
            mux=f'muxer = "{muxer}"',
        )
        path = self.root / "config.toml"
        path.write_text(body, encoding="utf-8")
        return config_module.load_config(path, root=self.root)

    def test_auto_uses_mkvmerge_for_mkv(self) -> None:
        result = pipeline_module._resolve_muxer(self.config("auto", "mkv"), _tools(True))
        self.assertEqual(result, "mkvmerge")

    def test_auto_never_uses_mkvmerge_for_mp4(self) -> None:
        result = pipeline_module._resolve_muxer(self.config("auto", "mp4"), _tools(True))
        self.assertEqual(result, "ffmpeg")

    def test_auto_falls_back_to_ffmpeg_without_mkvmerge(self) -> None:
        result = pipeline_module._resolve_muxer(self.config("auto", "mkv"), _tools(False))
        self.assertEqual(result, "ffmpeg")

    def test_explicit_ffmpeg_is_honoured(self) -> None:
        result = pipeline_module._resolve_muxer(self.config("ffmpeg", "mkv"), _tools(True))
        self.assertEqual(result, "ffmpeg")

    def test_explicit_mkvmerge_without_tool_is_rejected(self) -> None:
        with self.assertRaises(pipeline_module.PipelineError):
            pipeline_module._resolve_muxer(self.config("mkvmerge", "mkv"), _tools(False))

    def test_mkvmerge_with_mp4_is_rejected_at_load(self) -> None:
        with self.assertRaises(config_module.ConfigError):
            self.config("mkvmerge", "mp4")


class _DeadStream:
    """Stands in for a pipe whose reader process has exited."""

    def write(self, _data: object) -> None:
        raise BrokenPipeError("pipe closed")

    def close(self) -> None:
        pass


class _FakeProcess:
    def __init__(self, stdin: object) -> None:
        self.stdin = stdin


class FrameSinkTests(unittest.TestCase):
    def test_broken_pipe_becomes_encoder_closed_error(self) -> None:
        import numpy as np

        sink = pipeline_module._FrameSink(_FakeProcess(_DeadStream()))
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        with self.assertRaises(pipeline_module.EncoderClosedError):
            sink.write(frame)

    def test_write_after_close_is_rejected(self) -> None:
        import numpy as np

        sink = pipeline_module._FrameSink(_FakeProcess(_DeadStream()))
        sink.close()
        with self.assertRaises(pipeline_module.PipelineError):
            sink.write(np.zeros((2, 2, 3), dtype=np.uint8))

    def test_encoder_closed_error_is_a_pipeline_error(self) -> None:
        # main.py handles PipelineError per file, so the subclass must qualify.
        self.assertTrue(issubclass(pipeline_module.EncoderClosedError, pipeline_module.PipelineError))


if __name__ == "__main__":
    unittest.main()
