"""Tests for ffprobe output parsing (ffprobe is faked)."""

from __future__ import annotations

import json
import pathlib
import subprocess
import unittest
from unittest import mock

from src import probe as probe_module


def _payload(**video_overrides: object) -> str:
    video = {
        "codec_type": "video",
        "width": 1920,
        "height": 1080,
        "r_frame_rate": "24000/1001",
        "avg_frame_rate": "24000/1001",
        "pix_fmt": "yuv420p",
        "start_time": "0.000000",
    }
    video.update(video_overrides)
    return json.dumps({"streams": [video], "format": {"duration": "10.0"}, "chapters": []})


def _probe(stdout: str, returncode: int = 0) -> probe_module.MediaInfo:
    completed = subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="boom")
    with mock.patch.object(probe_module.subprocess, "run", return_value=completed):
        return probe_module.probe(pathlib.Path("ffprobe"), pathlib.Path("clip.mkv"))


class StartTimeTests(unittest.TestCase):
    def test_start_time_is_read(self) -> None:
        self.assertAlmostEqual(_probe(_payload(start_time="0.959000")).video.start_seconds, 0.959)

    def test_missing_start_time_is_zero(self) -> None:
        payload = json.loads(_payload())
        del payload["streams"][0]["start_time"]
        self.assertEqual(_probe(json.dumps(payload)).video.start_seconds, 0.0)

    def test_unusable_start_times_are_zero(self) -> None:
        for bad in ("N/A", "", "nan", "inf", "-inf", "abc", None):
            with self.subTest(value=bad):
                self.assertEqual(_probe(_payload(start_time=bad)).video.start_seconds, 0.0)


class ProbeFailureTests(unittest.TestCase):
    def test_ffprobe_failure_is_a_probe_error(self) -> None:
        with self.assertRaises(probe_module.ProbeError):
            _probe("", returncode=1)

    def test_unreadable_json_is_a_probe_error(self) -> None:
        with self.assertRaises(probe_module.ProbeError):
            _probe("not json")

    def test_no_video_stream_is_a_probe_error(self) -> None:
        with self.assertRaises(probe_module.ProbeError):
            _probe(json.dumps({"streams": [], "format": {}}))


if __name__ == "__main__":
    unittest.main()
