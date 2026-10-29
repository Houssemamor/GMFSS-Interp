"""Tests for the command-line entry point's error handling and batching."""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from src import backend_gmfss
from src import main as main_module
from src import mux as mux_module
from src import pipeline as pipeline_module

SECTIONS = "[rate]\n[model]\n[dedup]\n[scene]\n[encode]\n[mux]\n[paths]\n"


class MainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.config_path = pathlib.Path(self._directory.name) / "config.toml"
        self.config_path.write_text(SECTIONS, encoding="utf-8")
        # main() sets CUDA_VISIBLE_DEVICES; keep that out of the test process.
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def invoke(self, inputs: list[str], side_effect: object) -> tuple[int, str, mock.Mock]:
        stderr = io.StringIO()
        with mock.patch.object(main_module, "_run_one", side_effect=side_effect) as run_one, \
                contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            code = main_module.main(["--config", str(self.config_path), *inputs])
        return code, stderr.getvalue(), run_one

    def test_success_returns_zero(self) -> None:
        code, _err, run_one = self.invoke(["a.mkv", "b.mkv"], None)
        self.assertEqual(code, main_module.EXIT_OK)
        self.assertEqual(run_one.call_count, 2)

    def test_mux_error_is_reported_not_raised(self) -> None:
        code, err, _ = self.invoke(["a.mkv"], mux_module.MuxError("boom"))
        self.assertEqual(code, main_module.EXIT_ERROR)
        self.assertIn("MUX ERROR: boom", err)

    def test_one_failing_file_does_not_skip_the_rest(self) -> None:
        outcomes = [pipeline_module.PipelineError("bad first"), None]
        code, err, run_one = self.invoke(["a.mkv", "b.mkv"], outcomes)
        self.assertEqual(code, main_module.EXIT_ERROR)
        self.assertEqual(run_one.call_count, 2)
        self.assertIn("1 of 2 file(s) failed", err)
        self.assertIn("a.mkv", err)

    def test_backend_error_aborts_the_batch(self) -> None:
        code, err, run_one = self.invoke(["a.mkv", "b.mkv"], backend_gmfss.BackendError("no cuda"))
        self.assertEqual(code, main_module.EXIT_ERROR)
        self.assertEqual(run_one.call_count, 1)
        self.assertIn("BACKEND ERROR", err)

    def test_missing_config_is_a_config_error(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = main_module.main(["--config", str(self.config_path.with_name("nope.toml")), "a.mkv"])
        self.assertEqual(code, main_module.EXIT_ERROR)
        self.assertIn("CONFIG ERROR", stderr.getvalue())

    def test_keyboard_interrupt_returns_130(self) -> None:
        code, _err, _ = self.invoke(["a.mkv"], KeyboardInterrupt())
        self.assertEqual(code, main_module.EXIT_INTERRUPTED)


if __name__ == "__main__":
    unittest.main()
