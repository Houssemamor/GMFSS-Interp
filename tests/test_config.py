"""Tests for configuration defaults and validation.

The [model] section grew two performance switches; these tests pin both the
defaults (so a missing key keeps the fast path) and the rejection of unknown
values (so a typo fails before a render starts rather than silently falling back
to whichever path the code happens to default to).
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from src import config as config_module

SECTIONS = "[rate]\n[model]\n[dedup]\n[scene]\n[encode]\n[mux]\n[paths]\n"


class LoadConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def write(self, body: str) -> pathlib.Path:
        path = self.root / "config.toml"
        path.write_text(body, encoding="utf-8")
        return path

    def test_defaults_use_the_fast_paths(self) -> None:
        config = config_module.load_config(self.write(SECTIONS), root=self.root)
        self.assertEqual(config.model.attention, "sdpa")
        self.assertEqual(config.model.fusion_layout, "channels_last")
        self.assertEqual(config.model.precision, "fp32")

    def test_model_values_are_read(self) -> None:
        body = SECTIONS.replace(
            "[model]",
            '[model]\nprecision = "fp16"\nattention = "reference"\nfusion_layout = "nchw"',
        )
        config = config_module.load_config(self.write(body), root=self.root)
        self.assertEqual(config.model.precision, "fp16")
        self.assertEqual(config.model.attention, "reference")
        self.assertEqual(config.model.fusion_layout, "nchw")

    def test_unknown_attention_is_rejected(self) -> None:
        body = SECTIONS.replace("[model]", '[model]\nattention = "flash"')
        with self.assertRaises(config_module.ConfigError) as caught:
            config_module.load_config(self.write(body), root=self.root)
        self.assertIn("attention", str(caught.exception))

    def test_unknown_fusion_layout_is_rejected(self) -> None:
        body = SECTIONS.replace("[model]", '[model]\nfusion_layout = "nhwc"')
        with self.assertRaises(config_module.ConfigError) as caught:
            config_module.load_config(self.write(body), root=self.root)
        self.assertIn("fusion_layout", str(caught.exception))

    def test_missing_section_is_rejected(self) -> None:
        with self.assertRaises(config_module.ConfigError) as caught:
            config_module.load_config(self.write("[rate]\n"), root=self.root)
        self.assertIn("[model]", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
