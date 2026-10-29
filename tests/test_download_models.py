"""Tests for checkpoint download verification (no network access)."""

from __future__ import annotations

import contextlib
import hashlib
import io
import pathlib
import tempfile
import unittest
from unittest import mock

from src import download_models


class _FakeResponse(io.BytesIO):
    """Minimal stand-in for the object returned by urlopen."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.destination = pathlib.Path(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def run_main(self, payloads: dict[str, bytes], manifest: dict) -> int:
        def fake_urlopen(url: str, timeout: int = 0) -> _FakeResponse:
            return _FakeResponse(payloads[url.rsplit("/", 1)[1]])

        with mock.patch.object(download_models, "MANIFEST", manifest), \
                mock.patch.object(download_models.urllib.request, "urlopen", fake_urlopen), \
                contextlib.redirect_stdout(io.StringIO()):
            return download_models.main(["--dest", str(self.destination)])

    @staticmethod
    def manifest_for(name: str, payload: bytes) -> dict:
        return {name: download_models.Checkpoint("test", hashlib.sha256(payload).hexdigest())}

    def test_matching_download_is_installed(self) -> None:
        payload = b"genuine weights"
        code = self.run_main({"a.pkl": payload}, self.manifest_for("a.pkl", payload))
        self.assertEqual(code, 0)
        self.assertEqual((self.destination / "a.pkl").read_bytes(), payload)
        self.assertFalse((self.destination / "a.pkl.part").exists())

    def test_tampered_download_is_rejected_and_discarded(self) -> None:
        manifest = self.manifest_for("a.pkl", b"genuine weights")
        code = self.run_main({"a.pkl": b"malicious pickle"}, manifest)
        self.assertEqual(code, 1)
        self.assertFalse((self.destination / "a.pkl").exists())
        self.assertFalse((self.destination / "a.pkl.part").exists())

    def test_corrupt_existing_file_is_replaced(self) -> None:
        payload = b"genuine weights"
        (self.destination / "a.pkl").write_bytes(b"x" * 200_000)  # big enough to pass a size check
        code = self.run_main({"a.pkl": payload}, self.manifest_for("a.pkl", payload))
        self.assertEqual(code, 0)
        self.assertEqual((self.destination / "a.pkl").read_bytes(), payload)

    def test_verified_existing_file_is_not_downloaded_again(self) -> None:
        payload = b"genuine weights"
        (self.destination / "a.pkl").write_bytes(payload)
        # An empty payload map would raise KeyError if a download were attempted.
        code = self.run_main({}, self.manifest_for("a.pkl", payload))
        self.assertEqual(code, 0)

    def test_manifest_hashes_are_well_formed(self) -> None:
        for name, checkpoint in download_models.MANIFEST.items():
            with self.subTest(name=name):
                self.assertEqual(len(checkpoint.sha256), 64)
                int(checkpoint.sha256, 16)  # raises ValueError if not hex


if __name__ == "__main__":
    unittest.main()
