"""Download the GMFSS Fortuna checkpoints into models/.

Run once during setup:

    python src\\download_models.py

Files are fetched from the public mirror used by ComfyUI-Frame-Interpolation,
written to a .part file first, and renamed only when complete AND when their
SHA-256 matches the pinned value below. The checkpoints are pickle-based, so an
unverified file is a code-execution risk; a mismatch is treated as a failure
and the file is discarded. Re-running skips files that are already present and
valid, and replaces present files that fail the check.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
import time
import urllib.error
import urllib.request
from typing import NamedTuple

BASE_URL = "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/"


class Checkpoint(NamedTuple):
    """One required weight file."""

    label: str
    sha256: str


# Hashes were taken from the files this project was developed against and
# cross-checked against repeated fresh downloads of BASE_URL. To update a
# checkpoint deliberately, replace its hash here in the same commit.
MANIFEST: dict[str, Checkpoint] = {
    "GMFSS_fortuna_flownet.pkl": Checkpoint(
        "optical flow network",
        "31fa34960d6326f231c35db9070728419dbad78b463567601dfb6b34049789ef",
    ),
    "GMFSS_fortuna_metric.pkl": Checkpoint(
        "flow metric network",
        "488d3ec1d7cfb04e84ee5a733b03086beb995cd3a51099d65e7a6e01e662fa16",
    ),
    "GMFSS_fortuna_feat.pkl": Checkpoint(
        "feature extractor",
        "7442a19d7e2eda6c8f70ddb5b477cb00ae311626355a84393189d8f4d4b1f03c",
    ),
    "GMFSS_fortuna_fusionnet.pkl": Checkpoint(
        "fusion network",
        "58809ca28e2594bfbef49d68d8e675a33c477cee05d18d4c75035010019c68bf",
    ),
}

# Read size when hashing; the largest checkpoint is about 30 MB.
HASH_BLOCK_BYTES = 1024 * 1024


def _human(size: int) -> str:
    """Format a byte count for progress output."""
    return f"{size / (1024 * 1024):.1f} MB"


def sha256_of(path: pathlib.Path) -> str:
    """Return the hex SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(HASH_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, target: pathlib.Path, expected_sha256: str) -> None:
    """Download one file, verify it, and only then move it into place."""
    partial = target.with_name(target.name + ".part")
    started = time.monotonic()
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        written = 0
        while True:
            block = response.read(1024 * 128)
            if not block:
                break
            handle.write(block)
            written += len(block)
            if total:
                percent = 100.0 * written / total
                sys.stdout.write(f"\r    {percent:5.1f}%  {_human(written)} / {_human(total)}")
            else:
                sys.stdout.write(f"\r    {_human(written)}")
            sys.stdout.flush()
    sys.stdout.write(f"\r    done in {time.monotonic() - started:.1f}s ({_human(written)})\n")

    actual = sha256_of(partial)
    if actual != expected_sha256:
        # Never keep an unverified pickle: it could be truncated or malicious.
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {target.name}: expected {expected_sha256}, got {actual}"
        )

    partial.replace(target)


def _is_valid(target: pathlib.Path, expected_sha256: str) -> bool:
    """True when the file exists and matches its pinned hash."""
    return target.is_file() and sha256_of(target) == expected_sha256


def main(argv: list[str] | None = None) -> int:
    """Download any missing or corrupt checkpoints."""
    parser = argparse.ArgumentParser(description="Download GMFSS Fortuna checkpoints.")
    parser.add_argument(
        "--dest",
        default=None,
        help="target directory (default: models/ in the project root)",
    )
    args = parser.parse_args(argv)

    root = pathlib.Path(__file__).resolve().parent.parent
    destination = pathlib.Path(args.dest).resolve() if args.dest else root / "models"
    destination.mkdir(parents=True, exist_ok=True)

    failures = 0
    for filename, checkpoint in MANIFEST.items():
        target = destination / filename
        if _is_valid(target, checkpoint.sha256):
            print(f"  {filename}: already present and verified ({_human(target.stat().st_size)})")
            continue
        if target.is_file():
            print(f"  {filename}: present but does not match the pinned SHA-256, replacing")
        print(f"  {filename} ({checkpoint.label})")
        try:
            _download(BASE_URL + filename, target, checkpoint.sha256)
        except (urllib.error.URLError, RuntimeError, OSError) as error:
            print(f"  FAILED: {error}")
            failures += 1

    if failures:
        print(f"\n{failures} file(s) failed. Re-run to retry; verified files are skipped.")
        return 1
    print(f"\nAll checkpoints present and verified in {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
