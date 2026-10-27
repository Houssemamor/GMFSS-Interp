"""Download the GMFSS Fortuna checkpoints into models/.

Run once during setup:

    python src\\download_models.py

Files are fetched from the public mirror used by ComfyUI-Frame-Interpolation,
written to a .part file first, and renamed only when complete, so an interrupted
download never leaves a truncated checkpoint that torch would fail to load.
Re-running skips files that are already present.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
import urllib.error
import urllib.request

BASE_URL = "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/"

MANIFEST: dict[str, str] = {
    "GMFSS_fortuna_flownet.pkl": "optical flow network",
    "GMFSS_fortuna_metric.pkl": "flow metric network",
    "GMFSS_fortuna_feat.pkl": "feature extractor",
    "GMFSS_fortuna_fusionnet.pkl": "fusion network",
}

# A checkpoint smaller than this is a truncated download or an error page. The
# smallest real file in the set (the metric network) is about 470 KB.
MIN_PLAUSIBLE_BYTES = 64 * 1024


def _human(size: int) -> str:
    """Format a byte count for progress output."""
    return f"{size / (1024 * 1024):.1f} MB"


def _download(url: str, target: pathlib.Path) -> None:
    """Download one file with a coarse progress line."""
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

    if written < MIN_PLAUSIBLE_BYTES:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is implausibly small ({written} bytes): {url}")

    partial.replace(target)


def main(argv: list[str] | None = None) -> int:
    """Download any missing checkpoints."""
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
    for filename, label in MANIFEST.items():
        target = destination / filename
        if target.is_file() and target.stat().st_size >= MIN_PLAUSIBLE_BYTES:
            print(f"  {filename}: already present ({_human(target.stat().st_size)})")
            continue
        print(f"  {filename} ({label})")
        try:
            _download(BASE_URL + filename, target)
        except (urllib.error.URLError, RuntimeError, OSError) as error:
            print(f"  FAILED: {error}")
            failures += 1

    if failures:
        print(f"\n{failures} file(s) failed. Re-run to retry; completed files are skipped.")
        return 1
    print(f"\nAll checkpoints present in {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
