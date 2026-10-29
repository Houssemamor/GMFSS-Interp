"""Entry point for the GMFSS interpolation pipeline.

Normal use is drag and drop: Interpolate.bat forwards the dropped file here.
The command line exists mainly for testing and for batching a folder.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import warnings

# The vendored operators still use the torch.cuda.amp decorators, which torch
# 2.4+ deprecates. They work, so the warnings are silenced rather than patched:
# touching vendored model code is riskier than the noise it removes.
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*torch\.cuda\.amp.*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*torch\.meshgrid.*")
# cupy warns when CUDA_PATH is unset even though its bundled runtime works fine.
warnings.filterwarnings("ignore", category=UserWarning, message=r".*CUDA path could not be detected.*")

# Failures are reported as single-line messages, so the traceback machinery is
# only wanted when the user asks for it.
from . import backend_gmfss
from . import config as config_module
from . import ffmpeg as ffmpeg_module
from . import mux as mux_module
from . import pipeline as pipeline_module
from . import probe as probe_module

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130


# Errors caused by one input file. In a batch they are reported and the next
# file is still processed. Tool and backend errors are environment-wide, so
# they abort the whole run instead.
_PER_FILE_ERRORS: tuple[tuple[type[Exception], str], ...] = (
    (pipeline_module.PipelineError, "ERROR"),
    (probe_module.ProbeError, "PROBE ERROR"),
    (mux_module.MuxError, "MUX ERROR"),
    (config_module.ConfigError, "CONFIG ERROR"),
)


def _project_root() -> pathlib.Path:
    """Return the directory holding config.toml (the src/ parent)."""
    return pathlib.Path(__file__).resolve().parent.parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Interpolate",
        description="Frame-interpolate video with GMFSS Fortuna and keep subs/chapters.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="one or more video files to process",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to config.toml (default: the one next to Interpolate.bat)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved plan and exit without rendering",
    )
    return parser


def _run_one(config: config_module.Config, source: pathlib.Path, dry_run: bool) -> None:
    """Process a single input, printing the plan and result."""
    plan = pipeline_module.build_plan(config, source)
    print(pipeline_module.describe_plan(plan, config))
    if dry_run:
        print("dry run: no frames rendered")
        return

    print("-" * 78)
    output, _stats = pipeline_module.run(config, source, log=print)
    print(f"Done: {output}")


def main(argv: list[str] | None = None) -> int:
    """Program entry point."""
    args = _build_parser().parse_args(argv)

    config_path = pathlib.Path(args.config) if args.config else _project_root() / "config.toml"

    try:
        config = config_module.load_config(config_path.resolve(), root=_project_root())

        # torch reads this at import time, so it must be set before the backend
        # module imports torch inside GmfssBackend.load.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.model.gpu_index)

        failed: list[str] = []
        for source in args.inputs:
            if len(args.inputs) > 1:
                print("=" * 78)
            try:
                _run_one(config, pathlib.Path(source), args.dry_run)
            except tuple(kind for kind, _label in _PER_FILE_ERRORS) as error:
                label = next(text for kind, text in _PER_FILE_ERRORS if isinstance(error, kind))
                print(f"{label}: {error}", file=sys.stderr)
                failed.append(source)

        if failed:
            print(f"\n{len(failed)} of {len(args.inputs)} file(s) failed:", file=sys.stderr)
            for source in failed:
                print(f"  {source}", file=sys.stderr)
            return EXIT_ERROR
        return EXIT_OK

    except KeyboardInterrupt:
        print("\nInterrupted. Partial output removed.")
        return EXIT_INTERRUPTED
    except config_module.ConfigError as error:
        print(f"CONFIG ERROR: {error}", file=sys.stderr)
        return EXIT_ERROR
    except ffmpeg_module.ToolError as error:
        print(f"TOOL ERROR: {error}", file=sys.stderr)
        return EXIT_ERROR
    except backend_gmfss.BackendError as error:
        print(f"BACKEND ERROR: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
