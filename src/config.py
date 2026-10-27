"""Configuration loading and validation.

The config file lives next to the launcher so the user can tune the pipeline
once and then only ever drop video files onto Interpolate.bat. Every value is
validated up front: a bad setting produces a named error before any GPU work
starts, rather than a traceback three minutes into a render.
"""

from __future__ import annotations

import dataclasses
import pathlib
import tomllib
from fractions import Fraction
from typing import Any

# Tolerance when converting an absolute target FPS into a whole-number factor.
# 2% absorbs rounding such as 119.88 against a requested 120.
FACTOR_TOLERANCE = 0.02


class ConfigError(ValueError):
    """Raised when config.toml holds a value the pipeline cannot honour."""


@dataclasses.dataclass(frozen=True)
class RateConfig:
    """How the output frame rate is derived from the source rate."""

    mode: str
    factor: int
    target_fps: float


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Which interpolation network to load and how."""

    precision: str
    gpu_index: int


@dataclasses.dataclass(frozen=True)
class DedupConfig:
    """Duplicate-frame skipping."""

    mode: str
    luma_delta: float
    max_changed_ratio: float


@dataclasses.dataclass(frozen=True)
class SceneConfig:
    """Hard-cut detection."""

    enabled: bool
    threshold: float


@dataclasses.dataclass(frozen=True)
class EncodeConfig:
    """Output encoding."""

    encoder: str
    quality: int
    pixel_format: str
    container: str


@dataclasses.dataclass(frozen=True)
class MuxConfig:
    """How the final file is assembled and what is carried over."""

    muxer: str
    preserve_subtitles: bool
    preserve_chapters: bool
    preserve_attachments: bool
    preserve_audio: bool


@dataclasses.dataclass(frozen=True)
class PathConfig:
    """External tool locations."""

    ffmpeg: str
    mkvmerge: str
    cache_dir: str


@dataclasses.dataclass(frozen=True)
class Config:
    """Fully validated pipeline configuration."""

    root: pathlib.Path
    rate: RateConfig
    model: ModelConfig
    dedup: DedupConfig
    scene: SceneConfig
    encode: EncodeConfig
    mux: MuxConfig
    paths: PathConfig

    def resolve_factor(self, source_fps: Fraction) -> int:
        """Return the integer interpolation factor for this source rate.

        ``mode = "factor"`` uses the configured value directly. ``mode = "fps"``
        divides the target by the source rate and requires the result to be a
        whole number, because fractional multiplication would drift the output
        rate away from the requested one.
        """
        if self.rate.mode == "factor":
            return self.rate.factor

        implied = self.rate.target_fps / float(source_fps)
        rounded = round(implied)
        if rounded < 2:
            raise ConfigError(
                f"rate.target_fps = {self.rate.target_fps} against a source rate of "
                f"{float(source_fps):.3f} fps implies a factor of {implied:.3f}, "
                "which is below 2x. Lower rate.target_fps or use rate.mode = \"factor\"."
            )
        if abs(implied - rounded) > FACTOR_TOLERANCE * implied:
            raise ConfigError(
                f"rate.target_fps = {self.rate.target_fps} is not a whole multiple of the "
                f"source rate {float(source_fps):.3f} fps (implied factor {implied:.4f}). "
                f"Use rate.mode = \"factor\" with factor = {rounded}, or set target_fps to "
                f"{float(source_fps) * rounded:.3f}."
            )
        return rounded


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a required top-level TOML table."""
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"config.toml is missing the [{name}] section")
    return value


def _get(section: dict[str, Any], table: str, key: str, default: Any) -> Any:
    """Return a table value or its default."""
    return section.get(key, default)


def _require_choice(value: Any, table: str, key: str, allowed: tuple[str, ...]) -> str:
    """Validate a string against an allowed set."""
    if value not in allowed:
        joined = ", ".join(f'"{item}"' for item in allowed)
        raise ConfigError(f"[{table}] {key} = {value!r} is invalid; expected one of {joined}")
    return str(value)


def _require_int(value: Any, table: str, key: str, low: int, high: int) -> int:
    """Validate an integer inside an inclusive range."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"[{table}] {key} must be a whole number, got {value!r}")
    if not low <= value <= high:
        raise ConfigError(f"[{table}] {key} = {value} is out of range ({low}..{high})")
    return value


def _require_float(value: Any, table: str, key: str, low: float, high: float) -> float:
    """Validate a float inside an inclusive range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"[{table}] {key} must be a number, got {value!r}")
    number = float(value)
    if not low <= number <= high:
        raise ConfigError(f"[{table}] {key} = {number} is out of range ({low}..{high})")
    return number


def _require_bool(value: Any, table: str, key: str) -> bool:
    """Validate a boolean."""
    if not isinstance(value, bool):
        raise ConfigError(f"[{table}] {key} must be true or false, got {value!r}")
    return value


def _require_str(value: Any, table: str, key: str) -> str:
    """Validate a non-empty string."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"[{table}] {key} must be a non-empty string, got {value!r}")
    return value.strip()


def load_config(path: pathlib.Path, root: pathlib.Path | None = None) -> Config:
    """Read and validate config.toml.

    Args:
        path: location of the TOML file.
        root: project root used for models/, vendor/ and relative cache paths.
            Defaults to the config file's own directory.

    Raises:
        ConfigError: when the file is absent, malformed, or holds a bad value.
    """
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    with path.open("rb") as handle:
        raw = handle.read()
    try:
        # utf-8-sig tolerates the byte-order mark that Windows editors add.
        data = tomllib.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"{path.name} is not valid TOML: {error}") from error

    root = (root if root is not None else path.parent).resolve()

    rate_raw = _section(data, "rate")
    rate = RateConfig(
        mode=_require_choice(_get(rate_raw, "rate", "mode", "factor"), "rate", "mode", ("factor", "fps")),
        factor=_require_int(_get(rate_raw, "rate", "factor", 2), "rate", "factor", 2, 16),
        target_fps=_require_float(_get(rate_raw, "rate", "target_fps", 60.0), "rate", "target_fps", 2.0, 480.0),
    )

    model_raw = _section(data, "model")
    model = ModelConfig(
        precision=_require_choice(
            _get(model_raw, "model", "precision", "fp32"), "model", "precision", ("fp32", "fp16"),
        ),
        gpu_index=_require_int(_get(model_raw, "model", "gpu_index", 0), "model", "gpu_index", 0, 15),
    )

    dedup_raw = _section(data, "dedup")
    dedup = DedupConfig(
        mode=_require_choice(
            _get(dedup_raw, "dedup", "mode", "exact"), "dedup", "mode", ("off", "exact", "diff"),
        ),
        luma_delta=_require_float(
            _get(dedup_raw, "dedup", "changed_luma", 6.0), "dedup", "changed_luma", 0.0, 64.0,
        ),
        max_changed_ratio=_require_float(
            _get(dedup_raw, "dedup", "max_changed_ratio", 0.0005),
            "dedup", "max_changed_ratio", 0.0, 1.0,
        ),
    )

    scene_raw = _section(data, "scene")
    scene = SceneConfig(
        enabled=_require_bool(_get(scene_raw, "scene", "enabled", True), "scene", "enabled"),
        threshold=_require_float(_get(scene_raw, "scene", "threshold", 0.30), "scene", "threshold", 0.0, 1.0),
    )

    encode_raw = _section(data, "encode")
    encode = EncodeConfig(
        encoder=_require_str(_get(encode_raw, "encode", "encoder", "hevc_nvenc"), "encode", "encoder"),
        quality=_require_int(_get(encode_raw, "encode", "quality", 18), "encode", "quality", 0, 63),
        pixel_format=_require_str(
            _get(encode_raw, "encode", "pixel_format", "yuv420p10le"), "encode", "pixel_format",
        ),
        container=_require_choice(
            _get(encode_raw, "encode", "container", "mkv"), "encode", "container", ("mkv", "mp4"),
        ),
    )

    mux_raw = _section(data, "mux")
    mux = MuxConfig(
        muxer=_require_choice(
            _get(mux_raw, "mux", "muxer", "auto"), "mux", "muxer", ("auto", "ffmpeg", "mkvmerge"),
        ),
        preserve_subtitles=_require_bool(
            _get(mux_raw, "mux", "preserve_subtitles", True), "mux", "preserve_subtitles",
        ),
        preserve_chapters=_require_bool(
            _get(mux_raw, "mux", "preserve_chapters", True), "mux", "preserve_chapters",
        ),
        preserve_attachments=_require_bool(
            _get(mux_raw, "mux", "preserve_attachments", True), "mux", "preserve_attachments",
        ),
        preserve_audio=_require_bool(
            _get(mux_raw, "mux", "preserve_audio", True), "mux", "preserve_audio",
        ),
    )

    # mkvmerge only writes Matroska. Reject the impossible combination here so
    # the user is told before a multi-hour render starts.
    if mux.muxer == "mkvmerge" and encode.container != "mkv":
        raise ConfigError(
            '[mux] muxer = "mkvmerge" requires [encode] container = "mkv"; '
            'mkvmerge cannot write MP4. Use muxer = "auto" or "ffmpeg" for MP4 output.'
        )

    paths_raw = _section(data, "paths")
    paths = PathConfig(
        ffmpeg=_require_str(_get(paths_raw, "paths", "ffmpeg", "auto"), "paths", "ffmpeg"),
        mkvmerge=_require_str(_get(paths_raw, "paths", "mkvmerge", "auto"), "paths", "mkvmerge"),
        cache_dir=_require_str(_get(paths_raw, "paths", "cache_dir", "cache"), "paths", "cache_dir"),
    )

    # The cache path is resolved relative to the project root so the launcher
    # behaves the same regardless of the working directory it is invoked from.
    paths = dataclasses.replace(
        paths,
        cache_dir=str((root / paths.cache_dir).resolve()) if not pathlib.Path(paths.cache_dir).is_absolute()
        else paths.cache_dir,
    )

    return Config(
        root=root,
        rate=rate,
        model=model,
        dedup=dedup,
        scene=scene,
        encode=encode,
        mux=mux,
        paths=paths,
    )
