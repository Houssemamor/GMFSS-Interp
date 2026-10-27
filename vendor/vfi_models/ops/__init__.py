"""Softsplat operator used by the GMFSS architecture.

Standalone replacement for the ComfyUI-Frame-Interpolation version, which reads
config.yaml from the ComfyUI install root and re-exports a large surface used by
other VFI models. GMFSS needs only ``softsplat`` and ``init``, and this project
ships only the cupy implementation because the taichi variant runs every call
through a worker process and deadlocks when driven from a plain script.
"""

from .cupy_ops import init, softsplat

__all__ = ["init", "softsplat"]
