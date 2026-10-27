"""GMFSS Fortuna inference backend.

Wraps the vendored GMFSS architecture so the pipeline can ask for "the frames
between A and B at these timesteps" without knowing anything about padding,
device placement or the ten-tensor intermediate state GMFSS needs.

Design note (Template Method): the public ``interpolate`` method fixes the
sequence of steps (tensorise, pad, feature reuse, per-timestep inference,
crop), so callers never repeat that ordering.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from collections.abc import Sequence

import numpy as np

from . import config as config_module

# GMFSS pads both spatial dimensions to a multiple of this value. Small odd
# frame sizes otherwise fail inside the pyramid down/up sampling.
SIZE_MULTIPLE = 64

# Checkpoint set for GMFSS Fortuna. The union variant was dropped from this
# project: its extra RIFE stage is no longer mirrored anywhere reliable and the
# quality gain on animation did not justify the dependency risk.
_CHECKPOINTS = {
    "flownet": "GMFSS_fortuna_flownet.pkl",
    "metricnet": "GMFSS_fortuna_metric.pkl",
    "feat_ext": "GMFSS_fortuna_feat.pkl",
    "fusionnet": "GMFSS_fortuna_fusionnet.pkl",
}


class BackendError(RuntimeError):
    """Raised when the model cannot be loaded or run."""


@dataclasses.dataclass
class GmfssBackend:
    """Loaded GMFSS model bound to one CUDA device.

    Attributes:
        precision: "fp32" or "fp16" (autocast).
        device: torch device the weights live on.
    """

    precision: str
    device: object
    model: object
    torch: object

    # ------------------------------------------------------------------ setup

    @classmethod
    def load(cls, model_config: config_module.ModelConfig, root: pathlib.Path) -> "GmfssBackend":
        """Import the vendored network, load weights and return a backend.

        Args:
            model_config: validated [model] section.
            root: project root, used to locate vendor/ and models/.

        Raises:
            BackendError: if torch is missing, CUDA is unavailable, weights are
                absent, or the softsplat operator cannot initialise.
        """
        # The ops package imports cupy's CUDA kernels, so this must run before
        # the first inference; it also produces a clear error if cupy cannot see
        # the CUDA runtime.
        vendor = root / "vendor"
        if not vendor.is_dir():
            raise BackendError(f"Vendored model code not found at {vendor}")
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))

        try:
            import torch
        except ImportError as error:
            raise BackendError(
                "torch is not installed. Run setup.bat, or install it manually with "
                "the CUDA build matching your driver."
            ) from error

        if not torch.cuda.is_available():
            raise BackendError(
                "CUDA is not available to torch. Update the NVIDIA driver, or reinstall "
                "torch with the CUDA build (see setup.bat)."
            )

        device = torch.device("cuda", 0)
        try:
            # Fail early with a clear message if the softsplat kernels misbehave.
            from vfi_models.ops import init as ops_init

            ops_init()
        except Exception as error:  # noqa: BLE001 - backend init failures vary widely
            raise BackendError(
                f"The cupy softsplat backend failed to initialise: {error}\n"
                "cupy needs the CUDA headers to compile its kernels. Reinstall with:\n"
                "  .venv\\Scripts\\python.exe -m pip install \"cupy-cuda12x[ctk]\""
            ) from error

        try:
            from GMFSS_Fortuna_arch import Model as ModelClass
        except ImportError as error:
            raise BackendError(f"Could not import the GMFSS architecture: {error}") from error

        weights = root / "models"
        paths = {}
        missing = []
        for key, filename in _CHECKPOINTS.items():
            candidate = weights / filename
            if not candidate.is_file():
                missing.append(filename)
            paths[key] = str(candidate)
        if missing:
            raise BackendError(
                "Missing model weights: " + ", ".join(missing) +
                f"\nExpected in {weights}. Run: python src\\download_models.py"
            )

        model = ModelClass()
        model.eval()
        model.device()
        try:
            model.load_model(paths)
        except Exception as error:  # noqa: BLE001 - corrupt/incompatible checkpoints
            raise BackendError(f"Failed to load GMFSS weights: {error}") from error

        torch.backends.cudnn.benchmark = True
        return cls(
            precision=model_config.precision,
            device=device,
            model=model,
            torch=torch,
        )

    # -------------------------------------------------------------- inference

    def interpolate(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        timesteps: Sequence[float],
    ) -> list[np.ndarray]:
        """Return the interpolated frames between two RGB24 frames.

        Args:
            frame_a: uint8 (height, width, 3) frame at t=0.
            frame_b: uint8 (height, width, 3) frame at t=1.
            timesteps: fractions in (0, 1), typically k / factor.

        Returns:
            One uint8 frame per timestep, in the same order.
        """
        torch = self.torch
        height, width = frame_a.shape[:2]
        pad_h = (SIZE_MULTIPLE - height % SIZE_MULTIPLE) % SIZE_MULTIPLE
        pad_w = (SIZE_MULTIPLE - width % SIZE_MULTIPLE) % SIZE_MULTIPLE

        with torch.inference_mode():
            img0 = self._to_tensor(frame_a)
            img1 = self._to_tensor(frame_b)
            if pad_h or pad_w:
                padding = (0, pad_w, 0, pad_h)
                img0 = torch.nn.functional.pad(img0, padding)
                img1 = torch.nn.functional.pad(img1, padding)

            # Feature reuse happens once per pair; each timestep then only runs
            # the fusion stage. For a 5x multiply that is one reuse + 4 cheap
            # passes instead of 4 full forward passes.
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.precision == "fp16"):
                state = self.model.reuse(img0, img1, 1.0)
                outputs = [
                    self._to_numpy(
                        self.model.inference(img0, img1, *state, timestep)[:, :, :height, :width]
                    )
                    for timestep in timesteps
                ]

        return outputs

    def _to_tensor(self, frame: np.ndarray) -> object:
        """Convert an RGB24 numpy frame to a normalised NCHW float tensor."""
        torch = self.torch
        tensor = torch.from_numpy(np.ascontiguousarray(frame))
        tensor = tensor.to(self.device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        return tensor

    def _to_numpy(self, tensor: object) -> np.ndarray:
        """Convert an NCHW float tensor back to an RGB24 numpy frame."""
        array = tensor.squeeze(0).mul(255.0).clamp_(0, 255).byte()
        array = array.permute(1, 2, 0).contiguous()
        return array.cpu().numpy()
