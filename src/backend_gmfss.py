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
            import GMFSS_Fortuna_arch as arch_module
            from GMFSS_Fortuna_arch import Model as ModelClass
        except ImportError as error:
            raise BackendError(f"Could not import the GMFSS architecture: {error}") from error

        # Select the transformer attention kernel, the GridNet layout and the flow
        # pass before any forward pass. The vendored module reads these flags at
        # call time, and sys.modules caching means a reload would otherwise keep a
        # previous setting. fusion_layout must be set before Model.device() runs,
        # because that is where the GridNet weights are converted.
        arch_module.ATTENTION_IMPL = model_config.attention
        arch_module.CHANNELS_LAST_FUSION = model_config.fusion_layout == "channels_last"
        arch_module.BIDIR_FLOW = model_config.flow_pass == "bidir"

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
        previous_feature_state: dict | None = None,
    ) -> tuple[list[np.ndarray], dict]:
        """Return the interpolated frames and a feature state for pair reuse.

        Consecutive pairs overlap by one frame: pair N ends on frame f+1 and
        pair N+1 starts on it. `previous_feature_state` carries the second
        frame's feature tensors from the previous call, so when `frame_a` is
        that exact frame the overlapping `feat_ext(img0)` pass is skipped
        (~15 ms/pair at 1080p).

        The state is only reused on object identity (`state["frame"] is
        frame_a`), not on content: after a skipped duplicate pair or a scene
        cut the overlap is broken, and identity is what detects that without a
        6 MB comparison per pair. A mismatched call falls back to full
        extraction.

        Returns `(interpolated_frames, state)` where `state` holds `frame_b`
        and its feature tensors (`feat21`, `feat22`, `feat23`) for the next
        overlapping pair.
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
                # Overlap reuse: the previous pair's second-frame features are
                # this pair's first-frame features when the frames are the same
                # object. Any other case (first pair, duplicate skip, scene
                # cut, different caller) fails the identity check and extracts
                # from scratch.
                feat0 = None
                if (
                    previous_feature_state is not None
                    and previous_feature_state.get("frame") is frame_a
                ):
                    feat0 = (
                        previous_feature_state["feat21"],
                        previous_feature_state["feat22"],
                        previous_feature_state["feat23"],
                    )
                state = self.model.reuse(img0, img1, 1.0, feat0=feat0)
                outputs = [
                    self._to_numpy(
                        self.model.inference(img0, img1, *state, timestep)[:, :, :height, :width]
                    )
                    for timestep in timesteps
                ]

        # reuse() returns (flow01, flow10, metric0, metric1,
        # feat11, feat12, feat13, feat21, feat22, feat23). The reusable half is
        # the second frame's: pair N's img1 is pair N+1's img0.
        feature_state = {
            "frame": frame_b,
            "feat21": state[7],
            "feat22": state[8],
            "feat23": state[9],
        }
        return outputs, feature_state

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
