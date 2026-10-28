"""Tests for the optional precomputed-feature argument of Model.reuse.

``reuse(img0, img1, feat0=...)`` must behave exactly as if ``feat_ext(img0)``
had just run on identical tensors - the backend relies on that when a
consecutive video pair overlaps by one frame. The negative case matters too:
features extracted from a different frame must change the result, otherwise the
backend's object-identity guard would be defending against nothing.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

import torch

VENDOR = pathlib.Path(__file__).resolve().parent.parent / "vendor"

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    try:
        import GMFSS_Fortuna_arch as arch
    except Exception:  # pragma: no cover - cupy or the CUDA toolchain is absent
        CUDA_AVAILABLE = False


@unittest.skipUnless(CUDA_AVAILABLE, "needs the vendored architecture and cupy")
class ReuseFeatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # The vendored Model.device() moves its modules to this module global.
        arch.device = "cuda"
        torch.manual_seed(17)
        cls.model = arch.Model()
        cls.model.device()
        cls.model.eval()
        # Random-init weights are enough: nothing here loads checkpoints, and
        # the tensors stay small so the suite runs in seconds. The images are
        # multiples of 64 because GMFlow's window attention asserts that.
        cls.img0 = torch.rand(1, 3, 64, 128, device="cuda")
        cls.img1 = torch.rand(1, 3, 64, 128, device="cuda")

    @classmethod
    def tearDownClass(cls) -> None:
        del cls.model

    def test_reused_features_match_a_fresh_extraction(self) -> None:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            feat0 = self.model.feat_ext(self.img0)
            full = self.model.reuse(self.img0, self.img1, 1.0)
            reused = self.model.reuse(self.img0, self.img1, 1.0, feat0=feat0)

        self.assertEqual(len(full), len(reused))
        for index, (a, b) in enumerate(zip(full, reused)):
            drift = (a - b).abs().max().item()
            # feat_ext is a deterministic stack of convolutions on the very
            # same tensors, so nothing may drift at all; the tolerance only
            # keeps a hypothetical future kernel wobble from failing CI.
            self.assertLessEqual(drift, 1e-6, f"tensor {index}: drift {drift:.3e}")

    def test_features_from_a_different_frame_change_the_result(self) -> None:
        # Abuse case: the identity guard is meant to stop a stale or foreign
        # feature set from being trusted. Flows and metrics never see feat0 -
        # they are computed from the images - so the wrong features surface in
        # the returned feature state (index 4 = feat11). A drift of zero there
        # would also mean the feat0 branch silently ignored its argument, the
        # "accepted and ignored" bug this guards against.
        mirrored = torch.flip(self.img0, dims=(3,))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            wrong = self.model.feat_ext(mirrored)
            full = self.model.reuse(self.img0, self.img1, 1.0)
            poisoned = self.model.reuse(self.img0, self.img1, 1.0, feat0=wrong)

        for index in range(4):  # flows/metric untouched by feat0
            self.assertTrue(torch.equal(full[index], poisoned[index]), f"tensor {index} moved")
        drift = (full[4] - poisoned[4]).abs().max().item()
        self.assertGreater(drift, 1e-3, f"feat11 only moved by {drift:.3e}")


if __name__ == "__main__":
    unittest.main()
