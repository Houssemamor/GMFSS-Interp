"""Minimal replacement for comfy.model_management used by the vendored GMFSS code.

The upstream modules call these three helpers to pick an inference device. The
standalone pipeline has no ComfyUI, so they resolve to plain torch/CUDA logic.
"""
import torch


def get_torch_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def is_nvidia():
    if not torch.cuda.is_available():
        return False
    return "nvidia" in torch.cuda.get_device_name(0).lower()


def get_torch_device_name(device=None):
    device = device if device is not None else get_torch_device()
    if device.type == "cuda":
        return f"cuda:{torch.cuda.current_device()} {torch.cuda.get_device_name(0)}"
    return str(device)


def soft_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
