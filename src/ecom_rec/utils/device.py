"""设备选择工具：CUDA > MPS（Apple Silicon）> CPU"""
from __future__ import annotations

import torch


def pick_device(prefer: str = "auto") -> torch.device:
    """按优先级返回可用设备。

    prefer="auto" 时：CUDA > MPS > CPU；否则按字符串构造（如 "cpu"/"cuda"/"mps"）。
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
