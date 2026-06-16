"""Shared utilities for the Alzheimer ensemble project."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility.

    Also sets ``PYTHONHASHSEED`` and switches cuDNN to deterministic mode so
    that runs are repeatable across sessions (Phase 8 verification).

    Args:
        seed: The seed value applied to every RNG. Defaults to ``42``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(42)
    print("set_seed(42) applied")
