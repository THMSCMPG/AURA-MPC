"""Reproducibility helper: the single authorized seeding entry point.

Every module that uses randomness should call :func:`seed_everything`
exactly once per process. Direct calls to ``torch.manual_seed``,
``np.random.seed``, or ``random.seed`` are forbidden elsewhere in ``src/``.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from .logging import get_logger

_LOGGER = get_logger(__name__)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) deterministically.

    Also sets ``torch.backends.cudnn.deterministic`` when requested so that
    convolution algorithms become reproducible at a modest throughput cost.

    Args:
        seed: Integer seed broadcast to all RNGs.
        deterministic: When ``True`` (default) also enables cuDNN
            deterministic mode and disables its autotuner.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    _LOGGER.info("seeded all RNGs", extra={"seed": seed, "deterministic": deterministic})
