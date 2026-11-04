"""Small shared helpers."""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    box: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield box
    finally:
        box["seconds"] = time.perf_counter() - start
