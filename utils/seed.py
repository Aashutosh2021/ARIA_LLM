"""
AIRA-LLM
Reproducibility / Seeding
"""

import os
import random

import numpy as np
# pyrefly: ignore [missing-import]
import torch


def set_seed(seed: int = 42, deterministic: bool = False):
    """
    Seed all the usual RNGs (python, numpy, torch CPU + CUDA).

    If ``deterministic`` is True, also configure cuDNN for deterministic
    behaviour (slower, but repeatable).
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
