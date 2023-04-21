"""
Copyright (c) 2019 The MITRE Corporation.
"""

import random

import numpy as np
import torch

__all__ = ['seed_rng']

def seed_rng(seed: int):
    """
    Seed the random number generators (Python, Numpy)
    :param seed: The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
