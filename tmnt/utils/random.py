"""
Copyright (c) 2019 The MITRE Corporation.
"""

import random

import mxnet as mx
import numpy as np

__all__ = ['seed_rng']

def seed_rng(seed: int):
    """
    Seed the random number generators (Python, Numpy and MXNet)
    :param seed: The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)
