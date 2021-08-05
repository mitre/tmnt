# coding: utf-8
# Copyright (c) 2021. The MITRE Corporation.
"""
Utilities for recalibrating posterior distributions.
"""
import math
import numpy as np
from scipy.optimize import minimize_scalar

def entropy(x):
    return - ( x * np.log(x) ).sum()

def rescale(x, t):
    x0 = x ** t
    return x0 / np.sum(x0)

def recalibrate_scores(x, target_entropy=1.0):
    e_x = entropy(x)
    entropy_ratio = e_x / np.log(x.shape[0])
    ## Some heuristics to rescale and get entropies in the ball-park of 1.0
    ## This seems to be necessary as sometimes the line search is thrown off around boundaries
    if e_x < 1e-20:
        x = rescale(x, 0.1)
    elif e_x < 0.01:
        x = rescale(x, 0.5)
    elif entropy_ratio > 0.998:
        x = rescale(x, 32.0)
    elif entropy_ratio > 0.994:
        x = rescale(x, 16.0)
    elif entropy_ratio > 0.98:
        x = rescale(x, 8.0)
    elif e_x > 2.0:
        x = rescale(x, 4.0)
    e_x = entropy(x)
    if e_x < target_entropy:
        bounds = (0.05, 1.0)
    else:
        bounds = (1.0, 32.0)
    ## line search
    def obj_fn(t):
        rescaled = rescale(x, t)
        mval = np.min(rescaled)
        mxval = np.max(rescaled)
        if mval < 1e-50 or (1.0 - mxval) < 1e-50:
            return target_entropy
        else:
            return abs( entropy(rescale(x, t)) - target_entropy )
    res = minimize_scalar(obj_fn, method='bounded', bounds=bounds)
    return rescale(x, res.x)

