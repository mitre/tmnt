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
    ## line search
    def obj_fn(t):
        rescaled = rescale(x, t)
        mval = np.min(rescaled)
        mxval = np.max(rescaled)
        if mval < 1e-50 or (1.0 - mxval) < 1e-50:
            return target_entropy
        else:
            return abs( entropy(rescale(x, t)) - target_entropy )
    if entropy(x) < target_entropy:
        bounds = (0.001, 1.0)
    else:
        bounds = (1.0, 100.0)
    res = minimize_scalar(obj_fn, method='bounded', bounds=bounds)
    return rescale(x, res.x)

