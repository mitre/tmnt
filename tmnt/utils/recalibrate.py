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

def recalibrate(x, target_entropy_ratio=0.5):
    target_e = np.log(x.shape[0]) * target_entropy_ratio
    ## line search
    def obj_fn(t):
        return abs( entropy(rescale(x, t)) - target_e )
    res = minimize_scalar(obj_fn, method='bounded', bounds=(0.01, 20.0), tol=0.1)
    return rescale(x, res.x)

