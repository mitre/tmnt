# coding: utf-8

import json
import mxnet as mx
from mxnet import gluon
import io

__all__ = ['get_seed_matrix_from_file']

def get_seed_matrix_from_file(f, vocab):
    with io.open(f, 'r') as fi:
        s_terms = json.loads(fi.read())
    ## s_terms will be a dictionary with topic names/indices as keys and term lists as values
    inds = []
    mx_len = 0
    for t in s_terms.values():
        t_inds = list(map(lambda word: vocab[word], t))
        mx_len = max(mx_len, len(t_inds))
        inds.append(t_inds)
    for ind_set in inds:
        d = mx_len - len(ind_set)
        ind_set = ind_set + [0] * d
    return mx.nd.array(inds, dtype='int32')

