# coding: utf-8

import re
import io
import gluonnlp as nlp
import numpy as np
import mxnet as mx
from sklearn.datasets import load_svmlight_file

def load_tsv_to_array(fname):
    """
    Inputs: file path
    Outputs: list/array of 3-tuples, each representing a data instance
    """
    arr = []
    with io.open(fname, 'r') as fp:
        for line in fp:
            els = line.split('\t')
            arr.append((els[0], els[1], els[2]))
    return arr


def _sv_to_seq(sv):
    sv = sv.toarray().squeeze()
    nzs = np.argwhere(sv > 0).squeeze()
    li = []
    for i in range(nzs.shape[0]):
        v = sv[nzs[i]]
        li.extend([nzs[i]] * v)
    return li


def _convert_to_seqs(dataset, labels, max_length):
    seqs = []
    for i in range(dataset.shape[0]):
        seq = _sv_to_seq(dataset[i])
        seqs.append((labels[i], seq[:max_length]))
    return seqs

def load_sparse_dataset(train_vec, val_vec, test_vec, voc_size = 2000, max_length=64):
    train_sv, train_y = load_svmlight_file(train_vec, n_features=voc_size, dtype='int32')
    val_sv, val_y = load_svmlight_file(val_vec, n_features=voc_size, dtype='int32')
    test_sv, test_y = load_svmlight_file(test_vec, n_features=voc_size, dtype='int32')
    train_dataset = _convert_to_seqs(train_sv, train_y, max_length)
    val_dataset   = _convert_to_seqs(val_sv, val_y, max_length)
    test_dataset  = _convert_to_seqs(test_sv, test_y, max_length)
    transform = MaskTransform(max_len=max_length)
    return train_dataset, val_dataset, test_dataset, transform


class MaskTransform(object):
    """
    This is a callable object used by the transform method for a dataset. It will be
    called during data loading/iteration.  

    Parameters
    ----------
    labels : list string
        List of the valid strings for classification labels
    max_len : int, default 64
        Maximum sequence length - longer seqs will be truncated and shorter ones padded
    
    """
    def __init__(self, max_len=64):
        self._max_seq_length = max_len
    
    def __call__(self, label_id, data):
        padded_data = data + [0] * (self._max_seq_length - len(data))
        good = np.ones(len(data), dtype='float32')
        bad = np.zeros(self._max_seq_length - len(data), dtype='float32')
        mask = np.concatenate([good,bad])
        return np.array(padded_data, dtype='int32'), np.array([label_id], dtype='int32'), mask
        

        
        
