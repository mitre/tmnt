# coding: utf-8
# Copyright (c) 2019-2021. The MITRE Corporation.
"""
File/module contains routines for loading in text documents to sparse matrix representations
for efficient neural variational model training.
"""

import io
import itertools
import os
import logging
import scipy
import gluonnlp as nlp
import mxnet as mx
import numpy as np
import string
import re
import json
from mxnet import gluon
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform
from collections import OrderedDict
from mxnet.io import DataDesc, DataIter, DataBatch
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle as sk_shuffle
from tmnt.preprocess.vectorizer import TMNTVectorizer


def to_label_matrix(yvs, num_labels=0):
    """Convert [(id1, id2, ...), (id1,id2,...) ... ] to Numpy matrix with multi-labels
    """
    if num_labels == 0:
        mx_val = 0
        for yi in yvs:
            for v in yi:
                if v > mx_val:
                    mx_val = v
        num_labels = int(mx_val + 1)
    li = []
    for yi in yvs:
        a = np.zeros(num_labels)
        a[np.array(yi, dtype='int64')] = 1.0
        li.append(a)
    return np.array(li), num_labels



class SparseMatrixDataIter(DataIter):
    def __init__(self, data, label=None, batch_size=1, shuffle=False,
                 last_batch_handle='pad', data_name='data',
                 label_name='softmax_label'):
        super(SparseMatrixDataIter, self).__init__(batch_size)

        assert(isinstance(data, scipy.sparse.csr.csr_matrix))
        
        self.data = _init_data(data, allow_empty=False, default_name=data_name)
        self.label = _init_data(label, allow_empty=True, default_name=label_name)
        self.num_data = self.data[0][1].shape[0]

        # shuffle data
        if shuffle:
            sh_data = []
            d = self.data[0][1]
            if len(self.label[0][1]) > 0:
                l = self.label[0][1]
                ds, dl = sk_shuffle(d, l)
                self.data = _init_data(ds, allow_empty=False, default_name=data_name)
                self.label = _init_data(dl, allow_empty=True, default_name=label_name)
            else:
                ds = sk_shuffle(d)
                self.data = _init_data(ds, allow_empty=False, default_name=data_name)

        # batching
        if last_batch_handle == 'discard':
            new_n = self.data[0][1].shape[0] - self.data[0][1].shape[0] % batch_size
            self.num_data = new_n

        self.data_list = [x[1] for x in self.data] + [x[1] for x in self.label]
        assert self.num_data >= batch_size, "batch_size needs to be smaller than data size."
        self.cursor = -batch_size
        self.batch_size = batch_size
        self.last_batch_handle = last_batch_handle


    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator."""
        return [
            DataDesc(k, tuple([self.batch_size] + list(v.shape[1:])), v.dtype)
            for k, v in self.data
        ]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator."""
        return [
            DataDesc(k, tuple([self.batch_size] + list(v.shape[1:])), v.dtype)
            for k, v in self.label
        ]

    def hard_reset(self):
        """Ignore roll over data and set to start."""
        self.cursor = -self.batch_size


    def reset(self):
        if self.last_batch_handle == 'roll_over' and self.cursor > self.num_data:
            self.cursor = -self.batch_size + (self.cursor%self.num_data)%self.batch_size
        else:
            self.cursor = -self.batch_size

    def iter_next(self):
        self.cursor += self.batch_size
        return self.cursor < self.num_data

    def next(self):
        if self.iter_next():
            return DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=None)
        else:
            raise StopIteration

    def getdata(self):
        assert(self.cursor < self.num_data), "DataIter needs reset."
        if self.cursor + self.batch_size <= self.num_data:
            return [ x[1][self.cursor:self.cursor + self.batch_size] for x in self.data ]
        else:
            pad = self.batch_size - self.num_data + self.cursor
            return [ scipy.sparse.vstack([x[1][self.cursor:], x[1][:pad]]) for x in self.data ]

    def getlabel(self):
        assert(self.cursor < self.num_data), "DataIter needs reset."
        if self.cursor + self.batch_size <= self.num_data:
            return [ x[1][self.cursor:self.cursor + self.batch_size] for x in self.label ]
        else:
            pad = self.batch_size - self.num_data + self.cursor
            return [ np.concatenate([x[1][self.cursor:], x[1][:pad]]) for x in self.label ]

    def getpad(self):
        if self.last_batch_handle == 'pad' and self.cursor + self.batch_size > self.num_data:
            return self.cursor + self.batch_size - self.num_data
        else:
            return 0


class DataIterLoader():
    """
    DataIter wrapper that handles case where data may stay on disk with iterator
    using mx.io.LibSVMIter for extremely large datasets unable to fit into memory
    (even when using scipy sparse matrices).
    """
    def __init__(self, data_iter=None, data_file=None, col_shape=-1,
                 num_batches=-1, last_batch_size=-1, handle_last_batch='discard'):
        self.using_file = data_iter is None
        self.data_file = data_file
        self.col_shape = col_shape
        self.data_iter = data_iter
        self.num_batches = num_batches
        self.last_batch_size = last_batch_size
        self.handle_last_batch = handle_last_batch
        self.batch_index = 0
        self.batch_size = 1000

    def __iter__(self):
        if not self.using_file:
            self.data_iter.reset()
        else:
            self.data_iter = mx.io.LibSVMIter(data_libsvm=self.data_file, data_shape=(self.col_shape,),
                                              batch_size=self.batch_size)
        self.batch_index = 0
        return self

    def __next__(self):
        batch = self.data_iter.__next__()
        data = mx.nd.sparse.csr_matrix(batch.data[0], dtype='float32')
        if batch.label and len(batch.label) > 0 and len(batch.label[0]) > 0 and batch.data[0].shape[0] == batch.label[0].shape[0]:
            label = mx.nd.array(batch.label[0], dtype='float32')
        else:
            label = None
        self.batch_index += 1
        return data, label

    def get_data(self):
        return self.data_iter.data

    def next(self):
        return self.__next__()


class PairedDataLoader():
    
    def __init__(self, data_loader1, data_loader2):
        self.data_loader1 = data_loader1
        self.data_loader2 = data_loader2
        self.data_iter1   = iter(data_loader1)
        self.data_iter2   = iter(data_loader2) if data_loader2 is not None else None
        self.batch_index = 0
        self.end1         = False
        self.end2         = False


    def __iter__(self):
        self.data_iter1 = iter(self.data_loader1)
        self.data_iter2 = iter(self.data_loader2) if self.data_loader2 is not None else None
        self.batch_index = 0
        self.end1 = False
        self.end2 = False
        return self

    def __next__(self):
        try:
            batch1 = self.data_iter1.__next__()
        except StopIteration:
            if self.end2 or self.data_loader2 is None:
                raise StopIteration
            self.data_iter1 = iter(self.data_loader1)
            self.end1 = True
            batch1 = self.data_iter1.__next__()
        if self.data_loader2 is not None:
            try:
                batch2 = self.data_iter2.__next__()
            except StopIteration:
                if self.end1:
                    raise StopIteration
                self.data_iter2 = iter(self.data_loader2)
                self.end2 = True
                batch2 = self.data_iter2.__next__()
        else:
            batch2 = None
        return batch1, batch2

    def next(self):
        return self.__next__()
    


def _init_data(data, allow_empty, default_name):
    """Convert data into canonical form."""
    assert (data is not None) or allow_empty
    if data is None:
        data = []
    data = OrderedDict([(default_name, data)]) # pylint: disable=redefined-variable-type
    return list(data.items())
    

def load_vocab(vocab_file, encoding='utf-8'):
    """
    Load a pre-derived vocabulary, assumes format consisting of a single word on each line.
    Note: this is a bit of a hack to use a counter to sort the vocab items IN THE ORDER THEY ARE FOUND IN THE FILE.
    """
    w_dict = {}
    words = []
    with io.open(vocab_file, 'r', encoding=encoding) as fp:
        for line in fp:
            els = line.split(' ')
            words.append(els[0].strip())
    ln_wds = len(words)
    for i in range(ln_wds):
        w_dict[words[i]] = ln_wds - i
    counter = nlp.data.Counter(w_dict)
    return nlp.Vocab(counter, unknown_token=None, padding_token=None, bos_token=None, eos_token=None)


def file_to_data(sp_file, voc_size, batch_size=1000):
    with open(sp_file) as f:
        for i, l in enumerate(f):
            pass
    data_size = i+1
    num_batches = data_size // batch_size
    last_batch_size = data_size % batch_size
    X, y = load_svmlight_file(sp_file, n_features=voc_size, dtype='int32', zero_based=True)
    wd_freqs = mx.nd.array(np.array(X.sum(axis=0)).squeeze())
    total_words = X.sum()
    return X, y, wd_freqs, total_words



    
