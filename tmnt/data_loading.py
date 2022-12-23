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


from scipy import sparse as sp
from typing import List, Tuple, Dict, Optional, Union, NoReturn

import torch
from torch.utils.data import DataLoader


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


class SparseDataset():
    """
    Custom Dataset class for scipy sparse matrix
    """
    def __init__(self, 
                 data:Union[np.ndarray, sp.coo_matrix, sp.csr_matrix], 
                 targets: Optional[Union[np.ndarray, sp.coo_matrix, sp.csr_matrix]]):
        
        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == sp.coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data
            
        # Transform targets coo_matrix to csr_matrix for indexing
        if type(targets) == sp.coo_matrix:
            self.targets = targets.tocsr()
        else:
            self.targets = targets
        
    def __getitem__(self, index):
        targets_i = self.targets[index] if self.targets is not None else None
        return self.data[index], targets_i

    def __len__(self):
        return self.data.shape[0]
      
def sparse_coo_to_tensor(coo: sp.coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = (coo.row, coo.col)
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s)
    
def sparse_batch_collate(batch): 
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    # batch[0] since it is returned as a one element list
    data_batch, targets_batch = batch[0]
    
    if type(data_batch[0]) == sp.csr_matrix:
        data_batch = data_batch.tocoo() # removed vstack
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch)

    if targets_batch:
        if type(targets_batch[0]) == sp.csr_matrix:
            targets_batch = targets_batch.tocoo() # removed vstack
            targets_batch = sparse_coo_to_tensor(targets_batch)
        else:
            targets_batch = torch.FloatTensor(targets_batch)
    return data_batch, targets_batch


class SparseDataLoader(DataLoader):

    def __init__(self, 
                 X: Union[sp.csr_matrix, sp.coo_matrix], y: Union[np.array], shuffle=False, drop_last=False,
                 batch_size=1024, device='cpu'):
        self.batch_size = batch_size
        ds = SparseDataset(X, y)
        sampler = torch.utils.data.sampler.BatchSampler(
            torch.utils.data.sampler.RandomSampler(ds,
                                                   generator=torch.Generator(device=device)),
            batch_size=batch_size,
            drop_last=False)
        super().__init__(ds, batch_size=1, collate_fn=sparse_batch_collate, generator=torch.Generator(device=device), sampler=sampler, drop_last=drop_last)
        


class SingletonWrapperLoader():

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter   = iter(data_loader)

    def __iter__(self):
        self.data_iter = iter(self.data_loader)
        return self

    def __len__(self):
        return len(self.data_iter)

    def __next__(self):
        batch = self.data_iter.__next__()
        return (batch,)

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


    def __len__(self):
        if self.data_loader2 is not None:
            return max(len(self.data_loader1), len(self.data_loader2))
        else:
            return len(self.data_loader1)

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



class RoundRobinDataLoader():
    
    def __init__(self, data_loaders):
        self.num_loaders = len(data_loaders)
        self.data_loaders = data_loaders
        self.data_iters = [iter(d) for d in data_loaders]
        self.data_totals = None
        self.ratio_remaining = np.array([1.0 for _ in data_loaders])

    def _get_iter_length(self, it):
        c = 0
        try:
            while True:
                _ = next(it)
                c += 1
        except:
            return c

    def _set_lengths(self, iters):
        self.data_totals = [ self._get_iter_length(it) for it in iters ]
        
    def __iter__(self):
        self._set_lengths( [iter(d) for d in self.data_loaders] )
        self.ratio_remaining[:] = 1.0
        self.data_iters = [iter(d) for d in self.data_loaders]
        return self

    def __len__(self):
        return sum([len(it) for it in self.data_iters])

    def __next__(self):
        it_id = np.argsort(-self.ratio_remaining)[0] ## get iterator with most elements left
        it = self.data_iters[it_id]
        batch = it.__next__()
        self.ratio_remaining[it_id] = ((self.ratio_remaining[it_id] * self.data_totals[it_id]) - 1) / self.data_totals[it_id]
        return batch

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



    
