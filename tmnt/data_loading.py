# coding: utf-8

"""
Copyright (c) 2019-2020. The MITRE Corporation.

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
        if batch.label and len(batch.label) > 0 and batch.data[0].shape[0] == batch.label[0].shape[0]:            
            label = mx.nd.array(batch.label[0], dtype='float32')
        else:
            label = None
        self.batch_index += 1
        return data, label

    def get_data(self):
        return self.data_iter.data

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
    X, y = load_svmlight_file(sp_file, n_features=voc_size, dtype='int32')
    wd_freqs = mx.nd.array(np.array(X.sum(axis=0)).squeeze())
    total_words = X.sum()
    return X, y, wd_freqs, total_words
    

trans_table = str.maketrans(dict.fromkeys(string.punctuation))

def remove_punct_and_urls(txt):
    string = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', '', txt) ## wipe out URLs
    return string.translate(trans_table)

def get_single_vec(els_sp):
    pairs = sorted( [ (int(el[0]), float(el[1]) ) for el in els_sp ] )
    inds, vs = zip(*pairs)
    return pairs, inds, vs


def load_dataset_bert(json_file, voc_size, json_text_key="text", json_sp_key="sp_vec", max_len=64, ctx=mx.cpu()):
    indices = []
    values = []
    indptrs = [0]
    cumulative = 0
    total_num_words = 0
    ndocs = 0
    bert_model = 'bert_12_768_12'
    dname = 'book_corpus_wiki_en_uncased'
    bert_base, vocab = nlp.model.get_model(bert_model,  
                                             dataset_name=dname,
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)
    tokenizer = BERTTokenizer(vocab)
    transform = BERTSentenceTransform(tokenizer, max_len, pair=False) 
    x_ids = []
    x_val_lens = []
    x_segs = []
    with io.open(json_file, 'r', encoding='utf-8') as fp:
        for line in fp:
            if json_text_key:
                js = json.loads(line)
                line = js[json_text_key]
            ids, lens, segs = transform((line,)) # create BERT-ready inputs
            x_ids.append(ids)
            x_val_lens.append(lens)
            x_segs.append(segs)
            ## Now, get the sparse vector
            ndocs += 1
            sp_vec_els = js[json_sp_key]
            pairs, inds, vs = get_single_vec(sp_vec_els)
            cumulative += len(pairs)
            total_num_words += sum(vs)
            indptrs.append(cumulative)
            values.extend(vs)
            indices.extend(inds)
    csr_mat = mx.nd.sparse.csr_matrix((values, indices, indptrs), shape=(ndocs, voc_size))
    data_train = gluon.data.ArrayDataset(
        mx.nd.array(x_ids, dtype='int32'),
        mx.nd.array(x_val_lens, dtype='int32'),
        mx.nd.array(x_segs, dtype='int32'),
        csr_mat.tostype('default'))
    return data_train, bert_base, vocab, csr_mat
