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
from tmnt.preprocess.vectorizer import TMNTVectorizer


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
    

def get_single_vec(els_sp):
    pairs = sorted( [ (int(el[0]), float(el[1]) ) for el in els_sp ] )
    inds, vs = zip(*pairs)
    return pairs, inds, vs


def _load_dataset_bert(line_gen, voc_size, max_len=64, ctx=mx.cpu()):
    indices = []
    values = []
    indptrs = [0]
    cumulative = 0
    total_num_words = 0
    ndocs = 0
    bert_model = 'bert_12_768_12'
    dname = 'book_corpus_wiki_en_uncased'
    ## This is really only needed here to get the vocab
    ## GluonNLP API doesn't enable that
    bert_base, vocab = nlp.model.get_model(bert_model,  
                                             dataset_name=dname,
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)
    tokenizer = BERTTokenizer(vocab)
    transform = BERTSentenceTransform(tokenizer, max_len, pair=False) 
    x_ids = []
    x_val_lens = []
    x_segs = []
    for t in line_gen:
        if isinstance(t, tuple):
            line = t[0]
            sp_vec_els = t[1]
        else:
            line = t
            sp_vec_els = None
        ids, lens, segs = transform((line,)) # create BERT-ready inputs
        x_ids.append(ids)
        x_val_lens.append(lens)
        x_segs.append(segs)
        ## Now, get the sparse vector
        ndocs += 1
        if sp_vec_els:
            pairs, inds, vs = get_single_vec(sp_vec_els)
            cumulative += len(pairs)
            total_num_words += sum(vs)
            indptrs.append(cumulative)
            values.extend(vs)
            indices.extend(inds)
    if len(indices) > 0:
        csr_mat = mx.nd.sparse.csr_matrix((values, indices, indptrs), shape=(ndocs, voc_size)).tostype('default')
    else:
        csr_mat = None
    return x_ids, x_val_lens, x_segs, bert_base, vocab, csr_mat


def prepare_bert(content, max_len, bow_vocab_size=1000, vectorizer=None, ctx=mx.cpu()):
    """
    Utility function to take text content (e.g. list of document strings), a maximum sequence
    length and vocabulary size, returning a data_train object that can be used
    by a SeqBowEstimator object for the call to fit_with_validation. Also returns
    the BOW matrix as a SciPy sparse matrix along with the BOW vocabulary.
    """
    x_ids, x_val_lens, x_segs, bert_base, bert_vocab, _ = _load_dataset_bert(content, 0, max_len, ctx)
    tf_vectorizer = vectorizer or TMNTVectorizer(vocab_size = bow_vocab_size)
    X, _ = tf_vectorizer.transform(content) if vectorizer else tf_vectorizer.fit_transform(content)
    data_train = gluon.data.ArrayDataset(
            mx.nd.array(x_ids, dtype='int32'),
            mx.nd.array(x_val_lens, dtype='int32'),
            mx.nd.array(x_segs, dtype='int32'),
            mx.nd.sparse.csr_matrix(X, dtype='float32').tostype('default'))
    return data_train, X, tf_vectorizer, bert_base, bert_vocab


def prepare_bert_via_json(json_file, max_len, bow_vocab_size=1000, vectorizer=None, json_text_key="text", json_label_key=None, ctx=mx.cpu()):
    with io.open(json_file, 'r', encoding='utf-8') as fp:
        content = [json.loads(line)[json_text_key] for line in fp]
        x_ids, x_val_lens, x_segs, bert_base, bert_vocab, _ = _load_dataset_bert(content, 0, max_len, ctx)
        tf_vectorizer = vectorizer or TMNTVectorizer(text_key=json_text_key, label_key=json_label_key, vocab_size = bow_vocab_size)
        X, y = tf_vectorizer.transform_json(json_file) if vectorizer else tf_vectorizer.fit_transform_json(json_file)
        data_train = gluon.data.ArrayDataset(
            mx.nd.array(x_ids, dtype='int32'),
            mx.nd.array(x_val_lens, dtype='int32'),
            mx.nd.array(x_segs, dtype='int32'),
            mx.nd.sparse.csr_matrix(X, dtype='float32').tostype('default'))
    return data_train, X, tf_vectorizer, bert_base, bert_vocab, y
    


def load_dataset_bert(json_file, voc_size, json_text_key="text", json_sp_key="sp_vec", max_len=64, ctx=mx.cpu()):
    with io.open(json_file, 'r', encoding='utf-8') as fp:
        line_gen = ((json.loads(line)[json_text_key],json.loads(line)[json_sp_key]) for line in fp)
        x_ids, x_val_lens, x_segs, bert_base, vocab, csr_mat =  _load_dataset_bert(line_gen, voc_size, max_len, ctx)
        data_train = gluon.data.ArrayDataset(
            mx.nd.array(x_ids, dtype='int32'),
            mx.nd.array(x_val_lens, dtype='int32'),
            mx.nd.array(x_segs, dtype='int32'),
            csr_mat)
    return data_train, bert_base, vocab, csr_mat


## loading for non-BERT seq2vec encoders with embeddings

def _load_dataset_sequence(line_gen, max_len, tokenizer, vocab):
    toked_lines = [tokenizer(line)[:max_len] for line in line_gen]
    wd_ids = []
    for tl in toked_lines:
        wl = []
        for w in tl:
            try:
                i = vocab(w)
                wl.append(i)
            except Exception:
                pass
        wd_ids.append(wl)
    lens = [len(line) for line in wd_ids]
    for line_ids in wd_ids:
        line_ids.extend([0] * (max_len - len(line_ids)))
    return wd_ids, lens


def _load_bow_identical_sequence(X, max_len):
    from tmnt.classifier.load_data import _sv_to_seq
    seqs = []
    lens = []
    for i in range(X.shape[0]):
        seq, _ = _sv_to_seq(X[i])
        slen = len(seq[:max_len])
        lens.append(slen)
        seqs.append(seq[:max_len] + ([0] * (max_len - slen)))
    return seqs, lens


def prepare_dataset_sequence(content, max_len, labels=None, tokenizer=None, bow_vocab_size=1000, vectorizer=None, ctx=mx.cpu()):
    tf_vectorizer = vectorizer or TMNTVectorizer(vocab_size = bow_vocab_size)
    if tokenizer is None:
        tokenizer = nlp.data.SacreMosesTokenizer()
    X, _ = tf_vectorizer.transform(content) if vectorizer else tf_vectorizer.fit_transform(content)
    vocab = tf_vectorizer.get_vocab()
    #x_ids, x_val_lens = _load_dataset_sequence(content, max_len, tokenizer, vocab)
    x_ids, x_val_lens = _load_bow_identical_sequence(X, max_len)
    if labels is not None:
        larr = mx.nd.array(labels, dtype='float32')
    else:
        larr = mx.nd.full(len(x_ids), -1)
    data_train = gluon.data.ArrayDataset(
        mx.nd.array(x_ids, dtype='int32'),
        mx.nd.array(x_val_lens, dtype='int32'),
        mx.nd.sparse.csr_matrix(X, dtype='float32').tostype('default'),
        larr)
    return data_train, tf_vectorizer.get_vocab(), X
    
    


