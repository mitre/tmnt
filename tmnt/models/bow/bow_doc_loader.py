# coding: utf-8

"""
Copyright (c) 2019 The MITRE Corporation.

File/module contains routines for loading in text documents to sparse matrix representations
for efficient neural variational model training.
"""

import io
import itertools
import os
import logging

import gluonnlp as nlp
import mxnet as mx
import numpy as np
from collections import OrderedDict
from gluonnlp.data import SimpleDatasetStream, CorpusDataset
from tmnt.preprocess.tokenizer import BasicTokenizer
from mxnet.io import DataDesc, DataIter, DataBatch
import scipy
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle as sk_shuffle

__all__ = ['DataIterLoader', 'file_to_data', 'collect_sparse_test', 'collect_sparse_data', 'BowDataSet', 'collect_stream_as_sparse_matrix',
           'get_single_vec', 'load_vocab', 'SparseMatrixDataIter']

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


def preprocess_dataset_stream(stream, pre_vocab = None, min_freq=3, max_vocab_size=None):
    if pre_vocab:
        vocab = pre_vocab
    else:
        counter = None
        i = 0
        for data in iter(stream):
            counter = nlp.data.count_tokens(itertools.chain.from_iterable(data), counter = counter)
            i += 1
            if i % 100 == 0:
                logging.info("[ Pre-processed {} documents from training collection ]".format(i))
        logging.info("[ {} total documents processed .. ]".format(i))
        vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                              bos_token=None, eos_token=None, min_freq=min_freq,
                              max_size=max_vocab_size)
        orig_vocab_size = len(counter)
        vocab_size = len(vocab)
        logging.info("[ Original vocab size {}, reduced to {} ]".format(orig_vocab_size, vocab_size))
    
    
    def code(doc):
        """
        Parameters
        ----------
        Token sequence for all tokens in a file/document

        Returns
        -------
        Token ids with associated frequencies (sparse vector)
        """
        doc_tok_ids = [vocab[token] for token in doc if token in vocab]
        doc_counter = nlp.data.count_tokens(doc_tok_ids)        
        return sorted(doc_counter.items())

    def code_corpus(corpus):
        return corpus.transform(code)

    stream = stream.transform(code_corpus)
    return stream, vocab


def collect_stream_as_sparse_matrix(stream, pre_vocab=None, min_freq=3, max_vocab_size=None):
    """
    Read in a `DatasetStream` (specifically a `BowDataSet`) and load into main memory as a sparse matrix
    If no vocabulary is provided, one will be constructed from the data; otherwise it will be used    
    """
    strm, vocab = preprocess_dataset_stream(stream, pre_vocab=pre_vocab, min_freq=min_freq, max_vocab_size=max_vocab_size)
    indices = []
    values = []
    indptrs = [0]
    cumulative = 0
    ndocs = 0
    total_num_words = 0
    for i,doc in enumerate(strm):
        doc_li = list(doc)
        if len(doc_li) > 0:
            doc_toks = doc_li[0]  # each document is a single sample
            if len(doc_toks) < 2:
                continue
            ndocs += 1            
            inds, vs = zip(*doc_toks)
            ln = len(doc_toks)
            cumulative += ln
            total_num_words += sum(vs)
            indptrs.append(cumulative)
            values.extend(vs)
            indices.extend(inds)
    csr_mat = mx.nd.sparse.csr_matrix((values, indices, indptrs), shape = (ndocs, len(vocab)))
    return csr_mat, vocab, total_num_words


class DataIterLoader():
    """
    This is a simple wrapper around a `DataIter` geared for unsupervised learning datasets
    where the label is `None`. Not totally necessary but means means client code (e.g. in the `train_model`
    method can follow the standard API for `DataLoader` objects.
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
        if batch.data[0].shape[0] == batch.label[0].shape[0]:            
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
    


class BowDataSet(SimpleDatasetStream):
    def __init__(self, root, pattern, sampler='random'):
        self._root = root
        self._file_pattern = os.path.join(root, pattern)
        self.codec = 'utf-8'
        super(BowDataSet, self).__init__(
            dataset=CorpusDataset,
            file_pattern = self._file_pattern,
            file_sampler=sampler,
            tokenizer=BasicTokenizer(),
            sample_splitter=NullSplitter())
        

class NullSplitter(nlp.data.Splitter):
    """
    This splitter takes an entire document and returns it as a single list rather than
    splitting up into sentences.
    """
    def __call__(self, s):
        return [s]

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

def get_single_vec(els_sp):
    pairs = sorted( [ (int(el[0]), float(el[1]) ) for el in els_sp ] )
    inds, vs = zip(*pairs)
    return pairs, inds, vs
    

def file_to_sp_vec(sp_file, voc_size, label_map=None, scalar_labels=False, encoding='utf-8'):
    labels = []
    indices = []
    values = []
    indptrs = [0]
    cumulative = 0
    total_num_words = 0
    ndocs = 0
    lm = label_map if label_map and not scalar_labels else {}
    freqs = np.zeros(voc_size)
    with io.open(sp_file, 'r', encoding=encoding) as fp:
        for line in fp:
            els = line.split(' ')            
            lstr = els[0]
            if scalar_labels:
                label = float(lstr)
            else:
                try:
                    label = lm[lstr]
                except KeyError:
                    if label_map is None:
                        label = len(lm)
                        lm[lstr] = label
                    else:
                        label = -1
            ndocs += 1
            els_sp = list(map(lambda e: e.split(':'), els[1:]))
            pairs, inds, vs = get_single_vec(els_sp)
            for i,v in pairs:
                freqs[i] += v
            cumulative += len(pairs)
            total_num_words += sum(vs)
            labels.append(label)
            indptrs.append(cumulative)
            values.extend(vs)
            indices.extend(inds)
    csr_mat = mx.nd.sparse.csr_matrix((values, indices, indptrs), shape = (ndocs, voc_size))
    lm = None if len(lm) < 1 else lm
    return csr_mat, total_num_words, labels, lm, mx.nd.array(freqs)


def file_to_data(sp_file, voc_size, batch_size=1000):
    with open(sp_file) as f:
        for i, l in enumerate(f):
            pass
    data_size = i+1
    num_batches = data_size // batch_size
    last_batch_size = data_size % batch_size
    print("Number of batches = {}; last batch size = {}".format(num_batches, last_batch_size))
    X, y = load_svmlight_file(sp_file, n_features=voc_size, dtype='int32')
    wd_freqs = mx.nd.array(np.array(X.sum(axis=0)).squeeze())
    total_words = X.sum()
    return X, y, wd_freqs, total_words


def normalize_scalar_values(scalars):
    return (scalars - scalars.min()) / (scalars.max() - scalars.min())


def collect_sparse_data(sp_file, voc_size, scalar_labels=False, encoding='utf-8'):
    #vocab = load_vocab(vocab_file, encoding=encoding)
    X, total_words, tr_labels_li, label_map, wd_freqs = file_to_sp_vec(sp_file, voc_size, scalar_labels=scalar_labels, encoding=encoding)
    dt = 'float32' if scalar_labels else 'int'
    tr_labels = mx.nd.array(tr_labels_li, dtype=dt)
    if scalar_labels:
        tr_labels = normalize_scalar_values(tr_labels)
    return X, tr_labels, wd_freqs, total_words, label_map
    

def collect_sparse_test(sp_file, voc_size, scalar_labels=False, label_map=None, encoding='utf-8'):
    keep_sp_sparse = True
    tst_mat_sp, total_tst, tst_labels_li, _, wd_freqs = \
        file_to_sp_vec(sp_file, voc_size, label_map=label_map, scalar_labels=scalar_labels, encoding=encoding)
    X = tst_mat_sp if keep_sp_sparse else tst_mat_sp.tostype('default')
    dt = 'float32' if scalar_labels else 'int'    
    tst_labels = mx.nd.array(tst_labels_li, dtype=dt)
    if scalar_labels:
        tst_labels = normalize_scalar_values(tst_labels)
    return X, tst_labels, wd_freqs, total_tst, _
