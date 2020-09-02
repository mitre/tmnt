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
from gluonnlp.data import SimpleDatasetStream, CorpusDataset
from tmnt.preprocess.tokenizer import BasicTokenizer


__all__ = ['DataIterLoader', 'collect_sparse_test', 'collect_sparse_data', 'BowDataSet', 'collect_stream_as_sparse_matrix',
           'get_single_vec', 'load_vocab']

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
    def __init__(self, data_iter):
        self.data_iter = data_iter

    def __iter__(self):
        self.data_iter.reset()
        return self

    def __next__(self):
        batch = self.data_iter.__next__()
        data = batch.data[0]
        if len(batch.data) == len(batch.label):            
            label = batch.label[0]
        else:
            label = None
        return data, label

    def next(self):
        return self.__next__()
    

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


def normalize_scalar_values(scalars):
    return (scalars - scalars.min()) / (scalars.max() - scalars.min())


def collect_sparse_data(sp_vec_file, vocab_file, sp_vec_test_file=None, scalar_labels=False, encoding='utf-8'):
    vocab = load_vocab(vocab_file, encoding=encoding)
    tr_mat, total_tr, tr_labels_li, label_map, wd_freqs = file_to_sp_vec(sp_vec_file, len(vocab), scalar_labels=scalar_labels, encoding=encoding)
    dt = 'float32' if scalar_labels else 'int'
    tr_labels = mx.nd.array(tr_labels_li, dtype=dt)
    if scalar_labels:
        tr_labels = normalize_scalar_values(tr_labels)
    #tr_map = tr_mat.tostype('default')
    return vocab, tr_mat, total_tr, tr_labels, label_map, wd_freqs
    

def collect_sparse_test(sp_vec_file, vocab, scalar_labels=False, label_map=None, encoding='utf-8'):
    keep_sp_sparse = True
    tst_mat_sp, total_tst, tst_labels_li, _, _ = \
        file_to_sp_vec(sp_vec_file, len(vocab), label_map=label_map, scalar_labels=scalar_labels, encoding=encoding)
    tst_mat = tst_mat_sp if keep_sp_sparse else tst_mat_sp.tostype('default')
    dt = 'float32' if scalar_labels else 'int'    
    tst_labels = mx.nd.array(tst_labels_li, dtype=dt)
    if scalar_labels:
        tst_labels = normalize_scalar_values(tst_labels)
    return tst_mat, total_tst, tst_labels
