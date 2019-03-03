# coding: utf-8

"""
File/module contains routines for loading in text documents to sparse matrix representations
for efficient neural variational model training.
"""

import io
import itertools
import os
import logging

import gluonnlp as nlp
import mxnet as mx
from gluonnlp.data import SimpleDatasetStream, CorpusDataset

from tmnt.preprocess.tokenizer import BasicTokenizer


__all__ = ['DataIterLoader', 'collect_sparse_data', 'BowDataSet', 'collect_stream_as_sparse_matrix']

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

def collect_texts_as_sparse_matrix(texts, vocab):
    for doc in txts:
        doc_tok_ids = [vocab[token] for token in doc if token in vocab]
        doc_counter = nlp.data.count_tokens(doc_tok_ids)        
        items = sorted(doc_counter.items())
    

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


def load_vocab(vocab_file):
    """
    Load a pre-derived vocabulary, assumes format consisting of "word id" on each line
    """
    w_dict = {}
    with io.open(vocab_file, 'r') as fp:
        for line in fp:
            els = line.split(' ')
            cnt = int(els[1]) if len(els) > 1 else 1
            w_dict[els[0].strip()] = cnt
    counter = nlp.data.Counter(w_dict)
    return nlp.Vocab(counter, unknown_token=None, padding_token=None, bos_token=None, eos_token=None)

def file_to_sp_vec(sp_file, voc_size):
    labels = []
    indices = []
    values = []
    indptrs = [0]
    cumulative = 0
    total_num_words = 0
    ndocs = 0
    with io.open(sp_file, 'r') as fp:
        for line in fp:
            ndocs += 1
            els = line.split(' ')
            labels.append(int(els[0]))
            els_sp = list(map(lambda e: e.split(':'), els))
            pairs = sorted( [ (int(el[0]), float(el[1]) ) for el in els_sp[1:] ] )
            inds, vs = zip(*pairs)
            cumulative += len(pairs)
            total_num_words += sum(vs)
            indptrs.append(cumulative)
            values.extend(vs)
            indices.extend(inds)
    csr_mat = mx.nd.sparse.csr_matrix((values, indices, indptrs), shape = (ndocs, voc_size))
    return csr_mat, total_num_words, labels
                

def collect_sparse_data(sp_vec_file, vocab_file, sp_vec_test_file=None):
    vocab = load_vocab(vocab_file)
    tr_mat, total_tr, tr_labels_li = file_to_sp_vec(sp_vec_file, len(vocab))
    tr_labels = mx.nd.array(tr_labels_li, dtype='int') - 1
    if sp_vec_test_file:
        tst_mat, total_tst, tst_labels_li = file_to_sp_vec(sp_vec_test_file, len(vocab))
        tst_labels = mx.nd.array(tst_labels_li, dtype='int') - 1
    else:
        tst_mat = None
        tst_labels = None
        total_tst = 0    
    return vocab, tr_mat, total_tr, tst_mat, total_tst, tr_labels, tst_labels
    
