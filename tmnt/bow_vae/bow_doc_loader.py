# coding: utf-8

"""
File/module contains routines for loading in text documents to sparse matrix representations
for efficient neural variational model training.
"""

import mxnet as mx
import mxnet.ndarray as F
import codecs
import itertools
import gluonnlp as nlp
import functools
import warnings
import os

from gluonnlp.data import SimpleDatasetStream, CorpusDataset


def preprocess_dataset_stream(stream, min_freq=3, max_vocab_size=None):
    counter = None
    for data in iter(stream):
        counter = nlp.data.count_tokens(itertools.chain.from_iterable(data), counter = counter)
        #logging.info('.. counter size = {} ..'.format(str(len(counter))))
    vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                          bos_token=None, eos_token=None, min_freq=min_freq,
                          max_size=max_vocab_size)
    idx_to_counts = [counter[w] for w in vocab.idx_to_token]

    def code(doc):
        """
        Parameters
        ----------
        Token sequence for all tokens in a file/document

        Returns
        -------
        Token ids with associated frequencies (sparse vector)
        """
        ## just drop out of vocab items
        doc_tok_ids = [vocab[token] for token in doc if token in vocab]
        doc_counter = nlp.data.count_tokens(doc_tok_ids)        
        return sorted(doc_counter.items())

    def code_corpus(corpus):
        return corpus.transform(code)

    stream = stream.transform(code_corpus) 
    return stream, vocab, idx_to_counts


def collect_stream_as_sparse_matrix(stream, min_freq=3, max_vocab_size=None):
    strm, vocab, idx_to_counts = preprocess_dataset_stream(stream, min_freq, max_vocab_size)
    indices = []
    values = []
    indptrs = [0]
    cumulative = 0
    ndocs = 0
    total_num_words = 0
    for i,doc in enumerate(strm):
        ndocs += 1
        doc_toks = list(doc)[0]  # each document is a single sample        
        inds, vs = zip(*doc_toks)
        ln = len(doc_toks)
        cumulative += ln
        total_num_words += sum(vs)
        indptrs.append(cumulative)
        values.extend(vs)
        indices.extend(inds)
    # can use this with NDArrayIter
    # dataiter = mx.io.NDArrayIter(data, labels, batch_size, last_batch_handle='discard')
    ## inspect - [ batch.data[0] for batch in dataiter ]
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
    def __init__(self, root, pattern):
        self._root = root
        self._file_pattern = os.path.join(root, pattern)
        self.codec = 'utf-8'
        super(BowDataSet, self).__init__(
            dataset=CorpusDataset,
            file_pattern = self._file_pattern,
            file_sampler='random',
            sample_splitter=NullSplitter())

class NullSplitter(nlp.data.Splitter):

    def __call__(self, s):
        return [s]


def load_vocab(vocab_file):
    w_dict = {}
    with io.open(vocab_file, 'r') as fp:
        for line in fp.read():
            els = line.split(' ')
            w_dict[els[0]] = els[1]
    return nlp.Vocab(nlp.data.Counter(w_dict))

def file_to_sp_vec(sp_file):
    labels = []
    indices = []
    values = []
    indptrs = [0]
    cumulative = 0
    total_num_words = 0
    with io.open(sp_file, 'r') as fp:
        for line in fp.read():
            els = line.split(' ')
            labels.append(els[0])
            els_sp = map(els, lambda e: e.split(':'))
            pairs = sorted( [ (el[0], el[1] ) for el in els_sp[1:] ] )
            inds, vs = zip(*pairs)
            cumulative += len(pairs)
            total_num_words += sum(vs)
            indptrs.append(cumulative)
            values.extend(vs)
            indices.extend(inds)
    csr_mat = mx.nd.sparse.csr_matrix((values, indices, indptrs), shape = (ndocs, len(vocab)))
    return csr_mat, total_num_words
                
                

def collect_sparse_data(sp_vec_file, vocab_file, sp_vec_test_file=None):
    vocab = load_vocab(vocab_file)
    tr_mat, total_tr = file_to_sp_vec(sp_vec_file)
    return tr_mat, total_tr, vocab
    
