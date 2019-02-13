# coding: utf-8

"""
File/module contains routines for loading in text documents to sparse matrix representations
for efficient neural variational model training.
"""

import io
import itertools
import os

import gluonnlp as nlp
import mxnet as mx
from gluonnlp.data import SimpleDatasetStream, CorpusDataset


__all__ = ['DataIterLoader', 'collect_sparse_data', 'BowDataSet', 'collect_stream_as_sparse_matrix']

def preprocess_dataset_stream(stream, pre_vocab = None, min_freq=3, max_vocab_size=None):
    if pre_vocab:
        vocab = pre_vocab
    else:
        counter = None
        for data in iter(stream):
            counter = nlp.data.count_tokens(itertools.chain.from_iterable(data), counter = counter)
            #logging.info('.. counter size = {} ..'.format(str(len(counter))))
            vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                              bos_token=None, eos_token=None, min_freq=min_freq,
                              max_size=max_vocab_size)

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
    return stream, vocab


def collect_stream_as_sparse_matrix(stream, pre_vocab=None, min_freq=3, max_vocab_size=None):
    """
    Read in a `DatasetStream` and read into main memory as a sparse matrix
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
        for line in fp:
            els = line.split(' ')
            w_dict[els[0]] = int(els[1])
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
            pairs = sorted( [ (int(el[0]), int(el[1]) ) for el in els_sp[1:] ] )
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
    tr_mat, total_tr, _ = file_to_sp_vec(sp_vec_file, len(vocab))

    if sp_vec_test_file:
        tst_mat, total_tst, _ = file_to_sp_vec(sp_vec_test_file, len(vocab))
    else:
        tst_mat = None
        total_tst = 0    
    return vocab, tr_mat, total_tr, tst_mat, total_tst
    