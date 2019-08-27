import io
import itertools
import os
import logging

import gluonnlp as nlp
import mxnet as mx


def tokenize(txt):
    """
    Tokenize an input string. Something more sophisticated may help . . . 
    """
    return txt.split(' ')

def build_data(tr_file, get_vocab=True):
    all_tokens = []
    tr_array = []
    with open(tr_file, 'r') as fp:
        for text in fp:
            tokens = tokenize(text)
            tr_array.append(tokens)
            if get_vocab:
                all_tokens.extend(tokens)
    if get_vocab:
        counter = nlp.data.count_tokens(all_tokens)
        vocab = nlp.Vocab(counter)
    else:
        vocab = None
    return (tr_array, vocab)


def load_dataset(train_file, val_file, max_len=64):
    tr_data, vocabulary = build_data(train_file, get_vocab=True)
    tst_data, _ = build_data(val_file, get_vocab=False)
    tr_vecs = preprocess_dataset(tr_data, vocabulary, max_len)
    tst_vecs = preprocess_dataset(tst_data, vocabulary, max_len)
    return tr_vecs, tst_vecs, vocabulary


def _preprocess(x, vocab, max_len):
    """
    Inputs: data instance x (tokenized), vocabulary, maximum length of input (in tokens)
    Outputs: data mapped to token IDs, with corresponding label
    """
    x = x[:max_len]
    data = vocab[x]   ## map tokens (strings) to unique IDs
    return data

def preprocess_dataset(dataset, vocab, max_len):
    preprocessed_dataset = [ _preprocess(x, vocab, max_len) for x in dataset ]
    return preprocessed_dataset


class BasicTransform(object):
    """
    This is a callable object used by the transform method for a dataset. It will be
    called during data loading/iteration.  Added here as a general hook to 
    add additional data transforms if needed.

    Parameters
    ----------
    max_len : int, default 64
        Maximum sequence length - longer seqs will be truncated and shorter ones padded
    
    """
    def __init__(self, max_len=64, pad_id=0):
        self._max_seq_length = max_len
        self._pad_id = pad_id
    
    def __call__(self, data):
        padded_data = data + [self._pad_id] * (self._max_seq_length - len(data))
        return mx.nd.array(padded_data, dtype='int32')


