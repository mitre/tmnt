import io
import os
import time
import argparse
import random
import logging
import warnings
import multiprocessing
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import Block, nn
import gluonnlp as nlp
from gluonnlp.model import get_model
from gluonnlp.data import BERTTokenizer
from gluonnlp.data.dataset import SimpleDataset, Dataset
import json
import collections
from tmnt.preprocess.vectorizer import TMNTVectorizer

from gluonnlp.data import BERTSentenceTransform

class JsonlDataset(SimpleDataset):
    """A dataset wrapping over a jsonlines (.jsonl) file, each line is a json object.

    Parameters
    ----------
    filename : str
        Path to the .jsonl file.
    encoding : str, default 'utf8'
        File encoding format.
    """
    def __init__(self, filename, txt_key, label_key, encoding='utf8'):

        if not isinstance(filename, (tuple, list)):
            filename = (filename, )

        self._filenames = [os.path.expanduser(f) for f in filename]
        self._encoding = encoding
        self._txt_key = txt_key
        self._label_key = label_key

        super(JsonlDataset, self).__init__(self._read())

    def _read(self):
        all_samples = []
        for filename in self._filenames:
            samples = []
            with open(filename, 'r', encoding=self._encoding) as fin:
                for line in fin.readlines():
                    samples.append(json.loads(line, object_pairs_hook=collections.OrderedDict))
            samples = self._read_samples(samples)
            all_samples += samples
        return all_samples

    def _read_samples(self, samples):
        m_samples = []
        for s in samples:
            m_samples.append((s[self._txt_key], s[self._label_key]))
        return m_samples

class UnevenArrayDataset(Dataset):
    """A dataset that combines multiple dataset-like objects, e.g.
    Datasets, lists, arrays, etc. but does NOT require lengths to be the same.

    The i-th sample is defined as `(x1[i % len(x1)], x2[i % len(x2)], ...)`.

    Parameters
    ----------
    *args : one or more dataset-like objects
        The data arrays.
    """
    def __init__(self, *args):
        assert len(args) > 0, "Needs at least 1 arrays"
        self._sub_lengths = [len(a) for a in args]
        self._length = max(self._sub_lengths) # length is equal to maximum subdataset length
        self._data = []
        for i, data in enumerate(args):
            if isinstance(data, mx.nd.NDArray) and len(data.shape) == 1:
                data = data.asnumpy()
            self._data.append(data)

    def __getitem__(self, idx):
        if idx >= self._length:
            raise StopIteration
        if len(self._data) == 1:
            return self._data[0][idx]
        else:
            return tuple(data[idx % data_len] for data,data_len in zip(self._data, self._sub_lengths))

    def __len__(self):
        return self._length
    


class BERTDatasetTransform(object):
    """Dataset transformation for BERT-style sentence classification or regression.

    Parameters
    ----------
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    labels : list of int , float or None. defaults None
        List of all label ids for the classification task and regressing task.
        If labels is None, the default task is regression
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    label_dtype: int32 or float32, default float32
        label_dtype = int32 for classification task
        label_dtype = float32 for regression task
    """

    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 class_labels=None,
                 label_alias=None,
                 pad=True,
                 pair=True,
                 has_label=True,
                 vectorizer=None):
        self.class_labels = class_labels
        self.has_label = has_label
        self._label_dtype = 'int32' if class_labels else 'float32'
        if has_label and class_labels:
            self._label_map = {}
            for (i, label) in enumerate(class_labels):
                self._label_map[label] = i
            if label_alias:
                for key in label_alias:
                    self._label_map[key] = self._label_map[label_alias[key]]
        self._bert_xform = BERTSentenceTransform(
            tokenizer, max_seq_length, pad=pad, pair=pair)
        self.vectorizer = vectorizer


    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.

        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
          sequence or the second sequence.
        - generate valid length

        For sequence pairs, the input is a tuple of 3 strings:
        text_a, text_b and label.

        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
            label: '0'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens:  '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14
            label: 0

        For single sequences, the input is a tuple of 2 strings: text_a and label.
        Inputs:
            text_a: 'the dog is hairy .'
            label: '1'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a:  '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7
            label: 1

        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 3 strings:
            (text_a, text_b, label). For single sequences, the input is a tuple
            of 2 strings: (text_a, label).

        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)
        np.array: classification task: label id in 'int32', shape (batch_size, 1),
            regression task: label in 'float32', shape (batch_size, 1)
        """
        if self.has_label:
            input_ids, valid_length, segment_ids = self._bert_xform(line[:-1])
            label = line[-1]
            # map to int if class labels are available
            if self.class_labels:
                label = self._label_map[label]
            label = np.array([label], dtype=self._label_dtype)
            bow = None
            if self.vectorizer:
                bow,_ = self.vectorizer.transform(line[:-1])
                bow = mx.nd.array(bow, dtype='float32')
            return input_ids, valid_length, segment_ids, bow, label
        else:
            return self._bert_xform(line)


def get_vectorizer(train_json_file, txt_key, label_key, vocab_size=2000):
    vectorizer = TMNTVectorizer(text_key=txt_key, label_key=label_key, vocab_size = vocab_size)
    vectorizer.fit_transform_json(train_json_file)
    return vectorizer



def preprocess_data(tokenizer, vectorizer, class_labels, train_ds, dev_ds, batch_size, dev_batch_size, max_len, pad=False):
    """Train/eval Data preparation function."""
    pool = multiprocessing.Pool()

    # transformation for data train and dev
    label_dtype = 'float32' # if not task.class_labels else 'int32'
    bow_count_dtype = 'float32'
    trans = BERTDatasetTransform(tokenizer, max_len,
                                 class_labels=class_labels,
                                 label_alias=None,
                                 pad=pad, pair=False,
                                 has_label=True,
                                 vectorizer=vectorizer)
    # data train
    data_train = mx.gluon.data.SimpleDataset(pool.map(trans, train_ds))
    data_train_len = data_train.transform(
        lambda input_id, length, segment_id, bow, label_id: length, lazy=False)
    # bucket sampler for training
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype))
    num_buckets = min(6, len(train_ds) // batch_size)
    batch_sampler = nlp.data.sampler.FixedBucketSampler(
        data_train_len,
        batch_size=batch_size,
        num_buckets=num_buckets,
        ratio=0.2, # may avoid batches with size = 1 (which triggers a bug)
        shuffle=True)
    # data loader for training
    loader_train = gluon.data.DataLoader(
        dataset=data_train,
        num_workers=4,
        batch_sampler=batch_sampler,
        batchify_fn=batchify_fn)

    data_dev = mx.gluon.data.SimpleDataset(pool.map(trans, dev_ds))
    loader_dev = mx.gluon.data.DataLoader(
            data_dev,
            batch_size=dev_batch_size,
            num_workers=4,
            shuffle=False,
            batchify_fn=batchify_fn)

    #test_json_file = os.path.join(input_dir, 'test.jsonl')
    #test_ds = JsonlDataset(test_json_file, txt_key="sentence", label_key="label0")
    #data_test = mx.gluon.data.SimpleDataset(pool.map(test_trans, data))
    #loader_test = mx.gluon.data.DataLoader(
    #        data_test,
    #        batch_size=dev_batch_size,
    #        num_workers=4,
    #        shuffle=False,
    #        batchify_fn=test_batchify_fn)
    loader_test = None
    return loader_train, loader_dev, loader_test, len(data_train)


def preprocess_data_metriclearn(tokenizer, class_labels, train_a_json_file, train_b_json_file, batch_size, max_len, pad=False):
    """Train/eval Data preparation function."""
    pool = multiprocessing.Pool()

    vectorizer = get_vectorizer(train_a_json_file, "sentence", "label0")

    # transformation for data train and dev
    label_dtype = 'float32' # if not task.class_labels else 'int32'
    bow_count_dtype = 'float32'
    trans = BERTDatasetTransform(tokenizer, max_len,
                                 class_labels=class_labels,
                                 label_alias=None,
                                 pad=pad, pair=False,
                                 has_label=True,
                                 vectorizer=vectorizer)

    # data train
    train_a_ds = JsonlDataset(train_a_json_file, txt_key="sentence", label_key="label0")
    a_data_train = mx.gluon.data.SimpleDataset(pool.map(trans, train_a_ds))

    # data train
    train_b_ds = JsonlDataset(train_b_json_file, txt_key="sentence", label_key="label0")
    b_data_train = mx.gluon.data.SimpleDataset(pool.map(trans, train_b_ds))

    joined_data_train = UnevenArrayDataset(a_data_train, b_data_train)
    joined_len = joined_data_train.transform( lambda a, b: a[1] + b[1], lazy=False ) ## a[1] and b[1] and lengths, bucket by sum
    batchify_fn = nlp.data.batchify.Tuple(
        ## tuple for a_data
        nlp.data.batchify.Tuple(
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype)),
        ## tuple for b_data
        nlp.data.batchify.Tuple(
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype)))
    batch_sampler = nlp.data.sampler.FixedBucketSampler(
        joined_len,
        batch_size=batch_size,
        num_buckets=10,
        ratio=0,
        shuffle=True)
    loader_train = gluon.data.DataLoader(
        dataset=joined_data_train,
        num_workers=4,
        batch_sampler=batch_sampler,
        batchify_fn=batchify_fn)
    return loader_train, len(joined_data_train)
    

def get_bert_datasets(class_labels,
                      vectorizer,
                      train_ds,
                      dev_ds,
                      model_name,
                      dataset,
                      batch_size,
                      dev_bs,
                      max_len,
                      pad,
                      ctx):
    bert, vocabulary = get_model(
        name=model_name,
        dataset_name=dataset,
        pretrained=True,
        ctx=ctx,
        use_pooler=True,
        use_decoder=False,
        use_classifier=False)
    do_lower_case = 'uncased' in dataset    
    bert_tokenizer = BERTTokenizer(vocabulary, lower=do_lower_case)
    train_data, dev_data, test_data, num_train_examples = preprocess_data(
        bert_tokenizer, vectorizer, class_labels, train_ds, dev_ds, batch_size, dev_bs, max_len, pad)
    return train_data, dev_data, num_train_examples, bert


############
# Handle dataloading for Smoothed Deep Metric Loss with parallel batching
############

## 


        
