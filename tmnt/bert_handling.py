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
    label_remap : dict
        Dictionary to map labels.
    """
    def __init__(self, filename, txt_key, label_key, encoding='utf8', label_remap=None, random_drop_pct=0.0):

        if not isinstance(filename, (tuple, list)):
            filename = (filename, )

        self._filenames = [os.path.expanduser(f) for f in filename]
        self._encoding = encoding
        self._txt_key = txt_key
        self._label_key = label_key
        self._label_remap = label_remap
        self._random_drop_pct = random_drop_pct
        self._random_drop = random_drop_pct > 0.0
        super(JsonlDataset, self).__init__(self._read())

    def _read(self):
        all_samples = []
        for filename in self._filenames:
            samples = []
            with open(filename, 'r', encoding=self._encoding) as fin:
                for line in fin.readlines():
                    if not self._random_drop or (random.uniform(0,1) > self._random_drop_pct):
                        s = json.loads(line, object_pairs_hook=collections.OrderedDict)
                        label = s.get(self._label_key)
                        if self._label_remap is not None:
                            label = self._label_remap.get(label)
                        samples.append((s[self._txt_key], label))
            all_samples += samples
        return all_samples
    

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
    has_label: bool.
        Whether labels are present for supervised learning
    vectorizer: TMNTVectorizer
        TMNTVectorizer to generate bag of words
    bert_vocab_size: int
        Use the raw BERT word-pieces as the bag-of-words vocabulary
    """

    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 class_labels=None,
                 label_alias=None,
                 pad=True,
                 pair=True,
                 has_label=True,
                 vectorizer=None,
                 bert_vocab_size=0):
        self.class_labels = class_labels
        self.has_label = has_label
        self.use_bert_bow = bert_vocab_size > 0
        self.bert_vocab_size = bert_vocab_size
        self._label_dtype = 'int32' if class_labels else 'float32'
        if has_label and class_labels:
            self._label_map = {}
            for (i, label) in enumerate(class_labels):
                self._label_map[label] = i
            if label_alias:
                for key in label_alias:
                    if label_alias[key] in self._label_map:
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
        np.array: input token ids in 'int32', shape (seq_length,)
        np.array: valid length in 'int32', shape (1,)
        np.array: input token type ids in 'int32', shape (seq_length,)
        np.array: classification task: label id in 'int32', shape (1,),
            regression task: label in 'float32', shape (1,)
        """
        if self.has_label:
            input_ids, valid_length, segment_ids = self._bert_xform(line[:-1])
            label = line[-1]
            # map to int if class labels are available
            if self.class_labels:
                label = self._label_map.get(label)
                if label is None:
                    label = -1
            label = np.array([label], dtype=self._label_dtype)
            bow = None
            if self.use_bert_bow:
                bow = np.zeros(self.bert_vocab_size)
                inds, cnts = np.unique(input_ids, return_counts=True)
                bow[inds] = cnts
                bow = mx.nd.array(np.expand_dims(bow, 0), dtype='float32')
            elif self.vectorizer:
                bow,_ = self.vectorizer.transform(line[:-1])
                bow = mx.nd.array(bow, dtype='float32')
            return input_ids, valid_length, segment_ids, bow, label
        else:
            return self._bert_xform(line)


def preprocess_data(trans, class_labels, train_ds, dev_ds, batch_size, max_len,
                    pad=False):
    """Train/eval Data preparation function."""
    pool = multiprocessing.Pool()

    # transformation for data train and dev
    label_dtype = 'float32' # if not task.class_labels else 'int32'
    bow_count_dtype = 'float32'
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
            batch_size=batch_size,
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


def get_bert_datasets(class_labels,
                      vectorizer,
                      train_ds,
                      dev_ds,
                      batch_size,
                      max_len,
                      bert_model_name = 'bert_12_768_12',
                      bert_dataset = 'book_corpus_wiki_en_uncased',
                      pad=False,
                      use_bert_vocab=False,
                      label_alias=None,
                      ctx=mx.cpu()):
    bert, bert_vocabulary = get_model(
        name=bert_model_name,
        dataset_name=bert_dataset,
        pretrained=True,
        ctx=ctx,
        use_pooler=True,
        use_decoder=False,
        use_classifier=False)
    do_lower_case = 'uncased' in bert_dataset    
    bert_tokenizer = BERTTokenizer(bert_vocabulary, lower=do_lower_case)
    trans = BERTDatasetTransform(bert_tokenizer, max_len,
                                 class_labels=class_labels,
                                 label_alias=label_alias,
                                 pad=pad, pair=False,
                                 has_label=True,
                                 vectorizer=vectorizer,
                                 bert_vocab_size = len(bert_vocabulary) if use_bert_vocab else 0)
    train_data, dev_data, test_data, num_train_examples = preprocess_data(
        trans, class_labels, train_ds, dev_ds, batch_size, max_len, pad)
    return train_data, dev_data, num_train_examples, bert, bert_vocabulary


############
# Handle dataloading for Smoothed Deep Metric Loss with parallel batching
############

def preprocess_data_metriclearn(trans, class_labels, train_a_ds, train_b_ds, batch_size, max_len, pad=False, bucket_sample=False):
    """Train/eval Data preparation function."""
    pool = multiprocessing.Pool()
    label_dtype = 'float32' # if not task.class_labels else 'int32'
    bow_count_dtype = 'float32'

    a_data_train = mx.gluon.data.SimpleDataset(pool.map(trans, train_a_ds))
    b_data_train = mx.gluon.data.SimpleDataset(pool.map(trans, train_b_ds))

    # magic that "zips" these two datasets and pairs batches
    joined_data_train = UnevenArrayDataset(a_data_train, b_data_train)
    joined_len = joined_data_train.transform( lambda a, b: a[1] + b[1], lazy=False ) ## a[1] and b[1] and lengths, bucket by sum
    batchify_fn = nlp.data.batchify.Tuple(
        ## tuple for a_data: (ids, lengths, segments, bow vector, label)
        nlp.data.batchify.Tuple(
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype)),
        ## tuple for b_data
        nlp.data.batchify.Tuple(
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
            nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype)))
    if bucket_sample:
        batch_sampler = nlp.data.sampler.FixedBucketSampler(
            joined_len,
            batch_size=batch_size,
            num_buckets=4,
            ratio=0.2,
            shuffle=True)
        loader_train = gluon.data.DataLoader(
            dataset=joined_data_train,
            num_workers=4,
            batch_sampler=batch_sampler,
            batchify_fn=batchify_fn)
    else:
        loader_train = gluon.data.DataLoader(
            dataset=joined_data_train,
            num_workers=4,
            shuffle=True, batch_size = batch_size,
            batchify_fn=batchify_fn)
    return loader_train, len(joined_data_train)


def preprocess_data_metriclearn_separate(trans1, trans2, class_labels, train_a_ds, train_b_ds, batch_size, shuffle=True):
    """Train/eval Data preparation function."""
    pool = multiprocessing.Pool()
    label_dtype = 'float32' # if not task.class_labels else 'int32'
    bow_count_dtype = 'float32'

    a_data_train = mx.gluon.data.SimpleDataset(pool.map(trans1, train_a_ds))
    b_data_train = mx.gluon.data.SimpleDataset(pool.map(trans2, train_b_ds))
    
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype))
    a_loader_train = gluon.data.DataLoader(
            dataset=a_data_train,
            num_workers=4,
        last_batch = 'rollover', ## need to ensure all batches are the same size here
        shuffle=shuffle,  # shuffle optional (for training)
        batch_size = batch_size,
        batchify_fn=batchify_fn)
    b_loader_train = gluon.data.DataLoader(
        dataset=b_data_train,
        num_workers=4,
        shuffle=False,  # don't shuffle fixed set 'B'
        batch_size = batch_size,
        batchify_fn=batchify_fn)
    return a_loader_train, len(a_data_train), b_loader_train


def get_dual_bert_datasets(class_labels,
                           vectorizer,
                           train_ds1,
                           train_ds2,
                           model_name,
                           dataset,
                           batch_size,
                           dev_bs,
                           max_len1,
                           max_len2,
                           pad,
                           use_bert_vocab=False,
                           shuffle=True,
                           ctx=mx.cpu()):
    bert, bert_vocabulary = get_model(
        name=model_name,
        dataset_name=dataset,
        pretrained=True,
        ctx=ctx,
        use_pooler=True,
        use_decoder=False,
        use_classifier=False)
    do_lower_case = 'uncased' in dataset    
    bert_tokenizer = BERTTokenizer(bert_vocabulary, lower=do_lower_case)

    # transformation for data train and dev
    trans1 = BERTDatasetTransform(bert_tokenizer, max_len1,
                                  class_labels=class_labels,
                                  label_alias=None,
                                  pad=pad, pair=False,
                                  has_label=True,
                                  vectorizer=vectorizer,
                                  bert_vocab_size=len(bert_vocabulary) if use_bert_vocab else 0)

    trans2 = BERTDatasetTransform(bert_tokenizer, max_len2,
                                  class_labels=class_labels,
                                  label_alias=None,
                                  pad=pad, pair=False,
                                  has_label=True,
                                  vectorizer=vectorizer,
                                  bert_vocab_size=len(bert_vocabulary) if use_bert_vocab else 0)
    
    #train_data, num_train_examples = preprocess_data_metriclearn(
    #   trans, class_labels, train_ds1, train_ds2, batch_size, max_len, pad)
    batch_size = len(train_ds2)
    a_train_data, num_train_examples, b_train_data = preprocess_data_metriclearn_separate(
        trans1, trans2, class_labels, train_ds1, train_ds2, batch_size, shuffle=shuffle)
    return a_train_data, num_train_examples, bert, b_train_data, bert_vocabulary
