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
from tmnt.data_loading import to_label_matrix
from typing import Dict
from gluonnlp.data import BERTSentenceTransform

class JsonlDataset(SimpleDataset):
    """A dataset wrapping over a jsonlines (.jsonl) file, each line is a json object.

    Parameters:
        filename : Path to the .jsonl file.
        txt_key: Json attribute key to use for seleting text document strings
        label_key: Json attribute key to use to get string labels
        encoding : File encoding format. (default 'utf8')
        label_remap : Dictionary to map labels.
    """
    def __init__(self, filename: str, txt_key: str, label_key: str,
                 encoding: str = 'utf8', label_remap: Dict[str,str] = None, random_drop_pct: float = 0.0):

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

    Parameters:
        *args : one or more dataset-like objects. The data arrays.
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
    num_classes: int
        Must be provided if class_labels isn't provided
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
                 bert_vocab_size=0,
                 num_classes=None):
        self.class_labels = class_labels
        self.has_label = has_label
        self.use_bert_bow = bert_vocab_size > 0
        self.bert_vocab_size = bert_vocab_size
        self._label_dtype = 'int32' if class_labels else 'float32'
        self.num_classes = len(class_labels) if class_labels else num_classes
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
        np.array: classification task: label id in 'int32', shape (num_classes,),
            regression task: label in 'float32', shape (1,)
        """
        if self.has_label:
            input_ids, valid_length, segment_ids = self._bert_xform(line[:-1])
            label_str = line[-1]
            # map to int if class labels are available
            if self.class_labels:
                if label_str:
                    labels = [ self._label_map.get(label,0) for label in label_str.split(',') ]
                    if labels is None or len(labels) == 0:
                        labels = [0]
                else:
                    labels = [0]
            else:
                try:
                    labels=[int(label_str)]
                except:
                    labels=[0]
            #label = np.array(labels, dtype=self._label_dtype)
            if self.num_classes > 1:
                label_mat, _ = to_label_matrix([labels], num_labels=self.num_classes)
            else:
                label_mat = np.array([[0.0]]) # just fill with zeros; assumption is that labels will be ignored
            bow = None
            if self.use_bert_bow:
                bow = np.zeros(self.bert_vocab_size)
                inds, cnts = np.unique(input_ids, return_counts=True)
                bow[inds] = cnts
                bow = mx.nd.array(np.expand_dims(bow, 0), dtype='float32')
            elif self.vectorizer:
                bow,_ = self.vectorizer.transform(line[:-1])
                bow = mx.nd.array(bow, dtype='float32')
            return input_ids, valid_length, segment_ids, bow, label_mat[0]
        else:
            return self._bert_xform(line)



def preprocess_seq_data(trans, class_labels, dataset, batch_size, max_len, train_mode=True, pad=False, aux_dataset=None):
    pool = multiprocessing.Pool()
    # transformation for data train and dev
    label_dtype = 'float32' # if not task.class_labels else 'int32'
    bow_count_dtype = 'float32'
    # data train
    data_ds = mx.gluon.data.SimpleDataset(pool.map(trans, dataset))
    if aux_dataset is None:
        final_ds = data_ds.transform( lambda a,b,c,d,e: ((a,b,c,d,e),) ) # create a singleton tuple to keep data iterators simple
        data_ds_len = data_ds.transform(
            lambda input_id, length, segment_id, bow, label_id: length, lazy=False)
        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Tuple(
                nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
                nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype)))
    else:
        aux_ds = mx.gluon.data.SimpleDataset(pool.map(trans, aux_dataset))
        final_ds = UnevenArrayDataset(data_ds, aux_ds)
        logging.info("Uneven dataset created, size = {} (from data_ds = {}, aux_ds = {})".format(len(final_ds), len(data_ds), len(aux_ds)))
        data_ds_len = final_ds.transform( lambda a, b: a[1] + b[1], lazy=False )
        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Tuple(
                nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
                nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype)),
            nlp.data.batchify.Tuple(
                nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
                nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype)))
    if train_mode:
        # bucket sampler 
        num_buckets = min(6, len(data_ds) // batch_size)
        batch_sampler = nlp.data.sampler.FixedBucketSampler(
            data_ds_len,
            batch_size=batch_size,
            num_buckets=num_buckets,
            ratio=0.2, # may avoid batches with size = 1 (which may tigger a bug)
            shuffle=True)
        # data loader for training
        loader = gluon.data.DataLoader(
            dataset=final_ds,
            num_workers=4,
            batch_sampler=batch_sampler,
            batchify_fn=batchify_fn)
    else:
        loader = gluon.data.DataLoader(
            dataset=final_ds,
            batch_size=batch_size,
            num_workers=4,
            shuffle=False,
            batchify_fn=batchify_fn)
    return loader, len(final_ds)


def get_bert_datasets(class_labels,
                      vectorizer,
                      train_ds,
                      dev_ds,
                      batch_size,
                      max_len,
                      aux_ds = None,
                      bert_model_name = 'bert_12_768_12',
                      bert_dataset = 'book_corpus_wiki_en_uncased',
                      pad=False,
                      use_bert_vocab=False,
                      label_alias=None,
                      num_classes = None,
                      ctx=mx.cpu()):
    if class_labels is None and num_classes is None:
        raise Exception("Must provide class_labels or num_classes")
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
                                 bert_vocab_size = len(bert_vocabulary) if use_bert_vocab else 0,
                                 num_classes = num_classes)
    train_data, num_train_examples = preprocess_seq_data(trans, class_labels, train_ds, batch_size, max_len, train_mode=True, pad=pad,
                                                         aux_dataset=aux_ds)
    dev_data, _ = preprocess_seq_data(trans, class_labels, dev_ds, batch_size, max_len, train_mode=False, pad=pad)
    return train_data, dev_data, num_train_examples, bert, bert_vocabulary


############
# Handle dataloading for Smoothed Deep Metric Loss with parallel batching
############

def preprocess_data_metriclearn(trans, class_labels, train_a_ds, train_b_ds, batch_size, max_len, pad=False, bucket_sample=False, aux_dataset=None):
    """Train/eval Data preparation function."""
    pool = multiprocessing.Pool()
    label_dtype = 'float32' # if not task.class_labels else 'int32'
    bow_count_dtype = 'float32'

    a_data_train = mx.gluon.data.SimpleDataset(pool.map(trans, train_a_ds))
    b_data_train = mx.gluon.data.SimpleDataset(pool.map(trans, train_b_ds))

    # magic that "zips" these two datasets and pairs batches
    joined_data_train = UnevenArrayDataset(a_data_train, b_data_train)
    joined_len = joined_data_train.transform( lambda a, b: a[1] + b[1], lazy=False ) ## a[1] and b[1] and lengths, bucket by sum

    if aux_dataset is None:
        final_ds = joined_data_train.transform( lambda a,b: ((a,b),) ) # singleton tuple
        final_len = joined_len
        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Tuple(
                ## tuple for a_data: (ids, lengths, segments, bow vector, label)
                nlp.data.batchify.Tuple(
                    nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
                    nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype)),
                ## tuple for b_data
                nlp.data.batchify.Tuple(
                    nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
                    nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype))))
    else:
        aux_ds = mx.gluon.data.SimpleDataset(pool.map(trans, aux_dataset))
        final_ds = UnevenArrayDataset(joined_data_train, aux_ds)
        logging.info("Uneven dataset created, size = {} (from data_ds = {}, aux_ds = {})".format(len(final_ds), len(joined_data_train), len(aux_ds)))
        final_len = final_ds.transform( lambda a, b: a[0][1] + a[1][1] + b[1], lazy=False )
        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Tuple(
                ## tuple for a_data: (ids, lengths, segments, bow vector, label)
                nlp.data.batchify.Tuple(
                    nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
                    nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype)),
                ## tuple for b_data
                nlp.data.batchify.Tuple(
                    nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
                    nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype))),
            # tuple for auxilliary data
            nlp.data.batchify.Tuple(
                    nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
                    nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype)))
    if bucket_sample:
        batch_sampler = nlp.data.sampler.FixedBucketSampler(
            final_len,
            batch_size=batch_size,
            num_buckets=4,
            ratio=0.2,
            shuffle=True)
        loader = gluon.data.DataLoader(
            dataset=final_ds,
            num_workers=4,
            batch_sampler=batch_sampler,
            batchify_fn=batchify_fn)
    else:
        loader = gluon.data.DataLoader(
            dataset=final_ds,
            num_workers=4,
            shuffle=False, batch_size = batch_size,
            batchify_fn=batchify_fn)
    return loader, len(final_ds)


def preprocess_data_metriclearn_separate(trans1, trans2, class_labels, train_a_ds, train_b_ds, batch_size, shuffle=True, aux_dataset=None):
    """Train/eval Data preparation function."""
    pool = multiprocessing.Pool()
    label_dtype = 'float32' # if not task.class_labels else 'int32'
    bow_count_dtype = 'float32'

    a_data_train = mx.gluon.data.SimpleDataset(pool.map(trans1, train_a_ds))
    b_data_train = mx.gluon.data.SimpleDataset(pool.map(trans2, train_b_ds))

    if aux_dataset is None:
        a_final_ds = a_data_train.transform( lambda a,b,c,d,e: ((a,b,c,d,e),) ) # create a singleton tuple to keep data iterators simple
        a_batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Tuple(
                nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
                nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype)))
    else:
        aux_ds = mx.gluon.data.SimpleDataset(pool.map(trans2, aux_dataset))
        a_final_ds = UnevenArrayDataset(a_data_train, aux_ds)
        a_batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Tuple(
                nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
                nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype)),
            nlp.data.batchify.Tuple(
                nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
                nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype)))
    
    b_batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(),
        nlp.data.batchify.Pad(axis=0), nlp.data.batchify.Stack(bow_count_dtype), nlp.data.batchify.Stack(label_dtype))
    a_loader_train = gluon.data.DataLoader(
        dataset=a_final_ds,
        num_workers=4,
        last_batch = 'rollover', ## need to ensure all batches are the same size here
        shuffle=shuffle,  # shuffle optional (for training)
        batch_size  = batch_size,
        batchify_fn = a_batchify_fn)
    b_loader_train = gluon.data.DataLoader(
        dataset=b_data_train,
        num_workers=4,
        shuffle=False,  # don't shuffle fixed set 'B'
        batch_size  = batch_size,
        batchify_fn = b_batchify_fn)
    return a_loader_train, len(a_final_ds), b_loader_train


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
                           aux_dataset = None,
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
        trans1, trans2, class_labels, train_ds1, train_ds2, batch_size, shuffle=shuffle, aux_dataset=aux_dataset)
    return a_train_data, num_train_examples, bert, b_train_data, bert_vocabulary
