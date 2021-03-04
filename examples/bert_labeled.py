from tmnt.estimator import FullyLabeledSeqEstimator
import numpy as np
import gluonnlp as nlp
import os
import mxnet as mx
from sklearn.datasets import fetch_20newsgroups
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.inference import SeqVEDInferencer
from tmnt.bert_handling import get_bert_datasets
from mxnet.gluon.data import ArrayDataset

data, y = fetch_20newsgroups(shuffle=True, random_state=1,
                              remove=('headers', 'footers', 'quotes'),
                              return_X_y=True)
train_data = data[:2000]
dev_data   = data[-2000:]
train_y    = y[:2000]
dev_y      = y[-2000:]
model_name = 'bert_12_768_12'
dataset = 'book_corpus_wiki_en_uncased'
batch_size = 32
seq_len = 64
pad = True
tr_ds = ArrayDataset(train_data, train_y)
dev_ds = ArrayDataset(dev_data, dev_y)

vectorizer = TMNTVectorizer(vocab_size=2000)
vectorizer.fit_transform(train_data)

tr_dataset, dev_dataset, num_examples, bert_base = get_bert_datasets(None, vectorizer,
                                                                     tr_ds, dev_ds, model_name,
                                                                     dataset, batch_size, 8, seq_len, pad, mx.cpu())

num_classes = np.max(y) + 1

estimator = SeqBowEstimator(bert_base, n_labels = num_classes,
                                     optimizer='bertadam',
                                     batch_size=batch_size, ctx=mx.cpu(), log_interval=1,
                                     log_method='print', mix_val=1.0, n_latent=20,
                                     lr=2e-5, decoder_lr=0.001)

estimator.fit_with_validation(tr_dataset, dev_dataset, num_examples)

