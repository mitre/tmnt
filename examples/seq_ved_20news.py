"""
Building a sequence VED topic model with the API
================================================

Simple example loading 20 news data, building a topic model
using a BERT pre-trained sequence to bag-of-words model.
"""

from tmnt.estimator import BowEstimator, MetaBowEstimator, SeqBowEstimator
import numpy as np
import gluonnlp as nlp
import os
from sklearn.datasets import fetch_20newsgroups
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.inference import SeqVEDInferencer
from tmnt.data_loading import prepare_bert

n_samples = 2000
n_features = 1000

data, _ = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)
data_samples = data[:n_samples]

max_seq_len = 128

data_train, X, vectorizer, bert_base, bert_vocab = prepare_bert(data_samples, max_seq_len, n_features)
estimator = SeqBowEstimator(bert_base, bert_vocab, vectorizer.get_vocab(), batch_size=8, epochs=1, log_method='print', max_batches=10)

val_samples = data[-100:]
data_val, X_val, _, _, _ = prepare_bert(val_samples, max_seq_len, vectorizer=vectorizer)

estimator.fit_with_validation((data_train, X), None, (data_val, X_val), None)

inferencer = SeqVEDInferencer(estimator.model, bert_vocab, max_seq_len)
encodings, tokens = inferencer.encode_text('The windows operating system was installed on the hard drive')


