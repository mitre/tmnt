"""
Training a Deep Averaging Topic Model
=====================================

Topic model on 20 news using a Deep Averaging Network
as the encoder.
"""

from tmnt.estimator import DeepAveragingBowEstimator
import numpy as np
import gluonnlp as nlp
import os
import umap

from sklearn.datasets import fetch_20newsgroups
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.data_loading import prepare_dataset_sequence

data, y = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)

data_train = data[:2000]
max_len = 256

tr_dset, vocab, X = prepare_dataset_sequence(data_train, max_len, bow_vocab_size=2000)

#estimator = DeepAveragingBowEstimator(vocab, 0, 0.0, 300, 0.1, max_len)
estimator = DeepAveragingBowEstimator(vocab, 0, 0.0, 300, 0.1, max_len, epochs=20, log_method='print')
#estimator = DeepAveragingBowEstimator(vocab, 20, 10.0, 300, 0.0, max_len, epochs=20, log_method='print')
_ = estimator.fit_with_validation((tr_dset, X), None, (tr_dset, X), None)
