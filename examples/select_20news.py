"""
Model Selection
===============

Model selection using the API.
"""

from tmnt.estimator import BowEstimator
import numpy as np
import gluonnlp as nlp
import os
import umap

from sklearn.datasets import fetch_20newsgroups
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.configuration import TMNTConfigBOW
from tmnt.trainer import BowVAETrainer
from tmnt.selector import BaseSelector

data, y = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)

tf_vectorizer = TMNTVectorizer(vocab_size=2000)
X, _ = tf_vectorizer.fit_transform(data)
vocab = tf_vectorizer.get_vocab()

tmnt_config = TMNTConfigBOW('examples/select_model/config.yaml').get_configspace()
selector = BaseSelector(tmnt_config, 8, 'random', 'fifo', 1, 4, False, 1, 1234, '_model_out')

trainer = BowVAETrainer(vocab, X[:8000], X[8000:], log_out_dir='_exps', model_out_dir='_model_out')
selector.select_model(trainer)

n_labels = int(np.max(y)) + 1

labeled_trainer = BowVAETrainer(vocab, (X[:8000],y[:8000]), (X[8000:],y[8000:]), n_labels=n_labels,
                                log_out_dir='_exps', model_out_dir='_model_out')

l_selector = BaseSelector(tmnt_config, 8, 'random', 'fifo', 1, 4, False, 1, 1234, '_model_out')
l_selector.select_model(labeled_trainer)
