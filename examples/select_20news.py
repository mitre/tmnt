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

from sklearn.datasets import fetch_20newsgroups, dump_svmlight_file
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.configuration import TMNTConfigBOW
from tmnt.trainer import BowVAETrainer, LabeledBowVAETrainer
from tmnt.selector import BaseSelector

data, y = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)

tf_vectorizer = TMNTVectorizer(vocab_size=2000)
X, _ = tf_vectorizer.fit_transform(data)
vocab = tf_vectorizer.get_vocab()

dump_svmlight_file(X[:8000], y[:8000], '_train.vec')
dump_svmlight_file(X[8000:], y[8000:], '_test.vec')

tmnt_config = TMNTConfigBOW('examples/select_model/config.yaml').get_configspace()
selector = BaseSelector(tmnt_config, 8, 'random', 'fifo', 1, 4, False, 1, 1234, '_model_out')

trainer = BowVAETrainer('_exps', '_model_out', vocab, None, '_train.vec', '_test.vec')
selector.select_model(trainer)

n_labels = int(np.max(y)) + 1

labeled_trainer = LabeledBowVAETrainer(n_labels, '_model_out', '_model_out', vocab, None, '_train.vec', '_test.vec')

l_selector = BaseSelector(tmnt_config, 8, 'random', 'fifo', 1, 4, False, 1, 1234, '_model_out')
l_selector.select_model(labeled_trainer)
