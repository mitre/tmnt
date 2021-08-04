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

config_space = 'data/configs/select_model/config.yaml'
if not os.path.exists(config_space):
    print("Run this example from the top-level TMNT directory")
    exit(0)

tmnt_config = TMNTConfigBOW(config_space).get_configspace()
selector = BaseSelector(tmnt_config, iterations=8, searcher='random',
                        scheduler='hyperband', cpus_per_task=4, log_dir='_model_out')

trainer = BowVAETrainer(vocab, X[:8000], X[8000:], log_out_dir='_exps', model_out_dir='_model_out')
selector.select_model(trainer)
