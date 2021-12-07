"""
Model Selection with Labels
===========================

Model selection with labeled data.
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
n_labels = int(np.max(y)) + 1
l_selector = BaseSelector(tmnt_config, iterations=4, searcher='random',
                          scheduler='hyperband', cpus_per_task=4, log_dir='_model_out')
labeled_trainer = BowVAETrainer(vocab, (X[:8000],y[:8000]), (X[8000:],y[8000:]), n_labels=n_labels,
                                log_out_dir='_exps', model_out_dir='_model_out')
l_selector.select_model(labeled_trainer)
