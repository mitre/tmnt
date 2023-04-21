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

# %%
# Example TODO ...
