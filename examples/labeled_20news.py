"""
Joint topic + labeling model
============================

Another example with 20 news dataset. This involves
building a model using the labels as prediction targets.
"""

from tmnt.estimator import BowEstimator, LabeledBowEstimator
import numpy as np
import gluonnlp as nlp
import os
import umap
from sklearn.datasets import fetch_20newsgroups
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.inference import BowVAEInferencer


data, y = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)
tf_vectorizer = TMNTVectorizer(vocab_size=1000)
X, _ = tf_vectorizer.fit_transform(data)

num_label_values = int(np.max(y)) + 1 # get the number of possible labels

## random masked labels (semi-supervised!)
rv = np.random.normal(0,1,y.shape)
yp = y.copy()
yp[rv < 0] = -1

def estimate_with_gamma(gamma, f=None):
    l_estimator = LabeledBowEstimator(tf_vectorizer.get_vocab(), num_label_values, gamma)
    _ = l_estimator.fit(X, y) # fit a covariate model using y
    ## get results from validation
    l_estimator.validate(X, y)
    l_inferencer = BowVAEInferencer(l_estimator.model)
    embeddings = l_inferencer.get_umap_embeddings(X)
    l_inferencer.plot_to(embeddings, y, f)
    return embeddings

