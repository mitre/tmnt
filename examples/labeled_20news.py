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
from sklearn.datasets import fetch_20newsgroups
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.inference import BowVAEInferencer


n_samples = 2000
n_features = 1000

data, y = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)
data_samples = data[:n_samples]
tf_vectorizer = TMNTVectorizer(vocab_size=1000)
X, _ = tf_vectorizer.fit_transform(data_samples)


num_label_values = int(np.max(y)) + 1 # get the number of possible labels
l_estimator = LabeledBowEstimator(tf_vectorizer.get_vocab(), num_label_values)
_ = l_estimator.fit(X, y) # fit a covariate model using y
l_inferencer = BowVAEInferencer(l_estimator.model)

## the following returns a list of top 5 words per topic per covariate/label
t_terms = l_inferencer.get_top_k_words_per_topic(5)

