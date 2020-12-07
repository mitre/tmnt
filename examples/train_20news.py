"""
Building a topic model with the API
===================================

Simple example loading 20 news data, building a topic model
and encoding strings. 
"""

from tmnt.estimator import BowEstimator, MetaBowEstimator
import numpy as np
import gluonnlp as nlp
import os
import umap

from sklearn.datasets import fetch_20newsgroups
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.inference import BowVAEInferencer


data, _ = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)

tf_vectorizer = TMNTVectorizer(vocab_size=2000)
X, _ = tf_vectorizer.fit_transform(data)

estimator = BowEstimator(tf_vectorizer.get_vocab()).fit(X)

inferencer = BowVAEInferencer(estimator.model)
encodings = inferencer.encode_texts(['Greater Armenia would stretch from Karabakh, to the Black Sea, to the Mediterranean, so if you use the term Greater Armenia use it with care.','I have two pairs of headphones I\'d like to sell.  These are excellent, and both in great condition'])

## write out model
os.mkdir('_model_dir')
estimator.write_model('_model_dir') 

## reload model
est2 = BowEstimator.from_config('_model_dir/model.config', '_model_dir/vocab.json', pretrained_param_file='_model_dir/model.params')

## instead of fitting with data; initialize with pretrained values
est2.initialize_with_pretrained()
est2.perplexity(X) # get preplexity
est2.validate(X, None) # get perplexity, NPMI and redundancy

## visualize encodings (on training data)

encs = inferencer.encode_data(X, None)
encodings = np.array([enc.asnumpy() for enc in encs])

## UMAP model
umap_model = umap.UMAP(n_neighbors=4, min_dist=0.5, metric='euclidean')
embeddings = umap_model.fit_transform(encodings)

import matplotlib.pyplot as plt
plt.scatter(*embeddings.T, c=y, s=0.8, alpha=0.9, cmap='coolwarm')
plt.show()
