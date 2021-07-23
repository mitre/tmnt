"""
Training a First Topic model
============================

This example 
"""

# %%
# Start with various imports

from tmnt.preprocess.vectorizer import TMNTVectorizer

# %%
# Let's start by first fetching the 20 newsgroups dataset
from sklearn.datasets import fetch_20newsgroups
data, y = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)

# %%
# Next step involves creating a vectorizer that maps text in the list of strings, ``data``,
# to a document-term matrix, ``X``
tf_vectorizer = TMNTVectorizer(vocab_size=2000)
X, _ = tf_vectorizer.fit_transform(data)

# %%
# Fitting a model involves creating an instance of the :py:class:`tmnt.estimator.BowEstimator` class
from tmnt.estimator import BowEstimator
estimator = BowEstimator(tf_vectorizer.get_vocab(), epochs=8).fit(X)

# %%
# An inference object is then created which enables the application of the model to raw text
# data and/or directly to document-term matrices
from tmnt.inference import BowVAEInferencer
inferencer = BowVAEInferencer(estimator, pre_vectorizer=tf_vectorizer)
encodings = inferencer.encode_texts(['Greater Armenia would stretch from Karabakh, to the Black Sea, to the Mediterranean, so if you use the term Greater Armenia use it with care.','I have two pairs of headphones I\'d like to sell.  These are excellent, and both in great condition'])

# %%
# The model can be saved to disk and reloaded for model deployment
inferencer.save(model_dir='_model_dir')
reloaded_inferencer = BowVAEInferencer.from_saved(model_dir='_model_dir')

# %%
# Now let's visualize the encodings using UMAP:

import numpy as np
import umap
import matplotlib.pyplot as plt

enc_list = reloaded_inferencer.encode_data(X)
encodings = np.array(enc_list)

umap_model = umap.UMAP(n_neighbors=4, min_dist=0.5, metric='euclidean')
embeddings = umap_model.fit_transform(encodings)

# %%
# We can plot the UMAP embeddings as a scatter plot.
# For this dataset, although we did not use the provided labels ``y`` to help fit the
# topic model, it can be helpful to use the labels to color code the documents
# in order to see how documents with the same label are encoded.
plt.scatter(*embeddings.T, c=y, s=0.8, alpha=0.9, cmap='coolwarm')
plt.show()
