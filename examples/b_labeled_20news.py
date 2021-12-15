"""
Joint topic + labeling model
============================

Another example with 20 news dataset. This involves
building a model using the labels as prediction targets.
"""

from tmnt.estimator import BowEstimator
import numpy as np
import gluonnlp as nlp
import os
from sklearn.datasets import fetch_20newsgroups
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.inference import BowVAEInferencer
from tmnt.distribution import HyperSphericalDistribution, LogisticGaussianDistribution
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# %%
# Get the 20news data using pre-partitioned training/test splits
train_data = fetch_20newsgroups(subset='train', shuffle=True, random_state=1)
test_data  = fetch_20newsgroups(subset='test', shuffle=True, random_state=1)


                 

# %%
# In setting up the vectorizer, this time we'll use a somewhat larger vocabulary
# size which may be more appropriate if we're interested in classifier labeling
# accuracy.
cv_params = dict(max_df=0.8)
tf_vectorizer = TMNTVectorizer(vocab_size=10000, count_vectorizer_kwargs=cv_params)
X_train, _ = tf_vectorizer.fit_transform(train_data.data)
X_test, _  = tf_vectorizer.transform(test_data.data)
y_train, y_test = train_data.target, test_data.target

# %%
# Note that when processing a JSON list
# formatted dataset, one typically computes the label map from whatever field denotes each documents label
# Here, in the case of the 20 Newsgroup dataset, the label map has already been calculated.
# We create the label map and explicitly attach to the vectorizer so the mapping between label indices
# and label names (strings) is preserved.
tf_vectorizer.label_map = dict([ (train_data.target_names[i], i) for i in range(len(train_data.target_names)) ])

# %%
# First, let's train a baseline comparison logistic regression
# model using sklearn's SGDClassifier using the document-term matrices.
sgd_params = dict(alpha=1e-5, penalty='l2', loss='log')
sgd_clf = SGDClassifier(**sgd_params)
sgd_clf.fit(X_train, y_train)
sgd_y_pred = sgd_clf.predict(X_test)
print("SGD accuracy score = {}".format(accuracy_score(y_test, sgd_y_pred)))

# %%
# For labeled/supervised topic models, the ``gamma`` (hyper-)parameter is crucial.
# It provides a weight that is applied to the classification loss to increase/decrease
# it's influence in the total loss.  In situations where classification is paramount,
# it's usually a good idea to set ``gamma`` to a high value such as 100, 1000 or even
# 1e7.  Extremely high values, however, will often result in a very poor quality
# topic model.

# %%
# For this example, let's set ``gamma = 100.0``. This should favor the classification
# loss (versus the elbo "topic loss")
gamma = 100.0

# %%
# Let's also use a larger latent space - size 40 - with a Hyperspherical distribution
latent_distribution = HyperSphericalDistribution(40)

num_label_values = int(np.max(y_train)) + 1

# %%
# Setting up the estimator now, we include some adjustments to default parameters
# to reduce potential for overfitting the classifier
l_estimator = BowEstimator(vocabulary = tf_vectorizer.get_vocab(),
                           latent_distribution = latent_distribution,
                           n_labels=num_label_values,
                           gamma=gamma,
                           log_method='print',
                           enc_dr=0.2,
                           classifier_dropout=0.1,
                           lr=0.0003, enc_hidden_dim=50,
                           epochs=12, batch_size=128)

# %%
# Fit the model
_ = l_estimator.fit(X_train, y_train)

# %%
# Perform validation/evaluation on the test split to assess performance
v_results = l_estimator.validate(X_test, y_test)
print("Validation results acc = {}, ppl = {}, npmi = {}"
      .format(v_results['accuracy'], v_results['ppl'], v_results['npmi']))

# %%
# Train a semi-supervised model using just the first 500 training points as
# labeled data and using the UNLABELED test data (transducive learning).  
l_estimator_semi_supervised = BowEstimator(vocabulary = tf_vectorizer.get_vocab(),
                                           latent_distribution = HyperSphericalDistribution(40),
                                           n_labels=num_label_values,
                                           gamma=10.0,
                                           log_method='print',
                                           enc_dr=0.2,
                                           classifier_dropout=0.1,
                                           lr=0.0003, enc_hidden_dim=50,
                                           epochs=20, batch_size=128)
_ = l_estimator_semi_supervised.fit_with_validation(X_train[:500], y_train[:500], None, None, X_test)
v_results_ss = l_estimator_semi_supervised.validate(X_test, y_test)
print("Validation results with semi-supervised learning. acc = {}, ppl = {}, npmi = {}"
      .format(v_results_ss['accuracy'], v_results_ss['ppl'], v_results_ss['npmi']))


# %%
# We can create an inference object used to encode new data points. For convenience,
# the inferencer keeps the vectorizer used to process text strings and convert to
# term frequency vectors.  This allows the inferencer to be used directly as
# a text classifier by using the ``predict_text`` method.
l_inferencer = BowVAEInferencer(l_estimator, pre_vectorizer=tf_vectorizer)
predicted_labels, encodings, posteriors = l_inferencer.predict_text(test_data.data[:2])


# %%
# Now, let's create UMAP embeddings from the model encodings when applied to the test data
# Note here the use of ``use_probs=True`` and a higher ``target_entropy`` of 2.0.
# The target entropy rescales the topic proportions to avoid extremely skewed distributions
# This can be helpful when plotting embeddings and/or interpeting topic probabilities/proportions.
# Note that since the model was trained in a supervised manner, the embeddings show 

embeddings = l_inferencer.get_umap_embeddings(X_test, use_probs=True, target_entropy=2.5)
l_inferencer.plot_to(embeddings, y_test, None)





