"""
Model Selection
===============

Model selection using the API.
"""

from tmnt.estimator import BowEstimator
import os
from sklearn.datasets import fetch_20newsgroups
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.estimator import BowEstimator
from tmnt.distribution import LogisticGaussianDistribution, GaussianDistribution, VonMisesDistribution
from tmnt.inference import BowVAEInferencer
import optuna
from tmnt.utils.log_utils import logging_config
import pyLDAvis
import funcy

logging_config(folder='.', name='model_select_20news', level='info', console_level='info')


data, y = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'),
                             return_X_y=True)

tf_vectorizer = TMNTVectorizer(vocab_size=2000, count_vectorizer_kwargs=dict(max_df=0.8,
                                                                             stop_words='english',
                                                                             token_pattern=r'[A-Aa-z][A-Za-z][A-Za-z]+'))
X, _ = tf_vectorizer.fit_transform(data)


def get_estimator(enc_size, lr, emb_size, alpha):
    n_latent = 20
    epochs   = 64
    distribution = LogisticGaussianDistribution(enc_size, n_latent, dr=0.2, alpha=alpha)
    estimator = BowEstimator(vocabulary=tf_vectorizer.get_vocab(), latent_distribution=distribution,
                             log_method='log', lr=lr, batch_size=400, embedding_source='random', embedding_size=emb_size,
                             epochs=epochs, enc_hidden_dim=enc_size, validate_each_epoch=False, quiet=False)
    return estimator

def train_topic_model(trial):
    enc_size = trial.suggest_int("enc_size", 80, 120, 10)
    #n_latent = trial.suggest_int("n_latent", 10, 30, 5)
    lr       = trial.suggest_float("lr", 1e-3, 1e-2)
    emb_size = trial.suggest_int("emb_size", 180, 220, 10)
    alpha    = trial.suggest_float("alpha", 0.5, 3.0)
    estimator = get_estimator(enc_size, lr, emb_size, alpha)
    objective, _ = estimator.fit_with_validation(X, None, X, None)
    return objective

study = optuna.create_study(direction='maximize')
study.optimize(train_topic_model, n_trials=24)

best_lr = study.best_params['lr']
best_emb_size = study.best_params['emb_size']
best_enc_size = study.best_params['enc_size']
best_alpha    = study.best_params['alpha']

# get estimator with best parameters and refit
estimator = get_estimator(best_enc_size, best_lr, best_emb_size, best_alpha)
_,_ = estimator.fit_with_validation(X, None, X, None)


inferencer = BowVAEInferencer(estimator, pre_vectorizer=tf_vectorizer)

full_model_dict = inferencer.get_pyldavis_details(X)
pylda_opts = funcy.merge(full_model_dict, {'mds': 'mmds'})
vis_data = pyLDAvis.prepare(**pylda_opts)

pyLDAvis.save_html(vis_data, 'm1.html')
