"""
Training a Transformer VED model with model selection
=====================================================
"""
from tmnt.estimator import SeqBowEstimator
import numpy as np
import os
import logging
from sklearn.datasets import fetch_20newsgroups
from tmnt.preprocess.vectorizer import TMNTVectorizer
from tmnt.inference import SeqVEDInferencer
from tmnt.distribution import LogisticGaussianDistribution
from tmnt.utils.log_utils import logging_config
from tmnt.data_loading import get_llm_dataloader
import torch
import optuna

data, y = fetch_20newsgroups(shuffle=True, random_state=1,
                              remove=('headers', 'footers', 'quotes'),
                              return_X_y=True)

# %%
# For the purposes of this example, only a very small training set is used
# along with small maximum sequence lengths to allow for the example to complete
# in short order without a GPU
# Larger sequence sizes and training sets should be used in practice
tr_size = 100
train_data = data[:tr_size]
dev_data   = data[-tr_size:]
train_y    = y[:tr_size]
dev_y      = y[-tr_size:]
batch_size = 20
seq_len = 64
pad = True

vectorizer = TMNTVectorizer(vocab_size=2000)
vectorizer.fit_transform(train_data)

supervised  = True
use_logging = True

# %%
# Classes is None if unsupervised, otherwise a list of possible label/string values
if supervised:
    num_classes = int(np.max(y) + 1)
    classes = ['class_'+str(i) for i in range(num_classes)]
else:
    num_classes = 0
    classes = None

if use_logging:    
    logging_config(folder='.', name='f_seqbow_20news', level='info', console_level='info')
    log_method = 'log'
else:
    log_method = 'print'

# %%
# 20newsgroups has integers for labels; but we map these to string values here
# to showcase the common use in practice where a string describes each label in a labeled dataset
train_y_s = ['class_'+str(y) for y in train_y]
dev_y_s = ['class_'+str(y) for y in dev_y]

# %%
# We'll use distilbert here as it's more compute efficient than BERT
tf_llm_name = 'distilbert-base-uncased'

# %%
# Setup the labeled training and development datasets
train_ds = list(zip(train_y_s, train_data))
dev_ds   = list(zip(dev_y_s, dev_data))
aux_ds   = None
label_map = { l:i for i,l in enumerate(classes) }

device = torch.device('cpu')
epochs = 12

# %%
# Get the training and development data dataloaders
train_loader = get_llm_dataloader(train_ds, vectorizer, tf_llm_name, label_map, batch_size, seq_len, device=device )
dev_loader = get_llm_dataloader(dev_ds, vectorizer, tf_llm_name, label_map, batch_size, seq_len, device=device)

# %%
# Function to provide the estimator object required for a single training run.
# All searchable hyperparameters are provided as arguments to this function.
def get_estimator(n_topics, alpha, lr, decoder_lr, entropy_loss_coef):
    latent_distribution = LogisticGaussianDistribution(768,n_topics,dr=0.05,alpha=alpha, device=device)
    estimator = SeqBowEstimator(llm_model_name = tf_llm_name,
                            latent_distribution = latent_distribution,
                            n_labels = num_classes,
                            vocabulary = vectorizer.get_vocab(),
                            batch_size=batch_size, device=device, log_interval=1,
                            log_method=log_method, gamma=1.0, entropy_loss_coef=entropy_loss_coef, 
                            lr=lr, decoder_lr=decoder_lr, epochs=epochs)
    return estimator


# %%
# Define a function that will execute a single trial. This includes:
# a single model fit with specific hyperparameter values), 
# and evaluation against the development data that returns the
# overall value/utility of the model (e.g. topic quality or accuracy or AuPRC or .. )
def train_topic_model(trial):
    n_topics = trial.suggest_int("n_topics", 10, 40, 5)
    lr       = trial.suggest_float("lr", 1e-4, 1e-2)
    decoder_lr = trial.suggest_float("decoder_lr", 0.001, 0.1)
    alpha    = trial.suggest_float("alpha", 0.5, 3.0)
    entropy_loss_coef = trial.suggest_float("entropy_loss_coef", 10.0, 10000.0, log=True)
    estimator = get_estimator(n_topics, alpha, lr, decoder_lr, entropy_loss_coef)
    objective, _ = estimator.fit_with_validation(train_loader, dev_loader, None)
    return objective

# %%
# Create the study and optimize over a given number of trials
n_trials = 16
study = optuna.create_study(direction='maximize')
study.optimize(train_topic_model, n_trials=n_trials)

# %%
# Now, get the best parameters returned from the search and refit the
# model with those parameters.  In practice, the final refit model may
# use a larger set of training data than the data used for the hyperparameter
# search.
best_lr = study.best_params['lr']
best_decoder_lr = study.best_params['decoder_lr']
best_n_topics = study.best_params['n_topics']
best_alpha    = study.best_params['alpha']
best_entropy_loss_coef = study.best_params['entropy_loss_coef']

final_estimator = get_estimator(best_n_topics, best_alpha, best_lr, best_decoder_lr, best_entropy_loss_coef)
final_estimator.fit_with_validation(train_loader, dev_loader, None)
