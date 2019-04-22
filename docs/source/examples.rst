Model Configurations
====================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:

In order to train a topic model, a *configuration* must be provided that indicates
a number of model (hyper)parameters. These configurations can be set by the user
or learned through model selection using the ``select_model.py`` script.

Let's start with the example configuration provided in ``examples/train_model/model.config``.
The configuration is a JSON object::

  {"enc_hidden_dim": 150,
  "latent_distribution": "vmf",
  "lr": 0.005,
  "n_latent": 20,
  "optimizer": "adam",
  "kappa": 64.0,
  "embedding_size": 300,
  "training_epochs": 40}

These are common configuration options used in most topic models (see below for additional options).
