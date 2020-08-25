.. _training-label:

Training a Topic Model
======================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:

In order to train a topic model, a *configuration* must be provided that indicates
the values of a number of model hyper-parameters. The various hyperparameters that can
be specified in a configuration can be found here: :ref:`config-options-label`.

These configurations can be directly specified
by the user or learned through model selection using the ``select_model.py`` script;
see :ref:`model-selection-label`.

Let's start with the example configuration provided in ``examples/train_model/model.config``.
The configuration is a JSON object::

  {"enc_hidden_dim": 250,
  "latent_distribution": {"dist_type": "vmf", "kappa": 32.0},
  "lr": 0.005,
  "n_latent": 20,
  "optimizer": "adam",
  "embedding": {"source": "random", "size":200},
  "epochs": 24,
  "batch_size": 1000,
  "num_enc_layers": 2,
  "enc_dr": 0.0,
  "target_sparsity":0.0,
  "coherence_loss_wt":0.0,
  "redundancy_loss_wt":0.0,
  "covar_net_layers": 1} 


These are common configuration options used in most topic models. 
