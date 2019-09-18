.. _config-options-label:

Configuration/Hyperparameter Options
====================================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:

The following configuration/hyperparameter options are available in TMNT

===================  ===========    =================================================================
Option               Type           Description
===================  ===========    =================================================================
lr                   real           Learning rate
optimizer            categorical    MXNet optimizer (adam, sgd, etc.)
n_latent             integer        Number of latent topics
enc_hidden_dim       integer        Number of dimensions for encoding layer
batch_size           integer        Batch size to use during learning
embedding_size       integer        Number of embedding dimensions if pre-trained embedding not used
embedding_source     categorical    MXNet pre-trained embedding name (see embedding docs for details)
fixed_embedding      categorical    Either ``True`` or ``False``; fixes weights of embedding layer
latent_distribution  categorical    Either ``vmf``, ``gaussian`` or ``logistic_gaussian``
===================  ===========    =================================================================
