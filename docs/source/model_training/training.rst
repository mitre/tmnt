.. _training-label:

Training and Topic Model Configuration
======================================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:


Configuration Example
+++++++++++++++++++++

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


.. _config-options-label:

Configuration/Hyperparameter Options
++++++++++++++++++++++++++++++++++++

The following configuration/hyperparameter options are available in TMNT

===================  ===========    =================================================================
Option               Type           Description
===================  ===========    =================================================================
epochs               integer        Number of training epochs (should be fixed to a single value for hyperband)
lr                   real           Learning rate
batch_size           integer        Batch size to use during learning
latent_distribution  subconfig      Subconfigurations with ``dist_type:[vmf|gaussian|logistic_gaussian]`` with ``kappa`` for ``vmf`` and ``alpha`` for ``logistic_gaussian``
optimizer            categorical    MXNet optimizer (adam, sgd, etc.)
n_latent             integer        Number of latent topics
enc_hidden_dim       integer        Number of dimensions for encoding layer
num_enc_layers       integer        Number of encoder fully connected layers
enc_dr               real           Dropout used for encoder layers
coherence_loss_wt    real           Coefficient to weight coherence loss term
redundancy_loss_wt   real           Coefficient to weight redundancy loss term
embedding            subconfig      Subconfigurations with ``source`` categorical and optional ``size`` configuration for ``source: random``
===================  ===========    =================================================================


The following sub-configurations are used to define sub-spaces for ``latent_distribution`` and ``embedding`` configuration options


================  ===================    ==========     =======================================================
Sub-Option        Parent Option          Type           Description
================  ===================    ==========     =======================================================
source            embedding              categorical    Either ``random`` or a GluonNLP pre-trained embeddings
fixed             embedding              categorical    Either ``True`` or ``False``; fixes weights of embedding layer
kappa             latent_distribution    real           Concentration parameter when using ``vmf`` latent distribution, ``dist_type: vmf``
alpha             latent_distributionn   real           Prior variance when using ``logistic_gaussian`` latent distribution, ``dist_type: logistic_gaussian``
================  ===================    ==========     =======================================================

Some details on these options follows.

Pre-trained Word Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Word embeddings are used within TMNT to initialize the first fully connected
layer within the encoder (this is equivalent to averaging the word embeddings
for all in-vocabulary tokens). A pre-trained embedding can be used within
a configuration by simply including the GluonNLP registered name for the embedding
as the value of the ``embedding_source`` configuration option.  All embedding
names have the form ``source:name`` where source is the type of embedding. There
are four possible sources: ``glove``, ``fasttext``, ``word2vec`` and ``file``.
So, for example, ``glove:glove.42B.300d`` refers to Glove embeddings with 300 dimensions
trained on 42 billion tokens. Available Glove embeddings can be obtained via::

  >>> import gluonnlp as nlp
  >>> nlp.embedding.list_sources('GloVe')
  ['glove.42B.300d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d', 'glove.6B.50d', 'glove.840B.300d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', 'glove.twitter.27B.25d', 'glove.twitter.27B.50d']
  >>> nlp.embedding.list_sources() ## will list ALL sources (from glove, word2vec and fasttext "sources")

See ``https://gluon-nlp.mxnet.io/api/modules/embedding.html#gluonnlp.embedding.TokenEmbedding`` for other embeddings
available.  

It is also possible to use custom user-pretrained embeddings using the ``file`` source. These embeddings
should be in a compressed ``.npz`` file as generated using the ``train_embeddings.py`` script.

Latent Distributions
~~~~~~~~~~~~~~~~~~~~

TMNT provides three latent distributions, ``gaussian``, ``logisic_gaussian`` and ``vmf`` (von Mises Fisher).
After hundreds of experiments across many datasets, we have found that the ``vmf`` distribution generally works
best. Besides providing generally better coherence and perplexity, the ``vmf`` distribution allows much greater
flexibility to trade off coherence for perplexity or vice-versa. The ``logisic_gaussian`` distribution, however,
does tend to work as well or better than ``vmf`` with larger numbers of topics (e.g. over 80). The
``gaussian`` distribution is not recommended under most circumstances and is left here for comparison.

