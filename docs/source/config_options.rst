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
num_enc_layers       integer        Number of encoder fully connected layers
enc_dr               real           Dropout used for encoder layers
coherence_loss_wt    real           Coefficient to weight coherence loss term
redundancy_loss_wt   real           Coefficient to weight redundancy loss term
batch_size           integer        Batch size to use during learning
embedding_size       integer        Number of embedding dimensions if pre-trained embedding not used
embedding_source     categorical    MXNet pre-trained embedding registered name or file-path
fixed_embedding      categorical    Either ``True`` or ``False``; fixes weights of embedding layer
latent_distribution  categorical    Either ``vmf``, ``gaussian`` or ``logistic_gaussian``
kappa                real           Concentration parameter when using ``vmf`` latent distribution
alpha                real           Prior variance when using ``logistic_gaussian`` latent distribution
===================  ===========    =================================================================

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
does tend to work as well or better than ``vmf`` with larger numbers of topics (e.g. over 100). The
``gaussian`` distribution is not recommended under most circumstances and is left here for comparison,
as the Gaussian distribution is the most common/natural variational distribution.
