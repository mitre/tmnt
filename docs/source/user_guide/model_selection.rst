.. _model-selection-label:

Model Selection using HPBandSter
================================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:


Quick Start
~~~~~~~~~~~

With a given configuration space (see details below), model selection is performed via the ``select_model.py`` script.
Below is an example::

  python bin/select_model.py --tr_vec_file ./data/train.vec \
  --val_vec_file ./data/test.vec --vocab_file ./data/train.vocab \
  --save_dir ./_experiments/ --model_dir ./_model_dir/ \
  --config_space ./examples/select_model/config.yaml --budget 27 \
  --iterations 60 --coherence_coefficient 4.0

Configuration Space
~~~~~~~~~~~~~~~~~~~

Model selection is made easy through the use of a simple YAML configuration
file that specifies the hyperparameter space.  See the example here::

  ## Learnable hyper-parameters 
  ---
    lr: {range: [1e-6, 1e-2]}
    latent_distribution: [vmf]
    optimizer: [adam]
    n_latent: {i_range: [20, 40], step: 5}
    enc_hidden_dim: {i_range: [50, 400], step: 50}
    kappa: {range: [1.0, 100.0]}   ## conditional on latent_distribution==vmf
    embedding_source: [glove:glove.42B.300d, glove:glove.6B.200d, fasttext:wiki.en, random]
    fixed_embedding: [True, False]
    embedding_size: {i_range: [50, 400], step: 50}

There are three different types of values for each configuration hyperparameter:
categorical, bounded real-valued and bounded integer-valued. 

Categorical-valued hyperparameters are indicated by including a simple list
of strings as the value for each hyperparameter key.

Bounded integer-valued hyperparameters are specified using an object that includes
the keys ``i_range`` and ``step``. ``i_range`` should have a value consisting of a list
of two integers, the first the lower bound and the second the upper bound. The ``step``
key should take an integer as a value to provide the step-size. If the ``step`` key-value
pair is not provided, the default step of 1 is used. Thus::

  {i_range: [20, 50], step: 5}

Indicates that the hyperparameter ranges from 20 to 50 over increments of size 5, so:
20, 25, 30, 35, 40, 45, 50

Bounded real-valued hyperparameters are specified by simply including an object with
a ``range`` key with a list value having two elements. The first element as the lower
bound and the second as the upper bound.  Thus::

  {range: [1.0, 5.0]}

Provides for hyperparameter values ranging from 1.0 to 5.0.  When hyperparameter
values are selected entirely randomly, they will be drawn uniformly from the range
specified.

