Scripts and Command-line Options
================================

This section provides an overview of the command-line options for the various TMNT scripts.

1. ``train_model.py``
+++++++++++++++++++++

The ``train_model.py`` script will train a single model using the configuration specified via the ``-config``
argument. This can be used to do model selection "by hand" or to retrain a model using a configuration
found via the ``select_model.py`` script. In general, given the number of hyper-parameters it is recommended
to use ``select_model.py`` to find the best topic model for a given dataset.

===================  ===========    =================================================================
Argument             Type           Description
===================  ===========    =================================================================
config               string/path    Configuration file (see :ref:`config-options-label`)
tr_vec_file          string/path    Vector file containing the training set
val_vec_file         stirng/path    Vector file containing the validation set
test_vec_file        string/path    Vector file containing the held out test set
vocab_file           string/path    Vocabulary file
save_dir             string/path    Directory for log files, model selection configs, and saved model parameters
model_dir            string/path    Override directory for model outputs (default location is a ``MODEL`` sub-directory within the argument to ``save_dir``
str_encoding         string         Character encoding to use for ``vocab_file``
hybridize            flag           When set will use the symbolic computation graph (via Gluon ``hybridize``); may train certain models faster
gpu                  integer        Logical id for the gpu (default is -1, use CPU instead)
num_final_evals      integer        Number of (final) evaluations on validation or heldout data with random initializations with (final) model configuration
eval_freq            integer        Number of training epochs in between computing perplexity and coherence on validation data
trace_file           string/path    Output file with perplexities and coherence scores computed every ``eval_freq`` epochs
topic_seed_file      string/path    JSON file that provides seed terms for topics (see :ref:`guided-label`)
===================  ===========    =================================================================

2. ``select_model.py``
++++++++++++++++++++++

The ``select_model.py`` script implements the full model selection. The hyper-parameter selection
is carried out over the validation data (provided by ``--val_vec_file``). It is recommended
to compute the final held out perplexity and coherence on a separate held out dataset using ``--tst_vec_file``.

The arguments for ``select_model.py`` overlap to a large degree with the ``train_model.py`` script.
Below are the additional arguments used by ``select_model.py``.


=====================  ===========    =================================================================
Argument               Type           Description
=====================  ===========    =================================================================
config_space           string/path    Path to the YAML file specifying the configuration space (:ref:`model-selection-label`)
iterations             integer        The number of hyperband iterations
coherence_coefficient  float          The weight for coherence in the model search objective function
searcher               string         Search algortihm used by scheduler (random, bayesopt, skopt)
scheduler              string         Scheduling algorithm (hyperband or fifo)
brackets               integer        Number of hyperband brackets (if Hyperband algorithm used)
cpus_per_task          integer        Number of CPUs to use per task (per model being trained/evaluated)
=====================  ===========    =================================================================


