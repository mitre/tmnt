Examples
=================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:

Building a model from raw text files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assume a working directory ``$WORK`` containing two sub-directories ``$WORK/train`` and
``$WORK/test`` each containing a number of English raw text files, each with the
extension ``.txt``.  The following command will pre-process the files in those directories
and build a topic model with 20 latent topics and place the resulting model and
associated files in the directory ``WORK/model_1``::

  python3 train_bow_vae.py --train_dir $WORK/train --file_pat '*.txt' \
                           --test_dir $WORK/test \
			   --vocab_size 2000 \
                           --epochs 60 \
                           --n_latent 20 --batch_size 200 \
                           --lr 0.001 --latent_distribution vmf \
                           --model_dir $WORK/model_1


Caching to sparse vector files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			   
To then also create sparse vector files for faster subsequent processing, the command-line arguments
``--tr_vec_file``, ``--tst_vec_file`` and ``--vocab_file`` should be added. Those
corresponding files will be *created* if they do not already exist. ::


  python3 train_bow_vae.py --train_dir $WORK/train --file_pat '*.txt' \
                           --test_dir $WORK/test \
			   --vocab_size 2000 \
                           --epochs 60 \
                           --n_latent 20 --batch_size 200 \
                           --lr 0.001 --latent_distribution vmf \
                           --model_dir $WORK/model_1 \
			   --tr_vec_file $WORK/train.vec \
			   --tst_vec_file $WORK/test.vec \
			   --vocab_file $WORK/train.vocab


Training from sparse vector files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assuming the files ``$WORK/train.vec``, ``$WORK/test.vec`` and ``$WORK/train.vocab`` have been
created as above, running the same command *with or without* the ``--train_dir``, ``--test_dir``
and ``--vocab_size`` flags will then train directly from the vector files and built vocabularly.
In other words, if the ``--tr_vec_file``, ``--tst_vec_file`` and ``--vocab_file`` options
are provided and the corresponding files already exist, they will be used and the ``--train_dir``,
``--test_dir`` and ``--vocab_size`` arguments will be ignored.  If the vector and vocab files
do not exist, the raw text and vocabulary size arguments will be used and the vector files
will be created/cached after pre-processing.

In all of these cases, the resulting training and test corpus will be used to train and
evaluate topic models as per the other arguments provided, including....
