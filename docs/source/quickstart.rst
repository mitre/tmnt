Quickstart Guide
================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:

What is TMNT?
~~~~~~~~~~~~~

TMNT (Topic Modeling Neural Toolkit) implements a number of recently published methods for
training and applying neural network variational autoencoders (VAEs) to the problem of
modeling latent topics in large documents collections. While various open source implementations
of some of the algorithms in these recent papers are available, TMNT strives to include
the best variations across the literature within a software package that is quick to use and
deploy for applications.  Some novel algorithms are also implemented, including an approach
to leverage word embeddings directly as the input representation as well as methods for
guided topic modeling that allow users to influence the make-up of learned topics.

Training a Topic Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training a topic model requires both a training file containing sparse vector representations of documents
along with a test/validation file in the same format. In addition, a vocabulary file is needed to
map token indices back to their string representations.  See preparing data (below) for how to
process a corpus of text data into this sparse vector format.

Once the files are in place, training a model invovles invoking the ``train_model.py`` script
found in the ``bin/`` directory.  Using the example data provided (20 news corpus), we can build
a model as follows::

  python bin/train_model.py --tr_vec_file ./data/train.2.vec --tst_vec_file ./data/test.2.vec --vocab_file ./data/train.2.vocab --save_dir ./_experiments/ --model_dir ./_model_dir_final/ --config ./examples/train_model/model.config --trace_file ./TRACE.csv 



2. Preparing text data
++++++++++++++++++++++

The sparse vector representation for a corpus can be obtained from two different input formats:
1) plain text documents (one document per file) or 2) json objects with one document per json object.

The key arguments required for Raw Text mode are the directory of training files (``--train_dir``),
directory of test files (``--test_dir``) a file pattern (``--file_pat``) and (``--vocab_size``).
The data in the training and test directories should consiste of raw UTF-8 encoded text files where
each file represents a document or other natural unit of text for the target domain. The file pattern
should be provided to select files that match a specified regular expression.  Finally, the vocabulary
size ``--vocab_size`` will indicate the total number of word types that will be used in the model.

TMNT does it's own pre-processing of the text and includes a built-in stop-word list for English
to remove certain common terms that tend to act as distractors for the purposes of generating coherent topics.
This pre-processing is implemented in Python and relatively unoptimized.  Fortunately, the pre-processing need only
be done once, up front, in order to experiment with a wide variety of topic model variations. 
