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

Command-line modes for Training Topic Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two primary methods for building topic models in TMNT: 1) Cached vector *mode* and
2) Raw text *mode*.   The latter assumes raw text files are provided as the input while the
former assumes a corpus of text files that have been pre-processed and converted into a sparse-vector
format for faster processing.

1. Cached Vector Mode
+++++++++++++++++++++

Cached Vector Mode requires the command arguments: ``--tr_vec_file``, ``tst_vec_file`` and ``--vocab_file``
which should refer to a training sparse vector file, test/validation sparse vector file and a
vocabulary file (with a single vocabularly item on each line). The sparse vector files follow the
libSVM format where each line represents a document.  The first element on each line should be
an integer coding for the *label* of the document, if one exists. If labels are not available, this value
should be -1. The remaining elements on each line should be space-separated and have the form ``<index>:<count>``
where both ``index`` and ``count`` are integers and denote the vocabulary id and its frequency/count
in the document, respectively.


2. Raw Text Mode
++++++++++++++++

The key arguments required for Raw Text mode are the directory of training files (``--train_dir``),
directory of test files (``--test_dir``) a file pattern (``--file_pat``) and (``--vocab_size``).
The data in the training and test directories should consiste of raw UTF-8 encoded text files where
each file represents a document or other natural unit of text for the target domain. The file pattern
should be provided to select files that match a specified regular expression.  Finally, the vocabulary
size ``--vocab_size`` will indicate the total number of word types that will be used in the model.

In Raw Text Mode, TMNT does it's own pre-processing of the text and includes a built-in stop-word list for English
to remove certain common terms that tend to act as distractors for the purposes of generating coherent topics.
This pre-processing is implemented in Python and relatively unoptimized.  Fortunately, the pre-processing need only
be done once, up front, in order to experiment with a wide variety of topic model variations. By including
the same command arguments for *Cached Vector Mode* together with the arguments for Raw Text Mode, TMNT
will save the cached training vector file, test vector file and vocabulary file.
