Quickstart Guide
================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:



1. Training a Topic Model
+++++++++++++++++++++++++

Training a topic model requires both a training file containing sparse vector representations of documents
along with a test/validation file in the same format. In addition, a vocabulary file is needed to
map token indices back to their string representations.  See preparing data (below) for how to
process a corpus of text data into this sparse vector format.

Once the prepared files are in place, training a model invovles invoking the ``train_model.py`` script
found in the ``bin/`` directory.  Using the example data provided (20 news corpus), we can build
a model as follows::

  mkdir _experiments
  mkdir _model_dir

  python bin/train_model.py --tr_vec_file ./data/train.2.vec \
  --tst_vec_file ./data/test.2.vec --vocab_file ./data/train.2.vocab \
  --save_dir ./_experiments/ --model_dir ./_model_dir/ \
  --config ./examples/train_model/model.config --trace_file ./TRACE.csv

In general, TMNT assumes a test/validation corpus is available to determine the held out perplexity
and coherence. If a separate held-out dataset is unavailable or not desired (testing on the training
data is less of an issue for unsupervised algorithms), the ``--test_vec_file`` vector file can
take the same file as for trainin (via the ``--tr_vec_file`` option).


2. Preparing text data
++++++++++++++++++++++

The sparse vector representation for a corpus can be obtained from two different input formats:
1) json objects with one document per json object or 2) plain text documents (one document per file) 

There are two input formats (currently).  The first assumes a single JSON object per line in a file.  The value of the key ``text`` will
be used as the document string.  All other fields are ignored. So, for example::


  {"id": "1052322266514673664", "text": "This is the text of one of the documents in the corpus."}
  {"id": "1052322266514673665", "text": "This is the text of another of the documents in the corpus."}
  ...

Two directories of such files should be provided, one for training and one for test.  Assuming the files end with ``.json`` extensions, the
following example invocation would prepare the data for the training and test sets, creating vector representations with a vocabulary
size of 2000.  Note that this script uses the built in pre-processing which tokenizes, downcases and removes common English stopwords.
An example invocation::


  python bin/prepare_corpus.py --vocab_size 2000 --file_pat '*.json' --tr_input_dir ./train-json-files/ --tst_input_dir ./test-json-files/ --tr_vec_file ./train.2k.vec --vocab_file ./2k.vocab  --tst_vec_file ./test.2k.vec 


Another input format assumes directories for training and test sets, where each file is a separate plain text document. This should be
invoked by adding the ``--txt_mode`` option::


  python bin/prepare_corpus.py --vocab_size 2000 --file_pat '*.txt' --tr_input_dir ./train-txt-files/ --tst_input_dir ./test-txt-files/ --tr_vec_file ./train.2k.vec --vocab_file ./2k.vocab  --tst_vec_file ./test.2k.vec --txt_mode
   

TMNT does its own rudimentary pre-processing of the text and includes a built-in stop-word list for English
to remove certain common terms that tend to act as distractors for the purposes of generating coherent topics.
This pre-processing is implemented in Python and relatively unoptimized.  
