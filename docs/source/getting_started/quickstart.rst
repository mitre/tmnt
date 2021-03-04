Quickstart Guide
================

Training a topic model requires both a training file containing sparse vector representations of documents
along with a test/validation file in the same format. In addition, a vocabulary file is needed to
map token indices back to their string representations.  


Preparing text data
++++++++++++++++++++++

The sparse vector representation for a corpus can be obtained from two different input formats:
1) json objects with one document per json object or 2) plain text documents (one document per file) 

Using the JSON input format, the value of the key ``text`` will be used as the document string.
All other fields are ignored. So, for example::


  {"id": "3664", "text": "This is the text of one of the documents in the corpus."}
  {"id": "3665", "text": "This is the text of another document in the corpus."}
  ...

Two directories of such files should be provided, one for training and one for test.  Assuming the files end with ``.json`` extensions, the
following example invocation would prepare the data for the training and test sets, creating vector representations with a vocabulary
size of 2000.  Note that this script uses the built in pre-processing which tokenizes, downcases and removes common English stopwords.
An example invocation::

  python bin/prepare_corpus.py --vocab_size 2000 --file_pat '*.json' --tr_input ./train-json-files/ \
  --val_input ./val-json-files/ --tr_vec_file ./train.2k.vec --vocab_file ./2k.vocab  --val_vec_file ./val.2k.vec 

TMNT does its own rudimentary pre-processing of the text leveraging the sklearn `CountVectorizer`.
The preprocessor defaults to `CountVectorizer`'s default English stop words. Custom stop-word lists can also be provided
via a command-line option to `prepare_corpus.py`.  Finer-grained control over preprocessing is available
using the API.

Training a Topic Model
++++++++++++++++++++++

To train a topic model on the ouputs of the `prepare_corpus.py` script, the `train_model.py` script is used::

    python bin/train_model.py --tr_vec_file ./train.2k.vec \
    --val_vec_file ./val.2k.vec --vocab_file ./2k.vocab \
    --config ./examples/train_model/model.2.config \
    --save_dir ./_exps/ --log_level info

Note that most of the configuration options for training a topic model are provided by the supplied configuration
file passed as the argument to the `--config` option.  Configuration files are described in
detail :ref:`training-label`




