Modeling Covariates
===================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:


In many use-cases of topic modeling, additional meta-data is available associated
with each document such as the document source, time/date of publication, author(s),
etc. A key advantage of neural architectures for topic modeling is their flexibility.
TMNT includes a simple mechanism to introduce meta-data associated with each
document as covariates (currently just a single co-variate is supported).

The co-variate model largely follows the approach outlined in Card et al. (2018).
A single (one-hot) input categorical variable is provided as an additional feature
to both the encoder and the decoder. Let :math:`e_i` denote the probability assigned
by the decoder for each item in the vocabulary for document :math:`i`. Let
:math:`c_i` denote the one-hot vector encoding of the covariate for document :math:`i`
and let :math:`d` denote the bias vector capturing background term frequencies.
:math:`W` denotes the base term-topic matrix, :math:`W^{Cov}` captures
how each covariate value influences each term in the vocabulary (this is sort of
a prior over terms for each co-variate value) and finally :math:`W^{Int}` captures
interactions by conceptually providing a separate term-topic matrix for each
co-variate value. The resulting term probabilities for an input are determined by:

.. math::

   e_i = softmax(d + \theta_i^{\top}W + c_i^{\top}W^{Cov} + (\theta_i \otimes c_i)^{\top}W^{Int}


Preparing a Corpus with Covariate Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``prepare_corpus.py`` script is used to generate labeled vector files for training
with co-variates. Currently, only the json list format can be used. The flag
``--json_text_key`` should be used to indicate which json key has a value that corresponds
to the document input text.  The ``--json_label_key`` flag is used to denote which key
has a value that should be interpreted as a co-variate.  In the example, below
the meta-data associated with the key `year` is used as meta-data.  For convenience
with certain types of labels (e.g. years or other date information), the command-line
option ``--label_prefix_chars`` will use only the first `N` characters of the co-variate
value::

  python bin/prepare_corpus.py --tr_input_dir ./train.json.batches/ \
    --tst_input_dir ./test.json.batches/ --tr_vec_file ./ex.train.vec \
    --tst_vec_file ./ex.test.vec --vocab_size 5000 --vocab_file ./ex.vocab \
    --json_text_key text --json_label_key year --file_pat '*.json' --label_prefix_chars 3


The addition of the option ``--label_prefix_chars 3`` would use just the first three
characters/digits of the four digit year, capturing the decade instead of the
exact year.


Training a Co-variate Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once a corpus has been prepared as above and includes the desired labels/co-variates
in the sparse vector files, training proceeds as with standard TMNT models except that
the user must include the flag ``--use_labels_as_covars``::

  python bin/train_model.py --tr_vec_file ./ex.train.vec --tst_vec_file ./ex.test.vec \
  --vocab_file ./ex.vocab --save_dir ./_exps/ --model_dir ./_models/ \
  --use_labels_as_covars --config ./model.config 


Inspecting Resulting Topics
~~~~~~~~~~~~~~~~~~~~~~~~~~~
  
Currently, a simple script is provided to inspect the word-deviation matrices. The script
requires the resulting trained model along with a simple text file that includes the co-variate
values, one on each line, in the order they should appear in the output::

  python bin/print_covariate_model.py --model_dir ./_models/ --num_terms 10 \
  --covariate_values ./cv_values_decades.txt --output_file ./dynamic.un.output.txt
