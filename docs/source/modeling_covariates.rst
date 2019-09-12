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

.. math::

   e_i = softmax(d + \theta_i^{T}W + c_i^TW^{Cov} + (\theta_i \ctimes c_i)^TW^{Int}


Preparing a Corpus with Covariate Information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preparation::

  python bin/prepare_corpus.py --tr_input_dir ./train.json.batches/ --tst_input_dir ./test.json.batches/ \
    --tr_vec_file ./ex.train.vec --tst_vec_file ./ex.test.vec --vocab_size 5000 --vocab_file ./ex.vocab \
    --json_text_key text --json_label_key year --file_pat '*.json' --label_prefix_chars 3


Training::

  python bin/train_model.py --tr_vec_file ./ex.train.vec --tst_vec_file ./ex.test.vec --vocab_file ./ex.vocab \
  --save_dir ./_exps/ --model_dir ./_models/ --use_labels_as_covars --config ./model.config 


Word-topic deviation matrices::

  python bin/print_covariate_model.py --model_dir ./_models/ --num_terms 10 \
  --covariate_values ./cv_values_decades.txt --output_file ./dynamic.un.output.txt
