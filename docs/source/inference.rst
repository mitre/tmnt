Inference with Topic Models
===========================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:


Document Encodings
~~~~~~~~~~~~~~~~~~

Trained topic models have a number of potential uses.  One key piece of functionality is to take
a text document and provide an encoding of the document that corresonds to the distribution
of latent topics present in the text.  In practical terms, the topic model encoder provides
an algorithm for ``doc2vec`` where the vector representation can be interpreted as the latent
topic distribution.

To use a trained model in this fashion within a larger application, or perhaps as a web-service
the ``tmnt.bow_vae.runtime`` module provides a simple API to encode documents::

  python -i tmnt/bow_vae/runtime.py
  >>> infer = BowNTMInference('_model_dir/model.params','_model_dir/model.specs', '_model_dir/vocab.json')
  >>> top_k_terms = infer.get_top_k_words_per_topic(10)
  >>> encoding1 = infer.encode_texts([['gun', 'gun', 'gun', 'shooting', 'police', 'defense', 'gun', 'gun', 'gun']])
  >>> encoding2 = infer.encode_texts([['space', 'flight', 'shuttle', 'space', 'flight', 'shuttle', 'space', 'flight', 'shuttle',]])
  
