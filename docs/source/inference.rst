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
  >>> infer = BowNTMInference('_model_dir/model.params','_model_dir/model.config', '_model_dir/vocab.json')
  >>> text_encoder = TextEncoder(infer)
  >>> encodings = text_encoder.encode_batch(['While space flight has been ongoing for decades, thoughts of returning to the moon have dominated nasa engineers thinking in recent years.', 'Sales have been slow with increases in shipping costs depending on brand items or generic'])

The resulting ``encodings`` is an ``NDArray`` with shape ``(N,K)`` where ``N`` is the number of texts/documents encoded and ``K`` is the number of topics.
  
