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
  >>> encodings = text_encoder.encode_batch(['Greater Armenia would stretch from Karabakh, to the Black Sea, to the Mediterranean, so if you use the term Greater Armenia use it with care.', 'I have two pairs of headphones I\'d like to sell.  These are excellent, and both in great condition'])

The resulting ``encodings`` is an ``NDArray`` with shape ``(N,K)`` where ``N`` is the number of texts/documents encoded and ``K`` is the number of topics.

You can use the method ``mx.nd.argsort`` to get the order of components (i.e. topics) in ascending probability, e.g.::

  >>> mx.nd.argsort(encodings[0])

Note also the ``NDArray`` objects can be converted to numpy via::

  >>> enc_0_np = encodings[0].asnumpy()

  
  
