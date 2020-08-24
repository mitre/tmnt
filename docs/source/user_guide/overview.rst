TMNT Overview
~~~~~~~~~~~~~

The Topic Modeling Neural Toolkit (TMNT) provides implementations for
training and applying neural network variational autoencoders (VAEs) on text data in order to
model latent topics in large documents collections.  The toolkit makes use of
a number of approaches outlined in recent papers as well as some novel additions.

A few highlights include:

* Use of HPBandster for Bayesian hyperparameter optimization

* Ability to perform Guided Topic Modeling by explicitly adding topic terms and the
  use of a novel regularization method

* Dynamic topic modeling (of topics over time) through the use of `covariates`

* Multiple latent distributions, including the von Mises Fisher distribution 

* Ability to use pre-trained word embeddings in the encoder

* Use of the ``PyLDAvis`` library to visualize learned topics

* Runtime/inference API to allow for easy deployment of learned topic models

* Experimental Transformer-based Sequence Variational Auto-Encoder

