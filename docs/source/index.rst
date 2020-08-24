.. TMNT documentation master file, created by
   sphinx-quickstart on Mon Mar 18 14:49:48 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Topic Modeling Neural Toolkit Documentation
===========================================

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


Table of Contents
=================

.. toctree::
   :caption: User Guide
   :maxdepth: 3

   user_guide/installation
   user_guide/quickstart
   user_guide/training
   user_guide/command_options
   user_guide/config_options
   user_guide/model_selection
   user_guide/evaluation
   user_guide/inference
   user_guide/hpc
   user_guide/modeling_covariates
   user_guide/guided
   user_guide/sequence_vae

.. toctree::
   :caption: API
   :maxdepth: 3
	      
   apidoc/tmnt


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
