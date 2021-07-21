.. TMNT documentation master file, created by
   sphinx-quickstart on Mon Mar 18 14:49:48 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Topic Modeling Neural Toolkit Documentation
===========================================

The Topic Modeling Neural Toolkit (TMNT) provides implementations for
training and applying neural network variational autoencoders (VAEs) on text data in order to
model latent topics in large documents collections.

A few highlights include:

 * Use of AutoGluon for hyperparameter and architecture optimization
 * Multiple latent distributions, including the von Mises Fisher distribution 
 * Ability to use pre-trained word embeddings in the encoder
 * Use of the ``PyLDAvis`` library to visualize learned topics
 * Runtime/inference API to allow for easy deployment of learned topic models
 * Ability to perform Guided Topic Modeling by explicitly adding topic terms and the
   use of a novel regularization method
 * Dynamic topic modeling (of topics over time) through the use of `covariates`
 * Experimental Transformer-based Sequence Variational Auto-Encoder


Table of Contents
=================


.. toctree::
   :glob:
   :caption: Overview
   :maxdepth: 3
	      
   installing/installation
   about/what_is
   getting_started/common_formats

.. toctree::
   :glob:
   :caption: Core
   :maxdepth: 3

   getting_started/preprocessing
   model_training/training   
   model_selection/model_selection
   options/command_options
   inference/evaluation.rst
   inference/inference.rst

.. toctree::
   :glob:
   :caption: Examples
   :maxdepth: 3

   auto_examples/index.rst

.. toctree::
   :glob:
   :caption: Advanced Use
   :maxdepth: 3
	      
   user_guide/modeling_covariates
   embeddings/embeddings
   

.. toctree::
   :glob:
   :caption: API
   :maxdepth: 1

   api

   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
