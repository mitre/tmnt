.. TMNT documentation master file, created by
   sphinx-quickstart on Mon Mar 18 14:49:48 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Topic Modeling Neural Toolkit Documentation
===========================================

The Topic Modeling Neural Toolkit (TMNT) provides implementations for
training and applying neural network variational autoencoders (VAEs) on text data in order to
model latent topics in large documents collections.


Table of Contents
=================

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Intro	      

   about/what_is

.. toctree::
   :glob:
   :caption: Installation
   :maxdepth: 3
	      
   installing/installation

.. toctree::
   :glob:
   :caption: Guide
   :maxdepth: 3
	      
   getting_started/quickstart
   getting_started/prepare_corpus
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
