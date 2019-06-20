TMNT in High Performance Compute Environments
=============================================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:


Use of GPU
~~~~~~~~~~

In contrast to many deep neural network models, the variational autoencoders for TMNT can be fairly
lightweight in terms of computational requirements. A CPU is sufficient for building most models on
small to medium-sized datasets. Training a single model on a larger dataset and/or performing a
complex model search, however, benefits greatly from the acceration offered by GPUs.

A *single* GPU can be used by simply adding the ``--gpu`` flag, providing the integer logical device
ID as the argument.

