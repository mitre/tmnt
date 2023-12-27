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

