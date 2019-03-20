Guided Topic Models
===================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:


Influencing Topic Terms
~~~~~~~~~~~~~~~~~~~~~~~

TMNT provides a simple method to influence or *guide* the topics the model discovers.
A user's preference for topics is articulated by providing sets of terms grouped together
referred to as *seed sets*. The group of terms appearing together in a seed set
should, according to the user, feature prominently in a single topic. In addition,
such terms should *not* be prominent in one or more *other* topics. The coercion of
the topic model is achieved through a regularizer that considers the topic-term weights
associated with each term in a given seed set tries to minimize the entropy of each term
across the topics.

Provided Seed Sets
~~~~~~~~~~~~~~~~~~

Seed sets are specified with a simple JSON file, best shown via example ::

  { "topic 0": ["god", "jesus", "belief"],
    "topic 1": ["nsa", "disk", "computer"],
    "topic 2": ["sale", "shipping", "condition"],
    "topic 3": ["turks", "turkish", "armenian"]
  }

The string serving as the key for each seed set is not interpreted and provided for convenience.
Each of the terms must exist in the topic vocabulary.  Note that it is recommended that each
seed set include the same number of terms, though this is not a requirement.
