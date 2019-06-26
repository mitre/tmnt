Evaluation and Visualization
============================

.. toctree::
    :hidden:
    :maxdepth: 2

.. contents::
    :local:

Evaluation
~~~~~~~~~~

Trained models can be evaluated against a test corpus using the ``evaluate.py`` script.  An example::

  python bin/evaluate.py --test_file ./data/test.2.vec --vocab_file ./data/train.2.vocab --model_dir ./_model_dir/ --words_per_topic 10

This will provide the top N words (based on the argument to ``--words_per_topic``) for each topic to standard output.
Also, it will provide the normalized point-wise mutual information-based coherence score.


The ``evaluate.py`` script can also encode the test documents and produce an embedding visualized
using UMAP ( https://umap-learn.readthedocs.io/en/latest/ ) by simply adding the ``--plot_file`` argument
specifying the output PNG file. Note that on a Mac, typically Python must be installed as a Framework
and ``pythonw`` to interpret the program, e.g.::

  pythonw bin/evaluate.py --test_file ./data/test.2.vec --vocab_file ./data/train.2.vocab --model_dir ./_model_dir/ --words_per_topic 10 \
                          --plot_file 20news.plot.png

Note that the argument to ``--vocab_file`` must be the original (non JSON) vocab file used as the input to ``bin/train_model.py``.

Visualization
~~~~~~~~~~~~~

A simple interactive visualization of a trained topic model is possible using the ``export_model.py`` script.
This script has a few output files, but the primary visualization is done using an invocation such as::

  python bin/export_model.py --model_dir ./_model_dir/ --vec_file ./data/test.2.vec --html_vis ./20news.html

The resulting ``.html`` file should load into any browser.

