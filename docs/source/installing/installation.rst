Installation
~~~~~~~~~~~~

.. note::
   TMNT is currently only supported for Linux/MacOS. While the following steps should in theory
   work for Windows, some users have reported problems with installation of some of the 3rd party libraries.


Downloading TMNT
+++++++++++++++++++

TMNT can be obtained via GitHub::

  git clone https://github.com/mitre/tmnt.git

or via SSH::

  git clone git@github.com:mitre/tmnt.git


Dependencies and Environment
+++++++++++++++++++++++++++++++

TMNT is easiest to use via Conda (Miniconda or Anaconda). If
Conda is not already installed, grab the necessary install script from:

https://docs.conda.io/en/latest/miniconda.html

Once Conda is installed, install a new environment for TMNT as follows::

  conda create --name TMNT -c conda-forge python=3 pip numpy==1.17.0

In some cases, the conda-forge versions may run into SSL timeouts. If the
above command returns a HTTP 000 CONNECTION FAILED message, it is suggested
to run the command again after disabling SSL as follows::

  conda config --set ssl_verify false

Once the environment is setup, activate it and install TMNT-specific libraries::

  conda activate TMNT 
  cd tmnt
  pip install -r requirements.txt

If using a GPU, replace the last line above with ::
  
  pip install -r requirements.gpu.txt

Finally, TMNT must be installed as a package locally by running::

  python setup.py develop

  
Building the Test Model
++++++++++++++++++++++++++

The following invocation should train a topic model on the example provided 20 news data
for 27 training epochs::

  python bin/train_model.py --tr_vec_file ./data/train.vec \
  --val_vec_file ./data/test.vec --vocab_file ./data/train.vocab \
  --config ./examples/train_model/model.2.config \
  --save_dir ./_exps/ --log_level info

In general, TMNT assumes a test/validation corpus is available to determine the validation perplexity
and coherence, specified with the ``val_vec_file`` option.  If a validation file is not available/needed
it may be ommitted in which case no evaluation is performed.  See the :ref:`training-label`.
