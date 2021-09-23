Installation
~~~~~~~~~~~~

.. note::
   TMNT is currently only supported for Linux/MacOS. While the following steps should in theory
   work for Windows, some users have reported problems with installation of some of the 3rd party libraries.

Environment
+++++++++++

TMNT is often easiest to use via Conda (Miniconda or Anaconda). If
Conda is not already installed, grab the necessary install script from:

https://docs.conda.io/en/latest/miniconda.html

Once Conda is installed, setup a new environment for TMNT as follows::

  conda create --name TMNT python=3.7 --no-default-packages

Activate the environment::

  conda activate TMNT

If planning to use a GPU, ensure cudatoolkit and cudnn packages are installed::

  conda install cudatoolkit=10.1 cudnn=5.6.7
  

Pip Installation
++++++++++++++++

Install TMNT via pip for CPU-only::

  pip install tmnt --pre

Or with GPU (with Cuda 10.1)::

  pip install tmnt-cu101 --pre

Source Installation
+++++++++++++++++++

TMNT can be installed from source::

  git clone https://github.com/mitre/tmnt.git

or via SSH::

  git clone git@github.com:mitre/tmnt.git


Installation proceeds with a local pip install::

  cd tmnt
  pip install -e . 

If using a GPU, replace the last line above with ::
  
  USE_CUDA=1 pip install -e .

  
Testing the Installation
++++++++++++++++++++++++

If the installation was done by cloning the TMNT repository, the ``train_model.py`` script
in the ``bin/`` directory is a good place to start testing that everything is working correctly.

.. note::
   If installation was done without cloning the repository, download the ``train_model.py`` script
   from Github manually and/or clone the repository.

The following invocation should train a topic model on the example provided 20 news data
for 27 training epochs::

  python bin/train_model.py --tr_vec_file ./data/train.vec \
  --val_vec_file ./data/test.vec --vocab_file ./data/train.vocab \
  --config ./data/configs/train_model/model.2.config \
  --save_dir ./_exps/ --log_level info

As another test, try out the example: :ref:`sphx_glr_auto_examples_a_train_20news.py`

Troubleshooting
+++++++++++++++

There are a couple known platforms/environments where installing TMNT can be a challenge.

1. Windows installation.  TMNT can be installed on Windows, however it requires specific versions
   of Microsoft Visual C++.

2. Ubuntu installation.  Certain versions/images of Ubuntu, e.g. 18.04, have known problems using
   pre-built MXNet libraries.  To run TMNT on these platforms it may be necessary to build MXNet from
   source (see https://mxnet.apache.org/versions/1.8.0/get_started/build_from_source )
   
