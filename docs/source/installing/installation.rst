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

  conda create --name TMNT python=3.10 

Activate the environment::

  conda activate TMNT


Pip Installation
++++++++++++++++

To install the most recent stable version of TMNT::
  
  pip install tmnt

Install (latest; nightly) TMNT via pip::

  pip install tmnt --pre


Source Installation
+++++++++++++++++++

TMNT can be installed from source::

  git clone https://github.com/mitre/tmnt.git

or via SSH::

  git clone git@github.com:mitre/tmnt.git


Installation proceeds with a local pip install::

  cd tmnt
  pip install -e . 

