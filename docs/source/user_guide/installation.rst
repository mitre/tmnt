Installation
~~~~~~~~~~~~

.. note::
   TMNT is currently only supported for Linux/MacOS. While the following steps should in theory
   work for Windows, some users have reported problems with installation of some of the 3rd party libraries.


1. Downloading TMNT
+++++++++++++++++++

TMNT can be obtained via GitHub::

  git clone https://github.com/mitre/tmnt.git

or via SSH::

  git clone git@github.com:mitre/tmnt.git


2. Dependencies and Environment
+++++++++++++++++++++++++++++++

TMNT is easiest to use vai Conda (Miniconda or Anaconda). If
Conda is not installed, please install by grabbing the necessary install script from:

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

