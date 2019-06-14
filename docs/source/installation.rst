Installation
~~~~~~~~~~~~

1. Downloading TMNT
+++++++++++++++++++

TMNT can be obtained via GitLab::

  git clone https://gitlab.mitre.org/milu/tmnt.git


Note that the test data provided in the TMNT archive uses `git lfs`, see details here for installing
and setting up in your git environment: https://git-lfs.github.com

2. Dependencies and Environment
+++++++++++++++++++++++++++++++

TMNT is easiest to use by installing all necessary dependencies with Conda (Miniconda or Anaconda). If
Conda is not installed, please install by grabbing the necessary install script from:

https://docs.conda.io/en/latest/miniconda.html

Once Conda is installed properly, install a new conda environment for TMNT as follows::

  conda create --name TMNT pip numpy==1.16.4


For some platforms, it may be preferred to install the necessary
C compiler/environment via conda by adding the `gxx_linux-64`
and `gcc_linux-64` targets.  If the `conda create` step above does
not work, try the following::

  conda create --name TMNT pip gxx_linux-64 gcc_linux-64 numpy==1.16.4

Once the environment is setup, activate it and install TMNT-specific libraries::

  conda activate TMNT 
  cd tmnt
  pip install -r requirements.txt


Finally, TMNT must be installed as a package locally by running::

  python setup.py develop

