import os
from os import environ
from setuptools import setup, find_packages
from setuptools.command.install import install

with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as version_file:
    version = version_file.read().strip()

try:
    if not os.getenv('RELEASE'):
        from datetime import date
        today = date.today()
        day = today.strftime("b%Y%m%d")
        version += day
except Exception:
    pass

if __name__ == '__main__':
    setup_requires = [
        'numpy>=1.5.0'
    ]
    install_requires = [
        'numpy>=1.19.0,<1.20.0',
        'gluonnlp==0.10.0',
        'autogluon.core==0.3.1',
        'mantichora==0.9.5',
        'pandas<2.0',
        ## pyLDAvis dependencies expressed here to avoid long dep search
        'pyLDAvis==2.1.2',
        'MarkupSafe>=2.0',
        'joblib>=0.8.4',
        'future>=0.18.2',
        'funcy>=1.16',
        'pandas>=1.3.3',
        'pyOpenSSL==18.0.0',
        'PySocks==1.6.8',
        'sacremoses>=0.0.38',
        'sentence-splitter==1.4',
        'scikit-learn>=0.24.1',
        'numba<=0.52.0',
        'umap-learn==0.4.6',
        'tabulate>=0.8.7'
    ]
    ## need to stick with mxnet < 1.8 to maintain compatibility on Windows
    if(environ.get('USE_CUDA') == '1'):
        #install_requires.append('mxnet-cu102==1.8.0')
        install_requires.append('mxnet-cu101<1.8.0,>=1.6.0')
    else:
        install_requires += ['mxnet<1.8.0,>=1.6.0']

    class GPUCommand(install):
        user_options = install.user_options + [
            ('gpu', None, 'GPU install option'),
        ]

        def initialize_options(self):
            install.initialize_options(self)
            self.gpu = None

        def finalize_options(self):
            install.finalize_options(self)

        def run(self):
            install.run(self)

    setup(name=("tmnt-cu101" if environ.get('USE_CUDA') == '1' else "tmnt"),
          version=version,
          author="The MITRE Corporation",
          author_email="wellner@mitre.org",
          description="Topic modeling neural toolkit",
          url="https://github.com/mitre/tmnt.git",
          license='Apache',
          classifiers = [
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: Apache Software License",
              "Operating System :: OS Independent"
          ],
          python_requires='>=3.7',
          setup_requires=setup_requires,
          install_requires=install_requires,
          packages=find_packages())
