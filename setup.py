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
        'optuna',
        'mantichora>=0.9.5',
        'transformers[torch]',
        'torcheval',
        ## pyLDAvis dependencies expressed here to avoid long dep search
        'pyLDAvis==3.4.0',
        'MarkupSafe>=2.0',
        'joblib>=0.8.4',
        'future>=0.18.2',
        'funcy>=1.16',
        'pandas==1.5.3',
        'pyOpenSSL==18.0.0',
        'PySocks==1.6.8',
        'sacremoses>=0.0.38',
        'sentence-splitter==1.4',
        'umap-learn==0.4.6',
        'tabulate>=0.8.7',
        'torch>=1.13.0',
        'torchtext>=0.13.0'
    ]

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
