import os
from os import environ
from setuptools import setup, find_packages
from setuptools.command.install import install
from pathlib import Path

version = '0.7.61'

try:
    if not os.getenv('RELEASE'):
        from datetime import date
        today = date.today()
        day = today.strftime("b%Y%m%d")
        version += day
except Exception:
    pass


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


if __name__ == '__main__':
    setup_requires = [
        'numpy>=1.5.0'
    ]
    install_requires = [
        'optuna',
	'datasets',
        'mantichora>=0.9.5',
        'transformers[torch]',
        'torcheval',
        ## pyLDAvis dependencies expressed here to avoid long dep search
        'pyLDAvis==3.4.0',
        'MarkupSafe>=2.0',
        'joblib>=0.8.4',
        'future>=0.18.2',
        'funcy>=1.16',
        'pandas>=2.0.0',
        'pyOpenSSL==18.0.0',
        'PySocks==1.6.8',
        'sacremoses>=0.0.38',
        'sentence-splitter==1.4',
        'umap-learn[plot]>=0.5.5',
        'numba',
        'scipy==1.12.0',
        'tabulate>=0.8.7',
        'torch>=2.1.2'
    ]

    setup(name=("tmnt"),
          version=version,
          author="The MITRE Corporation",
          author_email="wellner@mitre.org",
          description="Topic modeling neural toolkit",
          long_description=long_description,
          long_description_content_type='text/markdown',
          url="https://github.com/mitre/tmnt.git",
          license='Apache',
          classifiers = [
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: Apache Software License",
              "Operating System :: OS Independent"
          ],
          python_requires='>=3.10',
          setup_requires=setup_requires,
          install_requires=install_requires,
          packages=find_packages())
