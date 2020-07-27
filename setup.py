from setuptools import setup, find_packages


requirements = [
    'autogluon==0.0.12',
    'gluonnlp>=0.8.1',
    'mantichora==0.9.5',
    'mxnet==1.6.0',
    'pyLDAvis==2.1.2',
    'pyOpenSSL==18.0.0',
    'PySocks==1.6.8',
    'sacremoses==0.0.35',
    'sentence-splitter==1.4',
    'sphinx-gallery==0.2.0',
    'sphinx-rtd-theme==0.4.3',
    'umap-learn==0.3.7' 
    ]

requirements_gpu = [
    'autogluon==0.0.12',
    'gluonnlp>=0.8.1',
    'mantichora==0.9.5',
    'mxnet_cu101==1.6.0',
    'pyLDAvis==2.1.2',
    'pyOpenSSL==18.0.0',
    'PySocks==1.6.8',
    'sacremoses==0.0.35',
    'sentence-splitter==1.4',
    'sphinx-gallery==0.2.0',
    'sphinx-rtd-theme==0.4.3',
    'umap-learn==0.3.7' 
    ]

if __name__ == '__main__':
    setup(name="tmnt",
          version="0.2",
          author="The MITRE Corporation",
          url="https://github.com/mitre/tmnt.git",
          license='Apache',
          install_requires = requirements
    )
