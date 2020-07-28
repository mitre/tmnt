from setuptools import setup, find_packages

requirements = [
    'autogluon==0.0.12',
    'mantichora==0.9.5',
    'pandas<1.0',
    'pyLDAvis==2.1.2',
    'pyOpenSSL==18.0.0',
    'PySocks==1.6.8',
    'sacremoses>=0.0.35',
    'sentence-splitter==1.4',
    'scikit-learn<0.23',
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
