from os import environ
from setuptools import setup, find_packages
from setuptools.command.install import install

if __name__ == '__main__':
    setup_requires = [
        'numpy>=1.5.0'
    ]
    install_requires = [
       'numpy>=1.5.0',
       'gluonnlp==0.9.1',
       'autogluon.core>=0.0.15b20201106',
       'mantichora==0.9.5',
       'pandas<2.0',
       'pyLDAvis==2.1.2',
       'pyOpenSSL==18.0.0',
       'PySocks==1.6.8',
       'sacremoses>=0.0.38',
       'sentence-splitter==1.4',
       'scikit-learn<0.23,>=0.22.0',
       'umap-learn==0.3.7',
       'tabulate>=0.8.7'
    ]
    if(environ.get('USE_CUDA') == '1'):
        install_requires.append('mxnet-cu101==1.7.0')
    else:
        install_requires += ['mxnet==1.7.0.post1']

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

    setup(name="tmnt",
          version="0.6",
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
