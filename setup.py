from setuptools import setup


setup(name='SFA',
      version='1.0.1',
      url='https://github.com/hggzjx/SFA',
      license='Apache 2',
      author='Aurelien Coet',
      author_email='21349110@suibe.edu.cn',
      description='Implementation of the Modeling Selective Feature Attention for Lightweight Text Matching',
      packages=[
        'esim'
      ],
      install_requires=[
        'wget',
        'numpy',
        'nltk',
        'matplotlib',
        'tqdm',
        'torch'
      ])
