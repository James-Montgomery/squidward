from setuptools import setup
# https://python-packaging.readthedocs.io/en/latest/minimal.html

setup(name='squidward',
      version='0.1',
      description='Package for implementing Gaussian Process models',
      url='TBD',
      author='James Montgomery',
      author_email='DontEmailMe@fake.com',
      license='MIT',
      packages=['squidward'],
      install_requires=[
          'numpy>=1.15.1',
          'scipy>=1.1.0',
          'matplotlib>=2.2.3',
          'seaborn>=0.9.0',
          'nose>=1.3.7'
      ],
      zip_safe=False)
