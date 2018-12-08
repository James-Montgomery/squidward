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
          'numpy',
          'scipy',
          'matplotlib',
          'seaborn'
      ],
      zip_safe=False)
