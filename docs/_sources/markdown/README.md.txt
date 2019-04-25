# Squidward

After working with gaussian processes (GPs) to build out robust reinforcement learning models in production for most of my early career as a machine learning engineer (MLE), I became frustrated with the packages available for building GPs. They often focus using the latest in optimization tools and are far from the elegant, efficient, and simple design that I believe a GP package should embody.

This is my attempt to create the product that I would want to use. Something simple and flexible that gives knowledgable data scientists the tools they need to do the research or production machine learning work that they need.

I'm open to all feedback, commentary, and suggestions as long as they are constructive and polite.

### Authors

**James Montgomery** - *Initial work* - [jamesmontgomery.us](http://jamesmontgomery.us)

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Installing squidward

This is a step by step guide to installing squidward for your local environment.

I recommend installing squidward in a virtual environment for organized dependency control. Personally, I prefer conda environments. First, let's create and open our environment.

```
conda create --name squidward_env python=3.6
source activate squidward_env
```

Using the MKL backend for numpy can help increase the performance of this code. Anaconda now comes with mkl by default. To make use of mkl simply set up your virtual environment with anaconda like below.

```
conda create --name squidward_env python=3.6 anaconda
source activate squidward_env
```

To install the most recent stable version, simply [pip install from pypi](https://pypi.org/project/squidward/)! Squidward is installed and ready to use. 

```
pip install squidward
```

If you want the latest (not necessarily stable) version, git clone this repository to your local environment instead. To run unit and style tests you will need to clone this repository as tests are not included with the pypi installed package.

```
git clone https://github.com/looyclark/squidward.git
```

Change directory (cd) into the root of the squidward repository. Updates staged for the next stable release will be in the `master` branch and experimental updates will be in branches ending in the suffix "\_dev".

```
cd ./squidward
```

Install squidward using pip from the setup file.

```
pip install .
```

When you are finished using squidward you can either deactivate you conda environment for later use or remove it completely.

```
source deactivate
conda env remove -n squidward_env
```

### Examples / Tutorials

I've included basic examples of how to use squidward to get new users started building gaussian process models with this package.

Many of these examples include visualizations of data and GP models. This will require the visualization packages below:

```
pip install "matplotlib>=2.2.3' "seaborn>=0.9.0"
```

Examples and Tutorials can be found on the squidward [docs site](https://james-montgomery.github.io/squidward/)

## Testing

Testing is an important part of creating maintainable, production grade code. Below are instructions for running unit and style tests as well as installing the necessary testing packages. Tests have intentionally been separated from the installable pypi package for a variety of reasons.

### Install required packages

These packages are required to run unit and style tests for squidward.

```
pip install "nose>=1.3.7" "coverage==4.0.0" "pylint>=1.8.2"
```

### Running unit tests

To run the unit tests cd to `squidward/squidward` so that `/tests` is a subdirectory.

```
cd ./squidward/tests
```

Use `nosetests` to run all unit tests for squidward. If you installed squidward in a virtual environment, please run the tests in that same environment.

```
source activate squidward_env
nosetests
```

You can also run the tests with coverage to see what code within the package is called in the tests.

```
nosetests --with-coverage --cover-package=squidward
```

### Running style tests

I attempt to adhere to the [pep8](https://www.python.org/dev/peps/pep-0008/) style guide for the squidward project. To run the style tests cd to the root directory of the repository `squidward/` so that `/squidward` is a subdirectory. Use `pylint squidward` to run all style tests for squidward.

```
cd ./squidward
pylint squidward
```

Some of the naming conventions I've chosen intentionally do not adhere to pep8 in order to better resemble mathematical conventions. For example, I often borrow the matrix naming conventions of Rassmussen such as `K` and `K_ss`. You can run `pylint --disable=invalid-name` if you would like to ignore the resulting pylint warnings.

```
pylint squidward --disable=invalid-name
```

## Acknowledgments

A big thanks to Keegan Hines and Josh Touyz who introduced me to Gaussian Processes. Additionally, a big thanks to Jason Wittenbach and
Mack Sweeney who have helped guide me along my journey implementing Bayesian models from scratch.

[//]: # (Comment Section)

[//]: # (Updated the Github Docs)
[//]: # (for further help: https://github.com/James-Montgomery/misc_musings/tree/master/sphinx/docs)
[//]: # (cd ./docs)
[//]: # (make github)
