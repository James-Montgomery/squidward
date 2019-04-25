import setuptools

# python3 -m pip install --user --upgrade setuptools wheel
# python3 setup.py sdist bdist_wheel
# python3 -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
# python3 -m pip install squidward

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="squidward",
    version="0.0.24",
    author="James Montgomery",
    author_email="jamesoneillmontgomery@gmail.com",
    description="Package for implementing Gaussian Process models in python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/James-Montgomery/squidward",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.16.2',
        'scipy>=1.1.0',
        #'distributed>=1.25.2'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
