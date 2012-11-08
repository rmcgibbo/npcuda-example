## npcuda-example

This is an example of a simple python C++ extension which uses CUDA and is compiled via nvcc.
The idea is to use this coda as an example or template from which to build your own python CUDA extensions.

The extension is a single C++ class which manages the GPU memory and provides methods to call operations on the GPU data.
This C++ class is wrapped via swig -- effectively exporting this class into python.

# Instalation

Requirements
- python 2.7
- python setuptools
- swig. Tested with version 1.3.40
- nvcc

`python setup.py install`

# The Files