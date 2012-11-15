# npcuda-example

This is an example of a simple Python C++ extension which uses CUDA and is compiled via nvcc. The idea is to use this coda as an example or template from which to build your own CUDA-accelerated Python extensions.

The extension is a single C++ class which manages the GPU memory and provides methods to call operations on the GPU
data. This C++ class is wrapped via *swig* or *cython* -- effectively exporting this class into python land.

To use this code in your own work, refer to the LICENSE file.

## swig vs cython

### swig
Swig is a widely used code generator for exposing C and C++ libraries in high level dynamically typed languages.
In principle, it involves minimal code rewriting. You just have to write swig interface files that instruct swig
on how to do the translation.

In practice, swig and numpy don't work together that well. The numpy interface relies on a bunch of magical macros
that are extremely difficult to debug.

### !!cython!!
Cython is sweet. It's basically python, with optional static type declarations and the ability to call c functions
directly. Take a look at `wrapper.pyx`. It looks like python, but it gets translated into C, and then compiled into
a shared object file which you import from python (look at the `test.py` file)

Cython is the way to go.

## difference from PyCUDA

The point of this project is not to enable you to access the CUDA API in python, to write cuda code in strings and have
them be dynamically compiled, or anything like that.

Instead, the goal is to demonstrate some of the biolerplate and tricks needed to make a CPython extension module that
uses CUDA compiled with setuptools/distutils just like your standard C exension modules.

## authors
- Robert McGibbon
- Yutong Zhao

## installation

Requirements:
- python 2.7
- python setuptools, numpy
- nvcc. I'm using version 4.2
- nose (for testing)

- swig for the swig wrapping method. I've tested with version 1.3.40
- cython for the cython wrapping method. I've tested with version 0.16

### NOTE: You should REALLY be using EPD python (http://www.enthought.com/products/epd.php). It will come with everything.


To install, `cd` into your directory of choice -- either `swig` or `cython`. Then, just run `$ python setup.py install`. To see if everything is working, run `$ nosetests`

Silence is golden!

