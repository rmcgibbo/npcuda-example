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

1. `gpuadder.cu`, `gpuadder.hh`: Implementation and interface file for the central C++ class. The function of this
class is to manage the GPU memory and execution lifecycle, and generally act as a "window" to expose the GPU side kernels.
The key methods are a constructor which takes as input an array and copies it over to the GPU, and a `retreive()` method
which copies back the data from the GPU to the CPU. Other methods on the C++ class can trigger CUDA kernels to perform
in place operations on the GPU memory. This trivial example contains only the code to increment all of the elements in an
array by one.

2. `device_kernel.cu`

3. `swig.i`