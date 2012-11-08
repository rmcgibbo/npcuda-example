## npcuda-example

This is an example of a simple python C++ extension which uses CUDA and is compiled via nvcc.
The idea is to use this coda as an example or template from which to build your own python CUDA extensions.

The extension is a single C++ class which manages the GPU memory and provides methods to call operations on the GPU data.
This C++ class is wrapped via swig -- effectively exporting this class into python.

# instalation

Requirements
- python 2.7
- python setuptools, numpy
- swig. Tested with version 1.3.40
- nvcc. I'm using version 4.2

`python setup.py install`

# files

1. `manager.cu`, `manager.hh`: Implementation and header file for the central C++ class. The function of this
class is to manage the GPU memory and execution lifecycle, and generally act as a "window" to expose the GPU side kernels.
The key methods are a constructor which takes as input an array and copies it over to the GPU, and a `retreive()` method
which copies back the data from the GPU to the CPU. Other methods on the C++ class can trigger CUDA kernels to perform
in place operations on the GPU memory. This trivial example contains only the code to increment all of the elements in an
array by one.

2. `kernel.cu`: This is a CUDA file where the GPU kernels go. It defines GPUfunctions that are called by operations
on the class defined in the manager

3. `swig.i`: This is a swig interface file that gives instructions for translating manager.cu into class that can be accessed
from python. Beyond simply the bridge, the main functionality provided by swig here is the ability to automatically resolve
numpy arrays into C pointers with minimal extra instructions.

4. `setup.py`: Python setup script that runs the compilation.

# compilation pipeline

1. Right at the top of the `setup.py` script, we re-run the swig inteface builder. It takes in the `swig.i` file (along with `manager.hh`
and `numpy.i`) and builds a python module and a c++ file called `swig_wrap.cpp`.

2. `manager.cu` is compiled by NVCC into `manager.o`. `manager.cu` #includes `kernel.cu`, so that's how the kernels get incorporated.
A special setuptools hack prevents NVCC from trying to make a .so file -- we only want the .o.

3. `swig_wrap.cpp` is compiled by GCC into `swig_wrap.o`

4. `swig_wrap.o` and `manager.o` are linked together by GCC into a final .so file that can be imported from python. This `.so` file has an
underscore at the start of its name, and goes along with the python file generated in step 1 by swig, which has the same name but without
the underscore. In this example, they are `gpuadder.py` and `_gpuadder.so`. These files get put into your python site-packages.

Note: the python module gets its name from the %module line in `swig.i` (line 3). That name (in this case `gpuadder`) needs to match the
name in the `py_modules` line of setup.py (line 112) so that the python file `gpuadder.py` and the shared object file `_gpuadder.so` match up.
