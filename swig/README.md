## files

1. `manager.cu`, `manager.hh`: Implementation and header file for the central C++ class. The function of this
class is to manage the GPU memory and execution lifecycle, and generally act as a "window" to expose the GPU side kernels. The key methods are a constructor which takes as input an array and copies it over to the GPU, and a `retreive()` method which copies back the data from the GPU to the CPU. Other methods on the C++ class can trigger CUDA kernels to perform in place operations on the GPU memory. This trivial project's kernel contains only the code to increment all of the elements in an array by one.

2. `kernel.cu`: This is a CUDA file where the GPU kernels go. The kernels are called by methods in the manager.

3. `gpuadder.i`: This is a swig interface file that gives instructions for translating manager.cu into class that can be accessed from python. Beyond simply the bridge, the main functionality provided by swig here is the ability to automatically resolve numpy arrays into C pointers with without too much extra programmer pain.

4. `setup.py`: Python setup script that runs the compilation.

## compilation pipeline

1. Right at the top of the `setup.py` script, we re-run the swig inteface builder. It takes in the `gpuadder.i` file (along with `manager.hh` and `numpy.i`) and builds a python module and a c++ file called `swig_wrap.cpp`.

2. `manager.cu` is compiled by NVCC into `manager.o`. `manager.cu` #includes `kernel.cu`, so that's how the kernels get incorporated. A special setuptools hack prevents NVCC from trying to make a .so file -- we only want the .o.

3. `swig_wrap.cpp` is compiled by GCC into `swig_wrap.o`

4. `swig_wrap.o` and `manager.o` are linked together by GCC into a final .so file that can be imported from python. This `.so` file has an underscore at the start of its name, and goes along with the python file generated in step 1 by swig, which has the same name but without the underscore. In this example, they are `gpuadder.py` and `_gpuadder.so`. These files get put into your python site-packages.

Note: the python module gets its name from the %module line in `gpuadder.i` (line 3). That name (in this case `gpuadder`) needs to match the name in the `py_modules` line of setup.py (line 130) so that the python file `gpuadder.py` and the shared object file `_gpuadder.so` match up.


## setuptools monkey patching and setup.py

setuptools does not know anything about NVCC by default, and doesn't really support having multiple compilers that are on the same OS (it can deal with microsoft vs. linux, but not with nvcc vs gcc). So we need some tricks.

To implement the "special hack" in step 2 above, we customize the compiler class used by setuptools to call NVCC for .cu files and its regular compiler (probably gcc) for all other files. We also use some special logic inthe extension class -- setting `extra_compile_args` as a dict -- so that we can specify separate compile arguments for gcc and nvcc.
