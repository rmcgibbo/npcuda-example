import sys, os
from os.path import join as pjoin
from setuptools import setup
from distutils.unixccompiler import UnixCCompiler
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import subprocess
from subprocess import CalledProcessError
import glob
import numpy

def find_in_path(name, path):
   #adapted fom http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
   for dir in path.split(os.pathsep):
      binpath = pjoin(dir, name)
      if os.path.exists(binpath):
         return os.path.abspath(binpath)
   return None

def locate_cuda():
   # first check if the CUDAHOME env variable is in use
   if 'CUDAHOME' in os.environ:
      home = os.environ['CUDAHOME']
      nvcc = pjoin(home, 'bin', 'nvcc')
   else:
      # otherwise, search the PATH for NVCC
      nvcc = find_in_path('nvcc', os.environ['PATH'])
      if nvcc is None:
         raise EnvironmentError('The nvcc binary could not be '
             'located in your $PATH. Either add it to your path, or set $CUDAHOME')
      home = os.path.dirname(os.path.dirname(nvcc))

   cudaconfig = {'home':home, 'nvcc':nvcc,
                 'include': pjoin(home, 'include'),
                 'lib64': pjoin(home, 'lib64')}
   for k, v in cudaconfig.iteritems():
      if not os.path.exists(v):
         raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

   return cudaconfig

CUDA = locate_cuda()

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()


class MyExtension(Extension):
    """subclass extension to add the kwarg 'glob_extra_link_args'
    which will get evaluated by glob right before the extension gets compiled
    and let the swig shared object get linked against the cuda kernel
    """
    def __init__(self, *args, **kwargs):
        self.glob_extra_link_args = kwargs.pop('glob_extra_link_args', [])
        Extension.__init__(self, *args, **kwargs)

class NVCC(UnixCCompiler):
    src_extensions = ['.cu']
    executables = {'preprocessor' : None,
                   'compiler'     : [CUDA['nvcc']],
                   'compiler_so'  : [CUDA['nvcc']],
                   'compiler_cxx' : [CUDA['nvcc']],
                   # TURN OFF NVCC LINKING -- we're going to link with gcc instead
                   'linker_so'    : ["echo"],
                   'linker_exe'   : None,
                   'archiver'     : None,
                   'ranlib'       : None,
                   }

# this code will get compiled up to a .o file by nvcc. the final .o file(s) that
# it makes will be just one for each input source file. Note that we turned off
# the nvcc linker so that we don't make any .so files.
nvcc_compiled = Extension('this_name_is_irrelevant',
                          sources=['src/manager.cu'],
                          extra_compile_args=['-arch=sm_20', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"],
                          # we need to include src as an input directory so that the header files and device_kernel.cu
                          # can be found
                          include_dirs=[CUDA['include'], 'src'],
                          )

# the swig wrapper for gpuaddr.cu gets compiled, and then linked to gpuaddr.o
swig_wrapper = MyExtension('_gpuadder',
                         sources=['src/swig_wrap.cpp'],
                         library_dirs=[CUDA['lib64']],
                         libraries=['cudart'],
                         include_dirs = [numpy_include],
                         # extra bit of magic so that we link this
                         # against the kernels -o file
                         # this picks up the build/temp.linux/src/manager.cu
                         glob_extra_link_args=['build/*/*/manager.o'])


# this cusom class lets us build one extension with nvcc and one extension with regular gcc
# basically, it just tries to detect a .cu file ending to trigger the nvcc compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        # we're going to need to switch between compilers, so lets save both
        self.default_compiler = self.compiler
        self.nvcc = NVCC()
        build_ext.build_extensions(self)

    def build_extension(self, *args, **kwargs):
        extension = args[0]
        # switch the compiler based on which thing we're compiling
        # if any of the sources end with .cu, use nvcc
        if any([e.endswith('.cu') for e in extension.sources]):
            # note that we've DISABLED the linking (by setting the linker to be "echo")
            # in the nvcc compiler
            self.compiler = self.nvcc
        else:
            self.compiler = self.default_compiler

        # evaluate the glob pattern and add it to the link line
        # note, this suceeding with a glob pattern like build/temp*/gpurmsd/RMSD.o
        # depends on the fact that this extension is built after the extension
        # which creates that .o file
        if hasattr(extension, 'glob_extra_link_args'):
            for pattern in extension.glob_extra_link_args:
                unglobbed = glob.glob(pattern)
                if len(unglobbed) == 0:
                    raise RuntimeError("glob_extra_link_args didn't match any files")
                self.compiler.linker_so += unglobbed
        
        # call superclass
        build_ext.build_extension(self, *args, **kwargs)


if find_in_path('swig', os.environ['PATH']):
   subprocess.check_call('swig -python -c++ -o src/swig_wrap.cpp src/swig.i', shell=True)
else:
   raise EnvironmentError('the swig executable was not found in your PATH')

setup(name='gpuadder',
      # random metadata. there's more you can supploy
      author='Robert McGibbon',
      version='0.1',

      # this is necessary so that the swigged python file gets picked up
      py_modules=['gpuadder'],
      package_dir={'': 'src'},

      ext_modules=[nvcc_compiled, swig_wrapper],

      # inject our custom trigger
      cmdclass={'build_ext': custom_build_ext},

      # since the package has c code, the egg cannot be zipped
      zip_safe=False,
      )
