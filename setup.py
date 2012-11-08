import sys
from setuptools import setup
from distutils.unixccompiler import UnixCCompiler
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
import subprocess
from subprocess import CalledProcessError
import glob
import IPython as ip

subprocess.check_call('swig -python -c++ -o src/swig_wrap.cpp src/swig.i', shell=True)

# make the clean command always run first
#sys.argv.insert(1, 'clean')
#sys.argv.insert(2, 'build')

# python distutils doesn't have NVCC by default. This is a total hack.
# what we're going to 

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
                   'compiler'     : ["nvcc"],
                   'compiler_so'  : ["nvcc"],
                   'compiler_cxx' : ["nvcc"],
                   'linker_so'    : ["echo"], # TURN OFF NVCC LINKING
                   'linker_exe'   : ["gcc"],
                   'archiver'     : ["ar", "-cr"],
                   'ranlib'       : None,
               }
    def __init__(self):
        # Check to ensure that nvcc can be located
        try:
            subprocess.check_output('nvcc --help', shell=True)
        except CalledProcessError:
            print >> sys.stderr, 'Could not find nvcc, the nvidia cuda compiler'
            sys.exit(1)
        UnixCCompiler.__init__(self)
        
nvcc_compiled = Extension('this_name_is_irrelevant',
                          sources=['src/gpuadder.cu'],
                          extra_compile_args=['-arch=sm_20', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"],
                          include_dirs=['/usr/local/cuda/include', 'src'],
                          )

swig_wrapper = MyExtension('gpuadder._gpuadder',
                         sources=['src/swig_wrap.cpp'],
                         library_dirs=['/usr/local/cuda/lib64'],
                         libraries=['cudart'],
                         # extra bit of magic so that we link this
                         # against the kernels -o file
                         glob_extra_link_args=['build/*/*/gpuadder.o'])

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

setup(name='gpuadder',
      author='Robert McGibbon',
      version='0.1',
      packages = ['gpuadder'],
      package_dir = {'gpuadder': 'src'},
      ext_modules=[nvcc_compiled, swig_wrapper],
      cmdclass={'build_ext': custom_build_ext},
      zip_safe=False,
      )
