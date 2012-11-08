%module gpuadder

/* unfortunately the word getattr is in this file three times
   the first, in the %module, leads to the name of the file
   gpuaddr.py and the shared object _gpuaddr.so

   the second and third point to the header file which
   contains the class definition we want to wrap. */



%{
    #define SWIG_FILE_WITH_INIT
    #include "manager.hh"    
%}

%include "numpy.i"

%init %{
    import_array();
%}

/* Because gpuadder.hh uses the swig default names for variables being
   passed to methods which are supposed to be interpreted as arrays,
   we don't need the following line: */
// %apply (int* ARGOUT_ARRAY1, int DIM1) {(int* array_host_, int length_)}

/* if instead the names of the pointers were not the standard ones, this
   type of translation would be necessary.
   http://www.scipy.org/Cookbook/SWIG_NumPy_examples */

%include "manager.hh"