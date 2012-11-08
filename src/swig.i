%module gpuadder

%{
    #define SWIG_FILE_WITH_INIT
    #include "gpuadder.hh"    
%}

%include "numpy.i"

%init %{
    import_array();
%}

#%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* array_host_, int length_)}

%include "gpuadder.hh"