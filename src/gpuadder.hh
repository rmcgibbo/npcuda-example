class GPUAdder {
  // pointer to the GPU memory where the array is stored
  int* array_device;
  // pointer to the CPU memory where the array is stored
  int* array_host;
  // length of the array (number of elements)
  int length;
  
public:
  GPUAdder(int* INPLACE_ARRAY1, int DIM1); // constructor (copies to GPU)
  ~GPUAdder(); // destructor
  void increment(); // does operation inplace on the GPU
  void retreive(); //gets results back from GPU
  void retreive_to(int* ARGOUT_ARRAY1, int DIM1); //gets results back from GPU
};
