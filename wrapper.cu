#include <kernel.cu>
#include <assert.h>
#include <iostream>
using namespace std;

class GPUAdder {
  // pointer to the GPU memory where the array is stored
  int* array_device;
  // pointer to the CPU memory where the array is stored
  int* array_host;
  // length of the array (number of elements)
  int length;
  
public:
  GPUAdder(int*, int); // constructor (copies to GPU)
  ~GPUAdder(); // destructor
  void increment(); // does operation inplace on the GPU
  void retreive(); //gets results back from GPU
};

GPUAdder::GPUAdder (int* array_host_, int length_) {
  array_host = array_host_;
  length = length_;
  int size = length * sizeof(int);

  cudaError_t err = cudaMalloc((void**) &array_device, size);
  assert(err == 0);

  cudaMemcpy(array_device, array_host, size, cudaMemcpyHostToDevice);
}

void GPUAdder::increment() {
  kernel_add_one<<<1, length>>>(array_device);
  //for (int i = 0; i < length; i++) {
  //  array_host[i] += 1;
  //}
}

void GPUAdder::retreive() {
  int size = length * sizeof(int);
  cudaMemcpy(array_host, array_device, size, cudaMemcpyDeviceToHost);
}

GPUAdder::~GPUAdder() {
  cudaFree(array_device);
}

int main() {
  const int blocksize = 15;
  int a[blocksize] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  GPUAdder g(a, blocksize);
  g.increment();
  g.retreive();
  
  for (int i = 0; i < blocksize; i++) {
    cout << a[0] << " ";
  }
}
