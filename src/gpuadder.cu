#include <device_kernel.cu>
#include <gpuadder.hh>
#include <assert.h>
#include <iostream>
using namespace std;

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
}

void GPUAdder::retreive() {
  int size = length * sizeof(int);
  cudaMemcpy(array_host, array_device, size, cudaMemcpyDeviceToHost);
}

void GPUAdder::retreive_to (int* array_host_, int length_) {
    assert(length == length_);
    int size = length * sizeof(int);
    cudaMemcpy(array_host_, array_device, size, cudaMemcpyDeviceToHost);
}

GPUAdder::~GPUAdder() {
  cudaFree(array_device);
}