/*
This is the central piece of code. This file implements a class
(interface in gpuadder.hh) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU

This class will get translated into python via swig
*/

#include <kernel.cu>
#include <manager.hh>
#include <assert.h>
#include <iostream>
using namespace std;

GPUAdder::GPUAdder (int* array_host_, int length_) {
  array_host = array_host_;
  length = length_;
  int size = length * sizeof(int);
  cudaError_t err = cudaMalloc((void**) &array_device, size);
  assert(err == 0);
  err = cudaMemcpy(array_device, array_host, size, cudaMemcpyHostToDevice);
  assert(err == 0);
}

void GPUAdder::increment() {
  kernel_add_one<<<64, 64>>>(array_device, length);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);
}

void GPUAdder::retreive() {
  int size = length * sizeof(int);
  cudaMemcpy(array_host, array_device, size, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  if(err != 0) { cout << err << endl; assert(0); }
}

void GPUAdder::retreive_to (int* array_host_, int length_) {
  assert(length == length_);
  int size = length * sizeof(int);
  cudaMemcpy(array_host_, array_device, size, cudaMemcpyDeviceToHost);
  cudaError_t err = cudaGetLastError();
  assert(err == 0);
}

GPUAdder::~GPUAdder() {
  cudaFree(array_device);
}
