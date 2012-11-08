void __global__ kernel_add_one(int* a) {
    a[threadIdx.x] += 1;
}
