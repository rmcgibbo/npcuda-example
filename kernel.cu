#include <stdio.h>

void __global__ kernel_add_one(int* a) {
    a[threadIdx.x] += 1;
}

int amain() {
    const int blocksize = 16;
    int a[blocksize] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    
    int *ad;
    int asize = blocksize*sizeof(int);

    cudaMalloc( (void**) &ad, asize);
    cudaMemcpy(ad, a, asize, cudaMemcpyHostToDevice);
    
    dim3 dimBlock( blocksize, 1 );
    dim3 dimGrid( 1, 1 );    
    
    kernel_add_one<<<dimGrid, dimBlock>>>(ad);
    
    cudaMemcpy(a, ad, asize, cudaMemcpyDeviceToHost);
    cudaFree(ad);
    
    printf("%d\n", a[0]);
    
    return 1;
}

