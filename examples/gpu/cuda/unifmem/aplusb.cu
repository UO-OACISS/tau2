#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cupti.h>

__global__ void AplusB(int *ret, int a, int b) { 
  ret[threadIdx.x] = a + b + threadIdx.x; 
} 

int main() { 
  int *ret; cudaMallocManaged(&ret, 1000 * sizeof(int)); 
  AplusB<<< 1, 1000 >>>(ret, 10, 100); 
  cudaDeviceSynchronize(); 
  for(int i=0; i<1000; i++) 
    printf("%d: A+B = %d\n", i, ret[i]); 
  cudaFree(ret); 
  return 0; 
}

