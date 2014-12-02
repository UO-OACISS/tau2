/*
 * Copyright 2014 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print sass to source correlation
 */ 

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <stdio.h>

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define CUPTI_CALL(call)                                                    \
  do {                                                                      \
    CUptiResult _status = call;                                             \
    if (_status != CUPTI_SUCCESS) {                                         \
      const char *errstr;                                                   \
      cuptiGetResultString(_status, &errstr);                               \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
              __FILE__, __LINE__, #call, errstr);                           \
      exit(-1);                                                             \
    }                                                                       \
  } while (0)

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer)) 

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;


__global__
void transpose(float *d_Outdata, const float *d_Indata)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[threadIdx.y+j][threadIdx.x] = d_Indata[(y+j)*width + x];
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    d_Outdata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

int
main(int argc, char *argv[])
{
  const int nx = 32;
  const int ny = 32;
  const int mem_size = nx*ny*sizeof(float);
  dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
  dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
  cudaDeviceProp g_deviceProp;

  // initTrace();

  RUNTIME_API_CALL(cudaGetDeviceProperties(&g_deviceProp, 0));
  printf("Device Name: %s\n", g_deviceProp.name);
  if (g_deviceProp.major < 2) {
    printf("INSTRUCTION EXECUTION not supported on pre-Fermi devices\n");
    return 0;
  }

  float *d_X, *d_Y;

  float *h_X = (float*)malloc(mem_size);
  float *h_Y = (float*)malloc(mem_size);
  if (!(h_X && h_Y)) {
    printf("Malloc failed\n");
    exit(-1);
  }
  // initialization of host data
  for (int j = 0; j < ny; j++) {
    for (int i = 0; i < nx; i++) {
      h_X[j*nx + i] = j*nx + i;
    }
  }
  RUNTIME_API_CALL(cudaMalloc(&d_X, mem_size));
  RUNTIME_API_CALL(cudaMalloc(&d_Y, mem_size));

  RUNTIME_API_CALL(cudaMemcpy(d_X, h_X, mem_size, cudaMemcpyHostToDevice));

  transpose<<<dimGrid, dimBlock>>>(d_Y, d_X);

  RUNTIME_API_CALL(cudaMemcpy(h_Y, d_Y, mem_size, cudaMemcpyDeviceToHost));

  free(h_X);
  free(h_Y);

  cudaFree(d_X);
  cudaFree(d_Y);

  cudaDeviceSynchronize();
  cudaDeviceReset();

  // finiTrace();
  return 0;
}

