#include <omp.h>
#include <stdio.h>      // stdio functions are used since C++ streams aren't necessarily thread safe
#include <unistd.h> 


#include <iostream>
#include <math.h>
#include <functional>
#include <stdlib.h>     
#include <time.h>      

#define ROW_TILE_WIDTH 32
#define COL_TILE_WIDTH 32

template<typename T>
__global__
void naive_matrix_multiply(T *A, T *B, T* C, int width, int C_rows, int C_cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;   
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  // check boundry conditions
  if( row < C_rows && col < C_cols ){
    // do the multiplication for one row and col
    T value = 0;
    for(int k = 0; k < width; k++){
      value += A[row * width + k] * B[k * C_cols + col];
    }
    // store result
    C[row * C_cols + col] = value;
  }
  

}

template<typename T>
void initialize_matrix(T* M, int rows, int cols, std::function<float()> F) {
  for(int i = 0; i < rows; i++){
    for(int j = 0; j < cols; j++){
      M[i * cols + j] = F();
    }
  }
}




int mult_with_size(int size) {
  int A_rows = size;
  int A_cols = size;
  int B_rows = size;
  int B_cols = size;
  int C_rows = A_rows;
  int C_cols = B_cols;
  int A_size = A_rows * A_cols;
  int B_size = B_rows * B_cols;
  int C_size = C_rows * C_cols;
  float *A, *B, *C, *C_cpu;

  cudaMallocManaged(&A, A_size*sizeof(float));
  cudaMallocManaged(&B, B_size*sizeof(float));
  cudaMallocManaged(&C, C_size*sizeof(float));
  cudaMallocManaged(&C_cpu, C_size*sizeof(float));

  srand (time(NULL));
  auto rand_numbers = []() -> float {
    auto f = static_cast<float>(rand())/(static_cast<float>(RAND_MAX/1000));
    int n = static_cast<int>(f);
    return static_cast<float>(n);
  };

  initialize_matrix<float>(A, A_rows, A_cols, rand_numbers);
  initialize_matrix<float>(B, B_rows, B_cols, rand_numbers);

  dim3 dim_grid(C_cols/COL_TILE_WIDTH, C_rows/ROW_TILE_WIDTH, 1);
  dim3 dim_block(COL_TILE_WIDTH, ROW_TILE_WIDTH, 1);

  naive_matrix_multiply<float><<<dim_grid, dim_block>>>(A, B, C, A_cols, C_rows, C_cols);

  cudaDeviceSynchronize();
  

  // Free memory
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  
  return 0; 
}

extern "C" int nv_main(int argc, char * argv[]) {
    int num_gpus = 0;       // number of CUDA GPUs

        /////////////////////////////////////////////////////////////////
        // determine the number of CUDA capable GPUs
        //
    cudaGetDeviceCount(&num_gpus);
        if(num_gpus < 1)
        {
                printf("no CUDA capable devices were detected\n");
                return 1;
        }

        /////////////////////////////////////////////////////////////////
        // display CPU and GPU configuration
        //
    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("number of CUDA devices:\t%d\n", num_gpus);
    for(int i = 0; i < num_gpus; i++)
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
                printf("   %d: %s\n", i, dprop.name);
    }
    printf("-------");
    mult_with_size(32);
    mult_with_size(64);
    mult_with_size(128);
    mult_with_size(256);
    mult_with_size(512);
    mult_with_size(1024);
    mult_with_size(2048);
    mult_with_size(4096);
    mult_with_size(8192);
    mult_with_size(16*1024);
    cudaDeviceReset();
    return 0;
}
 
