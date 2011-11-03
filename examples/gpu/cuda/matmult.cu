#include <stdlib.h>
#include <iostream>
#include "cuda_runtime_api.h"

#define SIZE_OF_MATRIX 1000
#define SIZE_OF_BLOCK 16 
#define M SIZE_OF_MATRIX
unsigned int m = SIZE_OF_MATRIX;

#define idx(i,j,lda) ((j) + ((i)*(lda)))

__global__ void multiply_matrices(float *d_a, float *d_b, float *d_c, int lda)
{
	unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int id  = idx(row,col,lda);

	float ctemp = 0.0;

	if (row < M && col < M)
	{
		for (unsigned int j=0; j<M; j++)
		{
			ctemp = ctemp + d_a[idx(row,j,lda)] * d_b[idx(j,col,lda)];
		}
		d_c[id] = ctemp;
	}
}

__global__ void multiply_matrices_shared_blocks(float *d_a, float *d_b, float *d_c,
int lda)
{

	int bs = SIZE_OF_BLOCK;
	unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int id  = idx(row,col,lda);
	
	//submatrices
	float *sub_a, *sub_b;

	//shared submatrices	
	__shared__ float a[SIZE_OF_BLOCK][SIZE_OF_BLOCK], b[SIZE_OF_BLOCK][SIZE_OF_BLOCK];
	//temp element of d_c
	float c = 0;
	
	//top-level row,col of block
	int block_row = blockIdx.y * bs;
	int block_col = blockIdx.x * bs;

	//id inside each block
	int sub_row = threadIdx.y;
	int sub_col = threadIdx.x;
	
	//for each block	
	for (int k = 0; k < (M / bs); k++)
	{

	  sub_a = &d_a[idx(block_row, bs*k, lda)];
		sub_b = &d_b[idx(bs*k, block_col, lda)];
		a[sub_row][sub_col] = sub_a[idx(sub_row, sub_col, lda)];
		b[sub_row][sub_col] = sub_b[idx(sub_row, sub_col, lda)];
		
		//wait for all threads to complete copy to shared memory.	
		__syncthreads();

		//multiply each submatrix
		for (int j=0; j < bs; j++)
		{
			c = c + a[sub_row][j] * b[j][sub_col];
		}
	
		// move results to device memory.
		d_c[id] = c;
	
		// wait for multiplication to finish before moving onto the next submatrix.
		__syncthreads();
		
	}
}

void multiply_by_element(dim3 grid, dim3 threads, float *d_a, float *d_b, float *d_c, int m)
{

	cudaError err;
	unsigned int matsize = SIZE_OF_MATRIX*SIZE_OF_MATRIX*sizeof(float);
	float* c = (float*)malloc(matsize);
	
	multiply_matrices<<< grid, threads >>>(d_a, d_b, d_c, m);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cout << "error in kernel, " << cudaGetErrorString(err) << std::endl;
	}
	
	cudaDeviceSynchronize();
	err = cudaMemcpy(c, d_c, matsize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		std::cout << "error in memcpy, #=" << cudaGetErrorString(err) << std::endl;
	}


}

void multiply_by_block(dim3 grid, dim3 threads, float *d_a, float *d_b, float *d_c, int m)
{
	cudaError err;
	unsigned int matsize = SIZE_OF_MATRIX*SIZE_OF_MATRIX*sizeof(float);
	float* c = (float*)malloc(matsize);

	multiply_matrices_shared_blocks<<< grid, threads >>>(d_a, d_b, d_c, m);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		std::cout << "error in kernel, " << cudaGetErrorString(err) << std::endl;
	}

	cudaDeviceSynchronize();
	err = cudaMemcpy(c, d_c, matsize, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		std::cout << "error in memcpy, #=" << cudaGetErrorString(err) << std::endl;
	}
}

int main(int argc, char** argv)
{
	
	unsigned int number_of_threads = min(SIZE_OF_MATRIX, SIZE_OF_BLOCK);
	unsigned int number_of_blocks;
	if (SIZE_OF_MATRIX > SIZE_OF_BLOCK)
		number_of_blocks = ceil(SIZE_OF_MATRIX / ((float) SIZE_OF_BLOCK));
	else
		 number_of_blocks = 1;

	unsigned int matsize = SIZE_OF_MATRIX*SIZE_OF_MATRIX*sizeof(float);

	//std::cout << "blocks: " << number_of_blocks << " threads: " <<
	//number_of_threads << std::endl;

	std::cout.flush();
	cudaSetDevice(0);

	float* a = (float*)malloc(matsize);
	float* b = (float*)malloc(matsize);
	float* c = (float*)malloc(matsize);

	//initalize matrices
	for (int i=0; i<m; i++) {
		for (int j=0; j<m; j++) {
			//a[i*m+j] = i;
			//b[i*m+j] = i;
			a[i*m+j] = i-j*2 + i-j+1 + 1;
			b[i*m+j] = i-j*2 + i-j+1 + 1;
			c[i*m+j] = 0;
			//std::cout << a[i*m+j] << ", ";
		}
		//std::cout << std::endl;
	}

	float *d_a, *d_b, *d_c;
	cudaError_t err;
	err = cudaMalloc((void **) &d_a, matsize);
	if (err != cudaSuccess)
	{
		std::cout << "error in malloc, #=" << cudaGetErrorString(err) << std::endl;
	}
	err = cudaMalloc((void **) &d_b, matsize);
	if (err != cudaSuccess)
	{
		std::cout << "error in malloc, #=" << cudaGetErrorString(err) << std::endl;
	}
	err = cudaMalloc((void **) &d_c, matsize);
	if (err != cudaSuccess)
	{
		std::cout << "error in malloc, #=" << cudaGetErrorString(err) << std::endl;
	}

	err = cudaMemcpy(d_a, a, matsize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cout << "error in memcpy, #=" << cudaGetErrorString(err) << std::endl;
	}
	err = cudaMemcpy(d_b, b, matsize, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		std::cout << "error in memcpy, #=" << cudaGetErrorString(err) << std::endl;
	}

	dim3 grid(number_of_blocks, number_of_blocks);
	dim3 threads(number_of_threads, number_of_threads, 1);

	//multiply each element at a time.
	multiply_by_element(grid, threads, d_a, d_b, d_c, m);

	//multiply by first load a 16x16 submatrix into shared memory.
	multiply_by_block(grid, threads, d_a, d_b, d_c, m);
	//print c	
	/*
	std::cout << " results: " << std::endl;
	for (int i=0; i<m; i++) {
		for (int j=0; j<m; j++) {
			std::cout << c[i*m+j] << ", ";
		}
		std::cout << std::endl;
	}
	*/
	
	
	//print c	
	/*
	std::cout << " results: " << std::endl;
	for (int i=0; i<m; i++) {
		for (int j=0; j<m; j++) {
			std::cout << c[i*m+j] << ", ";
		}
		std::cout << std::endl;
	}
	*/

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	cudaDeviceSynchronize();
	cudaThreadExit();
}
