#include "matmult.h"
#include "matmult_init.h"
#include <math.h>
#include <algorithm>
using namespace std;

#define SIZE_OF_MATRIX 1000
#define SIZE_OF_BLOCK 16 
#define M SIZE_OF_MATRIX
//unsigned int m = SIZE_OF_MATRIX;


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
	initialize(a,b,c,M);
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
	multiply_by_element(grid, threads, d_a, d_b, d_c, M);

	//multiply by first load a 16x16 submatrix into shared memory.
	multiply_by_block(grid, threads, d_a, d_b, d_c, M);
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

