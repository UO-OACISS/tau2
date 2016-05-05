#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <string>
#include <sstream>
using namespace std;
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

void multiply_by_element(dim3 grid, dim3 threads, float *d_a, float *d_b, float *d_c, int m, cudaStream_t cStream)
{

	cudaError err;
	unsigned int matsize = SIZE_OF_MATRIX*SIZE_OF_MATRIX*sizeof(float);
	float* c = (float*)malloc(matsize);
	
	multiply_matrices<<< grid, threads, 0, cStream >>>(d_a, d_b, d_c, m);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		cout << "error in kernel, " << cudaGetErrorString(err) << endl;
	}
	
	cudaStreamSynchronize(cStream);
	err = cudaMemcpyAsync(c, d_c, matsize, cudaMemcpyDeviceToHost, cStream);
	if (err != cudaSuccess)
	{
		cout << "error in memcpy, #=" << cudaGetErrorString(err) << endl;
	}


}

void multiply_by_block(dim3 grid, dim3 threads, float *d_a, float *d_b, float *d_c, int m, cudaStream_t cStream)
{
	cudaError err;
	unsigned int matsize = SIZE_OF_MATRIX*SIZE_OF_MATRIX*sizeof(float);
	float* c = (float*)malloc(matsize);

	multiply_matrices_shared_blocks<<< grid, threads, 0, cStream >>>(d_a, d_b, d_c, m);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		cout << "error in kernel, " << cudaGetErrorString(err) << endl;
	}

	cudaStreamSynchronize(cStream);
	err = cudaMemcpyAsync(c, d_c, matsize, cudaMemcpyDeviceToHost, cStream);
	if (err != cudaSuccess)
	{
		cout << "error in memcpy, #=" << cudaGetErrorString(err) << endl;
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

	//cout << "blocks: " << number_of_blocks << " threads: " <<
	//number_of_threads << endl;

	//cout.flush();

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
			//cout << a[i*m+j] << ", ";
		}
		//cout << endl;
	}
	cudaError_t err;

	int count = 0;

	err = cudaGetDeviceCount(&count);

	cout << count << " devices found." << endl;

	string device_list("");
	int number_of_iterations = 1;
	
	int opt = getopt(argc, argv, "d:i:");
	while(opt != -1) {
		stringstream str;
		switch(opt) {
			case 'd':
				device_list = string(optarg);
				break;
			case 'i':
				str << optarg;
				str >> number_of_iterations;
				break;
			case '?':
				if (optopt == 'd')
					cerr << "Error, option -d requires argument: comma delimted list of devices to run on." << endl;
				else if (optopt == 'i')
					cerr << "Error, option -i requires argument: number of iterations to run." << endl;
				else
					cerr << "Error, unknow option. Usage:\nmatmult [-d <device id>,...] [-i <number of iterations]" << endl;
				return 1;
			default:
				break;
		}
	  opt = getopt(argc, argv, "d:i:");
	}
	int devices[count];
	int nDevices = 0;
	//default: use all the devices
	if (device_list.compare("") == 0)
	{
		for (int d=0;d<count;d++)
		{
			devices[d] = d;
		}
		nDevices = count;
	}
	else
	{
		for (int d=0;d<count;d++)
		{
			stringstream str;
			str << d;
			char c = 0;
			if (str >> c) {
				if (device_list.find(c) != string::npos) {
					devices[nDevices++] = d;
				}
			}
		}
	}
	//cout << "finnished mapping devices." << endl;
	float *d_a[nDevices], *d_b[nDevices], *d_c[nDevices];
	cudaStream_t streams[nDevices];
	for (int d=0;d<nDevices;d++)
	{
		cudaSetDevice(devices[d]);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, devices[d]);
		cout << "Using device " << devices[d] << ", name: " << deviceProp.name << endl;

		err = cudaSetDevice(devices[d]);
		if (err != cudaSuccess)
		{
			cout << "error setting device, #=" << cudaGetErrorString(err) << endl;
		}
		err = cudaStreamCreate(&streams[d]);
		if (err != cudaSuccess)
		{
			cout << "error in stream creation, #=" << cudaGetErrorString(err) << endl;
		}



		err = cudaMalloc((void **) &d_a[d], matsize);
		if (err != cudaSuccess)
		{
			cout << "error in malloc, #=" << cudaGetErrorString(err) << endl;
		}
		err = cudaMalloc((void **) &d_b[d], matsize);
		if (err != cudaSuccess)
		{
			cout << "error in malloc, #=" << cudaGetErrorString(err) << endl;
		}
		err = cudaMalloc((void **) &d_c[d], matsize);
		if (err != cudaSuccess)
		{
			cout << "error in malloc, #=" << cudaGetErrorString(err) << endl;
		}
		
	}

	for (int i=0; i<number_of_iterations*nDevices; i++)
	{
		int cDevice = i%nDevices;
		cudaStream_t cStream = streams[cDevice];
		cudaSetDevice(devices[cDevice]);
		if (err != cudaSuccess)
		{
			cout << "error setting device: " << devices[i%nDevices] << " #=" << cudaGetErrorString(err) << endl;
		}

		err = cudaMemcpyAsync(d_a[cDevice], a, matsize, cudaMemcpyHostToDevice, cStream);
		if (err != cudaSuccess)
		{
			cout << "error in memcpy, #=" << cudaGetErrorString(err) << endl;
		}
		err = cudaMemcpyAsync(d_b[cDevice], b, matsize, cudaMemcpyHostToDevice, cStream);
		if (err != cudaSuccess)
		{
			cout << "error in memcpy, #=" << cudaGetErrorString(err) << endl;
		}

		//cout << "running on device " << cDevice << endl;

		dim3 grid(number_of_blocks, number_of_blocks);
		dim3 threads(number_of_threads, number_of_threads, 1);

		//multiply each element at a time.
		multiply_by_element(grid, threads, d_a[cDevice], d_b[cDevice], d_c[cDevice], m, cStream);

		//multiply by first load a 16x16 submatrix into shared memory.
		multiply_by_block(grid, threads, d_a[cDevice], d_b[cDevice], d_c[cDevice], m, cStream);
	}

	cout << "Finished " << number_of_iterations << " iterations on " << nDevices << " devices." << endl;

	for (int d=0;d<nDevices;d++)
	{
		cudaSetDevice(devices[d]);
		cudaStreamSynchronize(streams[d]);
	}
	for (int d=0;d<nDevices;d++)
	{
		cudaStreamDestroy(streams[d]);
	}
	//print c	
	/*
	cout << " results: " << endl;
	for (int i=0; i<m; i++) {
		for (int j=0; j<m; j++) {
			cout << c[i*m+j] << ", ";
		}
		cout << endl;
	}
	*/
	
	
	//print c	
	/*
	cout << " results: " << endl;
	for (int i=0; i<m; i++) {
		for (int j=0; j<m; j++) {
			cout << c[i*m+j] << ", ";
		}
		cout << endl;
	}
	*/
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	cudaThreadExit();
}
