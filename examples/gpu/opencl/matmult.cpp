#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include "CL/cl.h"

using namespace std;

#define CHECK_CL_ERROR(e) \
if (e != CL_SUCCESS) \
{ \
	cout << "cl ERROR " << e << " : line " << __LINE__ << endl; \
}

#define SIZE_OF_MATRIX 1000
#define SIZE_OF_BLOCK 16

#define M SIZE_OF_MATRIX

unsigned int m = SIZE_OF_MATRIX;

int source_lines = 65;
const char* multiply_matrices_source[] = {

"#define SIZE_OF_BLOCK 16\n",
"int idx(int i, int j, int lda)\n",
"{\n",
	"return ((j) + ((i)*(lda)));\n",
"}\n",
"__kernel void multiply_matrices(__global const float *d_a, __global const float *d_b, __global float *d_c, int lda)\n",
"{\n",
	"unsigned int row = get_global_id(1);\n",
	"unsigned int col = get_global_id(0);\n",
	"unsigned int id  = idx(row,col,lda);\n",
	"float ctemp;\n",
	"if (row < lda && col < lda)\n",
	"{\n",
		"ctemp = 0;\n",
		"for (unsigned int j=0; j<lda; j++)\n",
		"{\n",
			"ctemp = ctemp + d_a[idx(row,j,lda)] * d_b[idx(j,col,lda)];\n",
		"}\n",
		"d_c[id] = ctemp;\n",
	"}\n",
"}\n",
"__kernel void multiply_matrices_shared_blocks(__global float *d_a, __global float *d_b, __global float *d_c, __local float* a, __local float* b, int lda)\n",
"{\n",
" int bs = SIZE_OF_BLOCK;",
"\n",
"	unsigned int row = get_global_id(1);\n",
"	unsigned int col = get_global_id(0);\n",
"	unsigned int id  = idx(row,col,lda);\n",
"	\n",
"\n",
"	//temp element of d_c\n",
"	float c = 0;\n",
"	\n",
"	//top-level row,col of block\n",
"	int block_row = get_group_id(1) * bs;\n",
"	int block_col = get_group_id(0) * bs;\n",
"\n",
"	//id inside each block\n",
"	int sub_row = get_local_id(1);\n",
"	int sub_col = get_local_id(0);\n",
"	\n",
"	//for each block	\n",
"	for (int k = 0; k < (lda / bs); k++)\n",
"	{\n",
"\n",
"		a[idx(sub_row, sub_col, lda)] = a[idx(row, col, lda)];\n",
"		b[idx(sub_row, sub_col, lda)] = b[idx(row, col, lda)];\n",
"		\n",
"		//wait for all threads to complete copy to shared memory.	\n",
"		barrier(CLK_LOCAL_MEM_FENCE);\n",
"\n",
"		//multiply each submatrix\n",
"		for (int j=0; j < bs; j++)\n",
"		{\n",
"			c = c + a[idx(sub_row, j, lda)] * b[idx(j, sub_col, lda)];\n",
"		}\n",
"	\n",
"		// move results to device memory.\n",
"		d_c[id] = c;\n",
"	\n",
"		// wait for multiplication to finish before moving onto the next submatrix.\n",
"		barrier(CLK_LOCAL_MEM_FENCE);\n",
"		\n",
"	}\n",
"}\n",
};

int main(int argc, char**argv)
{
	unsigned int number_of_threads = min(SIZE_OF_MATRIX, SIZE_OF_BLOCK);
	unsigned int number_of_blocks, block_mult;
	if (SIZE_OF_MATRIX > SIZE_OF_BLOCK)
		block_mult = ceil(SIZE_OF_MATRIX / ((float) SIZE_OF_BLOCK));
	else
		 block_mult = 1;
 	 
	
	number_of_blocks = SIZE_OF_BLOCK * block_mult;

	unsigned int matsize = SIZE_OF_MATRIX*SIZE_OF_MATRIX*sizeof(float);
	unsigned int submatsize = SIZE_OF_BLOCK*SIZE_OF_BLOCK*sizeof(float);

	//std::cout << "blocks: " << number_of_blocks << " threads: " <<
	//number_of_threads << std::endl;

	std::cout.flush();

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

	cl_int ci;
	cl_platform_id cpPlatform;
	clGetPlatformIDs(1, &cpPlatform, NULL);

	cl_device_id cdDevice;
	ci = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevice, NULL);
	//ci = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 1, &cdDevice, NULL);
	CHECK_CL_ERROR(ci);

	cl_context GPUContext = clCreateContext(0, 1, &cdDevice, NULL, NULL, &ci);
	CHECK_CL_ERROR(ci);

	cl_command_queue cqCommandQueue = clCreateCommandQueue(GPUContext, cdDevice, CL_QUEUE_PROFILING_ENABLE, &ci);
	CHECK_CL_ERROR(ci);

	cl_mem d_a = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY |
	CL_MEM_COPY_HOST_PTR, matsize, a, &ci);
	CHECK_CL_ERROR(ci);
	cl_mem d_b = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY |
	CL_MEM_COPY_HOST_PTR, matsize, b, &ci);
	CHECK_CL_ERROR(ci);

	c = NULL;
	cl_mem d_c = clCreateBuffer(GPUContext, CL_MEM_WRITE_ONLY, matsize, c, &ci);
	CHECK_CL_ERROR(ci);

	cl_program OpenCLProgram = clCreateProgramWithSource(GPUContext, source_lines, multiply_matrices_source, NULL, NULL);

	ci = clBuildProgram(OpenCLProgram, 0, NULL, NULL, NULL, NULL);
	CHECK_CL_ERROR(ci);

	char log[60000];
	ci = clGetProgramBuildInfo(OpenCLProgram, cdDevice, CL_PROGRAM_BUILD_LOG,
	60000, log, NULL);

	CHECK_CL_ERROR(ci);
	
	//printf("build log: %s\n", log);
	//cout << log << endl;

	size_t thread_size[] = {number_of_threads, number_of_threads};
	size_t block_size[] = {number_of_blocks, number_of_blocks};
	
	cl_mem sub_a = clCreateBuffer(GPUContext, CL_MEM_ALLOC_HOST_PTR, submatsize,
	NULL, NULL);
	cl_mem sub_b = clCreateBuffer(GPUContext, CL_MEM_ALLOC_HOST_PTR, submatsize,
	NULL, NULL);
	
	cl_kernel OpenCL_multiply_matrices_shared_blocks = clCreateKernel(OpenCLProgram,
	"multiply_matrices_shared_blocks", &ci);
	
	CHECK_CL_ERROR(ci);
	
	ci = clSetKernelArg(OpenCL_multiply_matrices_shared_blocks, 0, sizeof(cl_mem), (void *) &d_a);
	CHECK_CL_ERROR(ci);
	ci = clSetKernelArg(OpenCL_multiply_matrices_shared_blocks, 1, sizeof(cl_mem), (void *) &d_b);
	CHECK_CL_ERROR(ci);
	ci = clSetKernelArg(OpenCL_multiply_matrices_shared_blocks, 2, sizeof(cl_mem), (void *) &d_c);
	CHECK_CL_ERROR(ci);
	ci = clSetKernelArg(OpenCL_multiply_matrices_shared_blocks, 3, sizeof(cl_mem),
	NULL);
	CHECK_CL_ERROR(ci);
	ci = clSetKernelArg(OpenCL_multiply_matrices_shared_blocks, 4, sizeof(cl_mem),
	NULL);
	CHECK_CL_ERROR(ci);
	ci = clSetKernelArg(OpenCL_multiply_matrices_shared_blocks, 5, sizeof(int), (void *) &m);
	CHECK_CL_ERROR(ci);


	cl_event event, shared_event;
	ci = clEnqueueNDRangeKernel(cqCommandQueue,
	OpenCL_multiply_matrices_shared_blocks, 2, NULL, block_size, thread_size, 0, NULL, &shared_event);
	CHECK_CL_ERROR(ci);

	cl_kernel OpenCL_multiply_matrices = clCreateKernel(OpenCLProgram, "multiply_matrices", &ci);
	CHECK_CL_ERROR(ci);

	
	clSetKernelArg(OpenCL_multiply_matrices, 0, sizeof(cl_mem), (void *) &d_a);
	clSetKernelArg(OpenCL_multiply_matrices, 1, sizeof(cl_mem), (void *) &d_b);
	clSetKernelArg(OpenCL_multiply_matrices, 2, sizeof(cl_mem), (void *) &d_c);
	clSetKernelArg(OpenCL_multiply_matrices, 3, sizeof(int), (void *) &m);

	ci = clEnqueueNDRangeKernel(cqCommandQueue, OpenCL_multiply_matrices, 2, NULL,
	block_size, thread_size, 0, NULL, &event);
	CHECK_CL_ERROR(ci);
  
	clWaitForEvents(1, &shared_event);
	clWaitForEvents(1, &event);
	clFinish(cqCommandQueue);
	/*
	std::cout << " results: " << std::endl;
	for (int i=0; i<m; i++) {
		for (int j=0; j<m; j++) {
			std::cout << c[i*m+j] << ", ";
		}
		std::cout << std::endl;
	}
	*/	
	 

	clReleaseKernel(OpenCL_multiply_matrices);
	clReleaseProgram(OpenCLProgram);
	clReleaseCommandQueue(cqCommandQueue);
	clReleaseContext(GPUContext);
	clReleaseMemObject(d_a);
	clReleaseMemObject(d_b);
	clReleaseMemObject(d_c);

}
