#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <unistd.h>
#include <math.h>
#include <algorithm>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

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

int source_lines = 67;
const char* multiply_matrices_source[] = {

"#define SIZE_OF_BLOCK 16\n",
"int idx(int i, int j, int lda)\n",
"{\n",
	"return ((j) + ((i)*(lda)));\n",
"}\n",
"__kernel void null_kernel()\n",
"{}\n",
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
"		/*for (int j=0; j < bs; j++)\n",
"		{\n",
"			c = c + a[idx(sub_row, j, lda)] * b[idx(j, sub_col, lda)];\n",
"		}*/\n",
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
        int rank, size, len = 0;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        char name[MPI_MAX_PROCESSOR_NAME]; 
  
        MPI_Get_processor_name(name, &len);
        printf("MPI: Rank %d out of %d on %s\n", rank, size, name); 
     
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
	ci = clGetPlatformIDs(1, &cpPlatform, NULL);
	CHECK_CL_ERROR(ci);

	cl_uint nDevices, count;
	cl_device_id *cdDevices = NULL;
	ci = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &count);
	
	cdDevices = (cl_device_id *)malloc(count * sizeof(cl_device_id));
	ci = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, count, cdDevices, NULL);
	//ci = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 1, &cdDevice, NULL);
	CHECK_CL_ERROR(ci);

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
	cl_device_id* devices = (cl_device_id*) malloc(count * sizeof(cl_device_id));
	nDevices = 0;
	//default: use all the devices
	if (device_list.compare("") == 0)
	{
		for (int d=0;d<count;d++)
		{
			devices[d] = cdDevices[d];
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
					devices[nDevices++] = cdDevices[d];
				}
			}
		}
	}
	//cout << "finnished mapping devices." << endl;

	//cl_context GPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &ci);
	cl_context GPUContext = clCreateContext(0, nDevices, devices, NULL, NULL, &ci);
	CHECK_CL_ERROR(ci);

	cl_command_queue cqCommandQueue[nDevices];

	for (int d=0;d<nDevices;d++)
	{
		char name[256];
		clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(name), &name, NULL);
		cout << "Using device name: " << name << endl;
	
		cqCommandQueue[d] = clCreateCommandQueue(GPUContext, devices[0], CL_QUEUE_PROFILING_ENABLE, &ci);
		CHECK_CL_ERROR(ci);

	}



	cl_program OpenCLProgram = clCreateProgramWithSource(GPUContext, source_lines, multiply_matrices_source, NULL, &ci);
	CHECK_CL_ERROR(ci);

	ci = clBuildProgram(OpenCLProgram, 0, NULL, NULL, NULL, NULL);
	CHECK_CL_ERROR(ci);

	char log[60000];
	ci = clGetProgramBuildInfo(OpenCLProgram, devices[0], CL_PROGRAM_BUILD_LOG,
	60000, log, NULL);

	CHECK_CL_ERROR(ci);
	
	//printf("build log: %s\n", log);
	//cout << log << endl;

	size_t thread_size[] = {number_of_threads, number_of_threads};
	size_t block_size[] = {number_of_blocks, number_of_blocks};
  /*	
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
	ci = clSetKernelArg(OpenCL_multiply_matrices_shared_blocks, 3, sizeof(float) * SIZE_OF_BLOCK * SIZE_OF_BLOCK, 0);
	CHECK_CL_ERROR(ci);
	ci = clSetKernelArg(OpenCL_multiply_matrices_shared_blocks, 4, sizeof(float) * SIZE_OF_BLOCK * SIZE_OF_BLOCK, 0);
	CHECK_CL_ERROR(ci);
	ci = clSetKernelArg(OpenCL_multiply_matrices_shared_blocks, 5, sizeof(int), (void *) &m);
	CHECK_CL_ERROR(ci);


	cl_event event, shared_event;
	ci = clEnqueueNDRangeKernel(cqCommandQueue,
	OpenCL_multiply_matrices_shared_blocks, 2, NULL, block_size, thread_size, 0, NULL, &shared_event);
	//cl_kernel OpenCL_null_kernel = clCreateKernel(OpenCLProgram,
	//"null_kernel", &ci);
	//CHECK_CL_ERROR(ci);
	//ci = clEnqueueNDRangeKernel(cqCommandQueue, OpenCL_null_kernel, 2, NULL, block_size, thread_size, 0, NULL, &shared_event);
	CHECK_CL_ERROR(ci);
	*/
	cl_kernel OpenCL_multiply_matrices = clCreateKernel(OpenCLProgram, "multiply_matrices", &ci);
	CHECK_CL_ERROR(ci);

	cl_event event, event_mem, event_read;
	cl_mem d_a, d_b, d_c;

	for (int i=0; i<number_of_iterations*nDevices; i++)
	{
		cl_command_queue cCQ = cqCommandQueue[i%nDevices];

		d_a = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, matsize, a, &ci);
		CHECK_CL_ERROR(ci);
		d_b = clCreateBuffer(GPUContext, CL_MEM_READ_ONLY |
		CL_MEM_COPY_HOST_PTR, matsize, b, &ci);
		CHECK_CL_ERROR(ci);

		d_c = clCreateBuffer(GPUContext, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, matsize, c, &ci);
		CHECK_CL_ERROR(ci);

		clSetKernelArg(OpenCL_multiply_matrices, 0, sizeof(cl_mem), (void *) &d_a);
		clSetKernelArg(OpenCL_multiply_matrices, 1, sizeof(cl_mem), (void *) &d_b);
		clSetKernelArg(OpenCL_multiply_matrices, 2, sizeof(cl_mem), (void *) &d_c);
		clSetKernelArg(OpenCL_multiply_matrices, 3, sizeof(int), (void *) &m);

		event_mem = clCreateUserEvent(GPUContext, &ci);
		clEnqueueWriteBuffer(cCQ, d_a, CL_TRUE, 0, matsize, a, 0, NULL, &event_mem);
		clEnqueueWriteBuffer(cCQ, d_b, CL_TRUE, 0, matsize, b, 0, NULL, &event_mem);
		clWaitForEvents(1, &event_mem);
		
		event = clCreateUserEvent(GPUContext, &ci);
		CHECK_CL_ERROR(ci);

		ci = clEnqueueNDRangeKernel(cCQ, OpenCL_multiply_matrices, 2, NULL,
		block_size, thread_size, 0, NULL, &event);
		CHECK_CL_ERROR(ci);
		
		//clWaitForEvents(1, &shared_event);
		clWaitForEvents(1, &event);
		CHECK_CL_ERROR(ci);

		event_read = clCreateUserEvent(GPUContext, &ci);
		ci = clEnqueueReadBuffer(cCQ, d_c, CL_TRUE, 0, matsize, c, 0, NULL, &event_read);
		CHECK_CL_ERROR(ci);
		//clWaitForEvents(1, &event_read);
		//clFinish(cCQ);

	}
	
	cout << "Finished " << number_of_iterations << " iterations on " << nDevices << " devices." << endl;
	/*
	std::cout << " results: " << std::endl;
	for (int i=0; i<m; i++) {
		for (int j=0; j<m; j++) {
			std::cout << c[i*m+j] << ", ";
		}
		std::cout << std::endl;
	}
	*/	

	free(a);
	free(b);
	free(c);

	clReleaseKernel(OpenCL_multiply_matrices);
	clReleaseProgram(OpenCLProgram);
	for (int d=0;d<nDevices;d++)
	{
		clFinish(cqCommandQueue[d]);
		clReleaseCommandQueue(cqCommandQueue[d]);
	}
	clReleaseContext(GPUContext);
	clReleaseMemObject(d_a);
	clReleaseMemObject(d_b);
	clReleaseMemObject(d_c);

        MPI_Finalize(); 
}
