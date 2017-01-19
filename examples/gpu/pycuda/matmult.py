#!/usr/bin/env python
import pycuda.driver as cuda
import pycuda.autoinit
import math
from pycuda.compiler import SourceModule

#import tau

import numpy

SIZE_OF_MATRIX = 1000
SIZE_OF_BLOCK = 16

number_of_threads = int(min(SIZE_OF_MATRIX, SIZE_OF_BLOCK))
if (SIZE_OF_MATRIX > SIZE_OF_BLOCK):
	number_of_blocks = int(math.ceil(SIZE_OF_MATRIX / float(SIZE_OF_BLOCK)))
else:
  number_of_blocks = 1;


multiply_source = SourceModule("""

#define SIZE_OF_BLOCK 16
#define idx(i,j,lda) ((j) + ((i)*(lda)))

__global__ void multiply_matrices(float *d_a, float *d_b, float *d_c, int lda)
{
	//unsigned int row = threadIdx.y;
	//unsigned int col = threadIdx.x;
	unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;
	unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int id  = idx(row,col,lda);
	if (row < lda && col < lda)
	{
		for (unsigned int j=0; j < lda; j++)
		{
			d_c[id] = d_c[id] + d_a[idx(row,j,lda)] * d_b[idx(j,col,lda)];
		}
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
	for (int k = 0; k < (lda / bs); k++)
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
""")

def main():

	#a = numpy.matrix('2 -1 -4 ; 4 1 -2; 6 3 0').astype(numpy.float32)
	a = numpy.random.rand(SIZE_OF_MATRIX, SIZE_OF_MATRIX).astype(numpy.float32)
	#print a
	b = a
	c = numpy.zeros((SIZE_OF_MATRIX,SIZE_OF_MATRIX), dtype=numpy.float32)
	lda = numpy.int32(SIZE_OF_MATRIX)

	d_a = cuda.mem_alloc(a.nbytes)
	d_b = cuda.mem_alloc(b.nbytes)
	d_c = cuda.mem_alloc(c.nbytes)

	cuda.memcpy_htod(d_a, a)
	cuda.memcpy_htod(d_b, b)

	print "threads:", number_of_threads, "blocks: ", number_of_blocks

	multiply_matrices = multiply_source.get_function("multiply_matrices")
	multiply_matrices_shared_blocks = multiply_source.get_function("multiply_matrices_shared_blocks")
	
	multiply_matrices(d_a, d_b, cuda.InOut(c), lda,
										block=(number_of_threads,number_of_threads,1),
										grid=(number_of_blocks,number_of_blocks))	

	
	pycuda.driver.Context.synchronize()
	
	multiply_matrices_shared_blocks(d_a, d_b, cuda.InOut(c), lda,
										block=(number_of_threads,number_of_threads,1),
										grid=(number_of_blocks,number_of_blocks))	

	
	pycuda.driver.Context.synchronize()


	#print "results:"
	#print c



#tau.run('main()')
main()
