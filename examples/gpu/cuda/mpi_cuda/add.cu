#include <stdio.h>      // stdio functions are used since C++ streams aren't necessarily thread safe
#include <unistd.h> 
 
// a simple kernel that simply increments each array element by b
__global__ void kernelAddConstant(int *g_a, const int b)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_a[idx] += b;
}
 
// a predicate that checks whether each array elemen is set to its index plus b
int correctResult(int *data, const int n, const int b)
{
        for(int i = 0; i < n; i++){
 	//	printf("%d ne %d + %d\n", data[i],i,b);
                if(data[i] != i + b)
                {
			printf("%d ne %d + %d\n", data[i],i,b);
                        return 0;
		}
	}
        return 1;
}

int local_sleep()
{
	sleep(5);
	return 0;
}

 
extern "C" int nv_main(int argc, char *argv[])
{
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
    printf("number of CUDA devices:\t%d\n", num_gpus);
    for(int i = 0; i < num_gpus; i++)
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
                printf("   %d: %s\n", i, dprop.name);
    }
        printf("---------------------------\n");
 
 
    ////////////////////////////////////////////////////////////////r
    // initialize data
        //
    unsigned int n = num_gpus * 8192;
    unsigned int nbytes = n * sizeof(int);
        int *a = 0;             // pointer to data on the CPU
        int b = 3;              // value by which the array is incremented
        a = (int*)malloc(nbytes);
        if(0 == a)
        {
                printf("couldn't allocate CPU memory\n");
                return 1;
        }
        for(unsigned int i = 0; i < n; i++)
        a[i] = i;
     
 
    ////////////////////////////////////////////////////////////////
        // run as many CPU threads as there are CUDA devices
        //   each CPU thread controls a different device, processing its
        //   portion of the data.  It's possible to use more CPU threads
        //   than there are CUDA devices, in which case several CPU
        //   threads will be allocating resources and launching kernels
        //   on the same device.  For example, try omp_set_num_threads(2*num_gpus);
        //   Recall that all variables declared inside an "omp parallel" scope are
        //   local to each CPU thread
        //
       // omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA devices
    //omp_set_num_threads(2*num_gpus);// create twice as many CPU threads as there are CUDA devices
#pragma omp parallel
    {
        unsigned int cpu_thread_id = 0; // omp_get_thread_num();
                unsigned int num_cpu_threads = 1; // omp_get_num_threads();
// local_sleep();
                // set and check the CUDA device for this CPU thread
                int gpu_id = -1;
                cudaSetDevice(cpu_thread_id % num_gpus);        // "% num_gpus" allows more CPU threads than GPU devices
                cudaGetDevice(&gpu_id);
 
                printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id, num_cpu_threads, gpu_id);
 
                int *d_a = 0;   // pointer to memory on the device associated with this CPU thread
                int *sub_a = a + cpu_thread_id * n / num_cpu_threads;   // pointer to this CPU thread's portion of data
                unsigned int nbytes_per_kernel = nbytes / num_cpu_threads;
                dim3 gpu_threads(128);  // 128 threads per block
                dim3 gpu_blocks(n / (gpu_threads.x * num_cpu_threads));
 
          cudaMalloc((void**)&d_a, nbytes_per_kernel);
          cudaMemset(d_a, 0, nbytes_per_kernel);
          cudaMemcpy(d_a, sub_a, nbytes_per_kernel, cudaMemcpyHostToDevice);
        kernelAddConstant<<<gpu_blocks, gpu_threads>>>(d_a, b);
 
          cudaMemcpy(sub_a, d_a, nbytes_per_kernel, cudaMemcpyDeviceToHost);
          cudaFree(d_a);
 
 
    }
        printf("---------------------------\n");
 
        if(cudaSuccess != cudaGetLastError())
                printf("%s\n", cudaGetErrorString(cudaGetLastError()));
 
 
        ////////////////////////////////////////////////////////////////
        // check the result
        //
    if(correctResult(a, n, b))
        printf("Test PASSED\n");
    else
        printf("Test FAILED\n");
 
    free(a);    // free CPU memory
 
    cudaThreadExit();
 
    return 0;
}
