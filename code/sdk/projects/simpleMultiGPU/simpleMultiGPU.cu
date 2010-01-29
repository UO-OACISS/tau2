/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
 * This application demonstrates how to use the CUDA API to use multiple GPUs.
 *
 * There's one CUDA context per thread. To use multiple CUDA contexts you
 * have to create multiple threads. One for each GPU. For optimal performance,
 * the number of CPU cores should be equal to the number of GPUs in the system.
 *
 * Creating CPU threads has a certain overhead. So, this is only worth when you
 * have a significant amount of work to do per thread. It's also recommended to
 * create a pool of threads and reuse them to avoid this overhead.
 *
 * Note that in order to detect multiple GPUs in your system you have to disable
 * SLI in the nvidia control panel. Otherwise only one GPU is visible to the 
 * application. On the other side, you can still extend your desktop to screens 
 * attached to both GPUs.
 */



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cutil.h>
#include <multithreading.h>




////////////////////////////////////////////////////////////////////////////////
// Simple reduction kernel.
// Refer to the 'reduction' CUDA SDK sample describing
// reduction optimization strategies
////////////////////////////////////////////////////////////////////////////////
__global__ static void reduceKernel(float *d_Result, float *d_Input, int N){
    const int     tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadN = gridDim.x * blockDim.x;

    float sum = 0;
    for(int pos = tid; pos < N; pos += threadN)
        sum += d_Input[pos];

    d_Result[tid] = sum;
}



////////////////////////////////////////////////////////////////////////////////
// GPU thread
////////////////////////////////////////////////////////////////////////////////
typedef struct {
    //Device id
    int device;

    //Host-side input data
    int dataN;
    float *h_Data;

    //Partial sum for this GPU
    float *h_Sum;
} TGPUplan;

static CUT_THREADPROC solverThread(TGPUplan *plan){
    const int  BLOCK_N = 32;
    const int THREAD_N = 256;
    const int  ACCUM_N = BLOCK_N * THREAD_N;

    float
        *d_Data,
        *d_Sum;

    float
        *h_Sum;

    float sum;

    int i;
    //Set device
    CUDA_SAFE_CALL( cudaSetDevice(plan->device) );

    //Allocate memory
    CUDA_SAFE_CALL( cudaMalloc((void**)&d_Data, plan->dataN * sizeof(float)) );
    CUDA_SAFE_CALL( cudaMalloc((void**)&d_Sum, ACCUM_N * sizeof(float)) );
    CUT_SAFE_MALLOC( h_Sum = (float *)malloc(ACCUM_N * sizeof(float)) );

    //Copy input data from CPU
    CUDA_SAFE_CALL( cudaMemcpy(d_Data, plan->h_Data, plan->dataN * sizeof(float), cudaMemcpyHostToDevice) );

    //Perform GPU computations
    reduceKernel<<<BLOCK_N, THREAD_N>>>(d_Sum, d_Data, plan->dataN);
    CUT_CHECK_ERROR("reduceKernel() execution failed.\n");

    //Read back GPU results
    CUDA_SAFE_CALL( cudaMemcpy(h_Sum, d_Sum, ACCUM_N * sizeof(float), cudaMemcpyDeviceToHost) );

    //Finalize GPU reduction for current subvector
    sum = 0;
    for(i = 0; i < ACCUM_N; i++)
        sum += h_Sum[i];
    *(plan->h_Sum) = (float)sum;

    //Shut down this GPU
    free(h_Sum);
    CUDA_SAFE_CALL( cudaFree(d_Sum) );
    CUDA_SAFE_CALL( cudaFree(d_Data) );
    CUT_THREADEND;
}



////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int MAX_GPU_COUNT = 8;
const int        DATA_N = 1048576*32;



////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){
    //Solver config
    TGPUplan      plan[MAX_GPU_COUNT];
    //GPU reduction results
    float     h_SumGPU[MAX_GPU_COUNT];
    //OS thread ID
    CUTThread threadID[MAX_GPU_COUNT];

    float *h_Data;
    float sumGPU;
    double sumCPU, diff;

    int i, gpuBase, GPU_N;
    unsigned int hTimer;

    CUT_SAFE_CALL(cutCreateTimer(&hTimer));

    CUDA_SAFE_CALL(cudaGetDeviceCount(&GPU_N));
    if(GPU_N > MAX_GPU_COUNT) GPU_N = MAX_GPU_COUNT;
    printf("CUDA-capable device count: %i\n", GPU_N);

    printf("main(): generating input data...\n");
        h_Data = (float *)malloc(DATA_N * sizeof(float));
        for(i = 0; i < DATA_N; i++)
            h_Data[i] = (float)rand() / (float)RAND_MAX;

    //Subdividing input data across GPUs
    //Get data sizes for each GPU
    for(i = 0; i < GPU_N; i++)
        plan[i].dataN = DATA_N / GPU_N;
    //Take into account "odd" data sizes
    for(i = 0; i < DATA_N % GPU_N; i++)
        plan[i].dataN++;
    //Assign data ranges to GPUs
    gpuBase = 0;
    for(i = 0; i < GPU_N; i++){
        plan[i].device = i;
        plan[i].h_Data = h_Data + gpuBase;
        plan[i].h_Sum = h_SumGPU + i;
        gpuBase += plan[i].dataN;
    }

    //Start timing of GPU code
    printf("main(): waiting for GPU results...\n");
    CUT_SAFE_CALL(cutResetTimer(hTimer));
    CUT_SAFE_CALL(cutStartTimer(hTimer));
        for(i = 0; i < GPU_N; i++)
            threadID[i] = cutStartThread((CUT_THREADROUTINE)solverThread, (void *)(plan + i));
        cutWaitForThreads(threadID, GPU_N);
        sumGPU = 0;
        for(i = 0; i < GPU_N; i++)
            sumGPU += h_SumGPU[i];
    CUT_SAFE_CALL(cutStopTimer(hTimer));
    printf("GPU Processing time: %f (ms) \n", cutGetTimerValue(hTimer));

    printf("Checking the results...\n");
    CUT_SAFE_CALL(cutResetTimer(hTimer));
    CUT_SAFE_CALL(cutStartTimer(hTimer));
        sumCPU = 0;
        for(i = 0; i < DATA_N; i++)
            sumCPU += h_Data[i];
    CUT_SAFE_CALL(cutStopTimer(hTimer));
    printf("CPU Processing time: %f (ms) \n", cutGetTimerValue(hTimer));

    diff = fabs(sumCPU - sumGPU) / fabs(sumCPU);
    printf("GPU sum: %f; CPU sum: %f\n", sumGPU, sumCPU);
    printf("Relative difference: %E \n", diff);
    printf((diff < 1e-6) ? "TEST PASSED\n" : "TEST FAILED\n");

    printf("Shutting down...\n");
        CUT_SAFE_CALL(cutDeleteTimer(hTimer));
        free(h_Data);

    CUT_EXIT(argc, argv);
}
