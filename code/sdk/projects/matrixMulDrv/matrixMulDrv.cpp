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

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication using the CUDA driver API.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, CUDA
#include <cuda.h>

// includes, project
#include <cutil.h>
#include "matrixMul.h"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

static CUresult initCUDA(int argc, char **argv, CUfunction *pMatrixMul );

////////////////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////////////////
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
    runTest(argc, argv);

    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char** argv)
{
    // initialize CUDA
    CUfunction matrixMul = NULL;
    CU_SAFE_CALL(initCUDA(argc, argv, &matrixMul ));

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // allocate device memory
    CUdeviceptr d_A;
    CU_SAFE_CALL(cuMemAlloc( &d_A, mem_size_A ));
    CUdeviceptr d_B;
    CU_SAFE_CALL(cuMemAlloc( &d_B, mem_size_B )); 

    // copy host memory to device
    CU_SAFE_CALL(cuMemcpyHtoD( d_A, h_A, mem_size_A ));
    CU_SAFE_CALL(cuMemcpyHtoD( d_B, h_B, mem_size_B ));

    // allocate device memory for result
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    CUdeviceptr d_C;
    CU_SAFE_CALL(cuMemAlloc(&d_C, mem_size_C));
    
    // create and start timer
    unsigned int timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
    CUT_SAFE_CALL(cutStartTimer(timer));

    // setup execution parameters
    CU_SAFE_CALL(cuFuncSetBlockShape( matrixMul, BLOCK_SIZE, BLOCK_SIZE, 1 ));
    CU_SAFE_CALL(cuFuncSetSharedSize( matrixMul, 2*BLOCK_SIZE*BLOCK_SIZE*sizeof(float) ) );
    CU_SAFE_CALL(cuParamSeti( matrixMul, 0,  d_C ));
    CU_SAFE_CALL(cuParamSeti( matrixMul, 4,  d_A ));
    CU_SAFE_CALL(cuParamSeti( matrixMul, 8,  d_B ));
    CU_SAFE_CALL(cuParamSeti( matrixMul, 12, WA ));
    CU_SAFE_CALL(cuParamSeti( matrixMul, 16, WB ));
    CU_SAFE_CALL(cuParamSetSize( matrixMul, 20 ));
    CU_SAFE_CALL(cuLaunchGrid( matrixMul, WC / BLOCK_SIZE, HC / BLOCK_SIZE ));

    // allocate mem for the result on host side
    float* h_C = (float*) malloc(mem_size_C);

    // copy result from device to host
    CU_SAFE_CALL(cuMemcpyDtoH((void *) h_C, d_C, mem_size_C) );

    // stop and destroy timer
    CUT_SAFE_CALL(cutStopTimer(timer));
    printf("Processing time: %f (ms)\n", cutGetTimerValue(timer));
    CUT_SAFE_CALL(cutDeleteTimer(timer));

    // compute reference solution
    float* reference = (float*) malloc(mem_size_C);
    computeGold(reference, h_A, h_B, HA, WA, WB);

    // check result
    CUTBoolean res = cutCompareL2fe(reference, h_C, size_C, 1e-6f);
    printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    CU_SAFE_CALL(cuMemFree(d_A));
    CU_SAFE_CALL(cuMemFree(d_B));
    CU_SAFE_CALL(cuMemFree(d_C));
    CU_SAFE_CALL_NO_SYNC(cuCtxDetach(cuContext));
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

static CUresult
initCUDA(int argc, char **argv, CUfunction *pMatrixMul )
{
    CUfunction cuFunction = 0;
    char* module_path;

    CUT_DEVICE_INIT_DRV(cuDevice, argc, argv);

    CUresult status = cuCtxCreate( &cuContext, 0, cuDevice );
    if ( CUDA_SUCCESS != status )
        goto Error;

    module_path = cutFindFilePath("matrixMul_kernel.cubin", argv[0]);
    if (module_path == 0) {
        status = CUDA_ERROR_NOT_FOUND;
        goto Error;
    }

    status = cuModuleLoad(&cuModule, module_path);
    cutFree(module_path);
    if ( CUDA_SUCCESS != status ) {
        goto Error;
    }

    status = cuModuleGetFunction( &cuFunction, cuModule, "matrixMul" );
    if ( CUDA_SUCCESS != status )
        goto Error;
    *pMatrixMul = cuFunction;
    return CUDA_SUCCESS;
Error:
    cuCtxDetach(cuContext);
    return status;
}


