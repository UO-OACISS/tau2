/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

#define WIDTH  1024
#define HEIGHT 1024
#define NUM    (WIDTH * HEIGHT)

#define THREADS_PER_BLOCK_X 16
#define THREADS_PER_BLOCK_Y 16

__global__ void 
vectoradd_float(float* a, const float* b, const float* c, int n)
{
    int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (i < n) {
        a[i] = b[i] + c[i];
    }
}

int main() {

    int numDevices = 0;
    HIP_ASSERT(hipGetDeviceCount(&numDevices));

    std::cout << "Number of GPUs: " << numDevices << std::endl;

    float* hostA = (float*)malloc(NUM * sizeof(float));
    float* hostB = (float*)malloc(NUM * sizeof(float));
    float* hostC = (float*)malloc(NUM * sizeof(float));

    for (int i = 0; i < NUM; i++) {
        hostB[i] = (float)i;
        hostC[i] = (float)i * 100.0f;
    }

    int chunkSize = NUM / numDevices;

    // Loop over all GPUs
    for (int dev = 0; dev < numDevices; dev++) {

        HIP_ASSERT(hipSetDevice(dev));

        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, dev);
        std::cout << "Using GPU " << dev << ": " << prop.name << std::endl;

        int offset = dev * chunkSize;
        int size = (dev == numDevices - 1) ? (NUM - offset) : chunkSize;

        float *deviceA, *deviceB, *deviceC;

        HIP_ASSERT(hipMalloc(&deviceA, size * sizeof(float)));
        HIP_ASSERT(hipMalloc(&deviceB, size * sizeof(float)));
        HIP_ASSERT(hipMalloc(&deviceC, size * sizeof(float)));

        HIP_ASSERT(hipMemcpy(deviceB, hostB + offset, size * sizeof(float), hipMemcpyHostToDevice));
        HIP_ASSERT(hipMemcpy(deviceC, hostC + offset, size * sizeof(float), hipMemcpyHostToDevice));

        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        hipLaunchKernelGGL(vectoradd_float,
                           dim3(blocks),
                           dim3(threads),
                           0, 0,
                           deviceA, deviceB, deviceC, size);

        HIP_ASSERT(hipMemcpy(hostA + offset, deviceA, size * sizeof(float), hipMemcpyDeviceToHost));

        HIP_ASSERT(hipFree(deviceA));
        HIP_ASSERT(hipFree(deviceB));
        HIP_ASSERT(hipFree(deviceC));
    }

    // Verify results
    int errors = 0;
    for (int i = 0; i < NUM; i++) {
        if (hostA[i] != (hostB[i] + hostC[i])) {
            errors++;
        }
    }

    if (errors)
        printf("FAILED: %d errors\n", errors);
    else
        printf("PASSED!\n");

    free(hostA);
    free(hostB);
    free(hostC);

    return errors;
}
