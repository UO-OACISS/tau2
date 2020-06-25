/*
 * Copyright 2015 NVIDIA Corporation. All rights reserved
 *
 * Sample OpenACC app computing a saxpy kernel.
 * Data collection of OpenACC records via CUPTI is implemented
 * in a shared library attached at runtime.
 */

#include <stdio.h>
#include <openacc.h>
#include <assert.h>

// Helper function

static void setClear(const int n, float *a)
{
    int i;
    for (i = 0; i < n; ++i) {
        a[i] = 0.0;
    }
}

// OpenACC kernels

static void openaccKernel(const int n, const float a, float *x, float *y)
{
    int i;
#pragma acc kernels
    for (i = 0; i < n; ++i)
        y[i] = a*x[i];
}

static void initVec(const int n, const float mult, float *x)
{
    int i;
    // CUPTI OpenACC only supports NVIDIA devices
#pragma acc kernels
#if (!defined(HOST_ARCH_PPC))
    assert(acc_on_device(acc_device_nvidia));
#endif
    for (i = 0; i < n; ++i)
        x[i] = mult*i;
}

/* declare the CUDA function in the other file */
void do_cuda(void);

// program main

int main(int argc, char **argv)
{
    int N = 32000;

    float *x = new float[N];
    float *y = new float[N];

    // initialize data
    initVec(N, 0.5, x);
    setClear(N, y);

    printf("Running openacc kernel...\n");
    // run saxpy kernel
    openaccKernel(N, 2.0f, x, y);

    // cleanup
    delete[] x;
    delete[] y;

    do_cuda();

    return 0;
}

