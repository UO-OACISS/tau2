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

/******************************************************************************
*
*   Module: threadMigration.cpp
*
*   Description:
*     Simple sample demonstrating multi-GPU/multithread functionality using 
*     the CUDA Context Management API.  This API allows the a CUDA context to be
*     associated with a CPU process.  CUDA Contexts have a one-to-one correspondence 
*     with host threads.  A host thread may have only one device context current 
*     at a time.
*
*    Refer to the CUDA programming guide 4.5.3.3 on Context Management
*
******************************************************************************/

#define MAXTHREADS  256
#define NUM_INTS    32

#ifdef _WIN32
  // Windows threads use different data structures
  #include <windows.h>
  DWORD rgdwThreadIds[MAXTHREADS];
  HANDLE rghThreads[MAXTHREADS];
  CRITICAL_SECTION g_cs;

  #define ENTERCRITICALSECTION EnterCriticalSection(&g_cs);
  #define LEAVECRITICALSECTION LeaveCriticalSection(&g_cs);
#else

  // Includes POSIX thread headers for Linux thread support
  #include <pthread.h>
  #include <stdint.h>
  pthread_t rghThreads[MAXTHREADS];
  pthread_mutex_t g_mutex;

  #define ENTERCRITICALSECTION pthread_mutex_lock(&g_mutex);
  #define LEAVECRITICALSECTION pthread_mutex_unlock(&g_mutex);
#endif

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cutil.h>

int NumThreads;
int ThreadLaunchCount;

typedef struct _CUDAContext_st {
    CUcontext   hcuContext;
    CUmodule    hcuModule;
    CUfunction  hcuFunction;
    CUdeviceptr dptr;
    int        	deviceID;
    int        	threadNum;
} CUDAContext;

CUDAContext g_ThreadParams[MAXTHREADS];

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
int runTest(int argc, char** argv);

#define CLEANUP_ON_ERROR(dptr, hcuModule, hcuContext, status) \
    if ( dptr ) cuMemFree( dptr ); \
    if ( hcuModule ) cuModuleUnload( hcuModule ); \
    if ( hcuContext ) cuCtxDetach( hcuContext ); \
    return status;

#define THREAD_QUIT \
    printf("Error\n"); \
    return 0;

// This sample uses the Driver API interface.  The CUDA context needs
// to be setup and the CUDA module (CUBIN) is built by NVCC
CUresult
InitCUDAContext( CUDAContext *pContext, CUdevice hcuDevice, int deviceID, char *argv )
{
    CUcontext  hcuContext  = 0;
    CUmodule   hcuModule   = 0;
    CUfunction hcuFunction = 0;
    CUdeviceptr dptr       = 0;
    CUdevprop devProps;

    // cuCtxCreate: Function works on floating contexts and current context
    CUresult status = cuCtxCreate( &hcuContext, 0, hcuDevice );
    if ( CUDA_SUCCESS != status ) {
        fprintf( stderr, "cuCtxCreate for <ThreadNum=%d> failed %d\n", 
		        pContext->threadNum, status );
        CLEANUP_ON_ERROR(dptr, hcuModule, hcuContext, status);
    }
    status = CUDA_ERROR_INVALID_IMAGE;

    if ( CUDA_SUCCESS != cuDeviceGetProperties( &devProps, hcuDevice ) ) {
        printf("cuDeviceGetProperties failed!\n");
        CLEANUP_ON_ERROR(dptr, hcuModule, hcuContext, status);
    }

    char *module_path = cutFindFilePath("threadMigration.cubin", argv);
    if (module_path == NULL) {
        fprintf( stderr, "cutFindFilePath() unable to find threadMigration.cubin\n" );
        CLEANUP_ON_ERROR(dptr, hcuModule, hcuContext, status);
    }
    status = cuModuleLoad(&hcuModule, module_path);
    if ( CUDA_SUCCESS != status ) {
        fprintf( stderr, "cuModuleLoad failed %d\n", status );
        CLEANUP_ON_ERROR(dptr, hcuModule, hcuContext, status);
    }

    status = cuModuleGetFunction( &hcuFunction, hcuModule, "kernelFunction" );
    if ( CUDA_SUCCESS != status ) {
        fprintf( stderr, "cuModuleGetFunction failed %d\n", status );
        CLEANUP_ON_ERROR(dptr, hcuModule, hcuContext, status);
    }

    status = cuMemAlloc( &dptr, NUM_INTS*sizeof(int) );
    if ( CUDA_SUCCESS != status ) {
        fprintf( stderr, "cuMemAlloc failed %d\n", status );
        CLEANUP_ON_ERROR(dptr, hcuModule, hcuContext, status);
    }
    
    // Here we must release the CUDA context from the thread context 
    status = cuCtxPopCurrent( NULL );
    if ( CUDA_SUCCESS != status ) {
        fprintf( stderr, "cuCtxPopCurrent failed %d\n", status );
        CLEANUP_ON_ERROR(dptr, hcuModule, hcuContext, status);
    }

    pContext->hcuContext  = hcuContext;
    pContext->hcuModule   = hcuModule;
    pContext->hcuFunction = hcuFunction;
    pContext->dptr        = dptr;
    pContext->deviceID    = deviceID;
	
    return CUDA_SUCCESS;
}



// ThreadProc launches the CUDA kernel on a CUDA context.  
// We have more than one thread that talks to a CUDA context
#ifdef _WIN32
  DWORD WINAPI ThreadProc(CUDAContext *pParams)
#else
  void* ThreadProc(CUDAContext *pParams)
#endif
{
    int wrong = 0;
    int *pInt = 0;

    printf( "<CUDA Device=%d, ContextID=%p, ThreadNum=%d> ThreadProc() Launched...\n",
		pParams->deviceID, pParams->hcuContext, pParams->threadNum );

    // cuCtxPushCurrent: Attach the caller CUDA context to the thread context. 
    CUresult status = cuCtxPushCurrent( pParams->hcuContext );
    if ( CUDA_SUCCESS != status ) {
        THREAD_QUIT;
    }

    cuFuncSetBlockShape( pParams->hcuFunction, 32, 1, 1 );
    cuParamSeti( pParams->hcuFunction, 0, pParams->dptr );
    cuParamSetSize( pParams->hcuFunction, sizeof(int) );

    // cuLaunch: we kick off the CUDA "kernelFunction"
    status = cuLaunch( pParams->hcuFunction );
    if ( CUDA_SUCCESS != status ) {
        fprintf( stderr, "cuLaunch failed %d\n", status );
        THREAD_QUIT;
    }
    pInt = (int *) malloc(NUM_INTS*sizeof(int));
    if ( ! pInt )
        return 0;
    if ( CUDA_SUCCESS == cuMemcpyDtoH( pInt, pParams->dptr, NUM_INTS*sizeof(int) ) ) {
        for ( int i = 0; i < NUM_INTS; i++ ) {
            if ( pInt[i] != 32-i ) {
                printf("<CUDA Device=%d, ContextID=%p, ThreadNum=%d> error [%d]=%d!\n", 
                       pParams->deviceID, pParams->hcuContext, 
                       pParams->threadNum, i, pInt[i] );
                wrong++;
            }
        }
        ENTERCRITICALSECTION
        if ( ! wrong ) ThreadLaunchCount += 1;
        LEAVECRITICALSECTION
    }
    free( pInt );
    fflush( stdout );
    cuMemFree( pParams->dptr );

    // cuCtxPopCurrent: Detach the current CUDA context from the calling thread.
    cuCtxPopCurrent( NULL );

    printf( "<CUDA Device=%d, ContextID=%p, ThreadNum=%d> ThreadProc() Finished!\n", 
	    pParams->deviceID, pParams->hcuContext, pParams->threadNum );

    return 0;
}

int FinalErrorCheck(int ThreadIndex, int NumThreads, int cDevices)
{
    if ( ThreadLaunchCount != NumThreads*cDevices ) {
        printf( "<Expected=%d, Actual=%d> ThreadLaunchCounts(s)\n", 
                NumThreads*cDevices, ThreadLaunchCount );
        printf( "\nTest FAILED!\n" );
        return 1;
    }
    else {
        // destroy floating contexts while unattached to threads
        ThreadIndex = 0;
        for ( int iDevice = 0; iDevice < cDevices; iDevice++ ) {
           for ( int iThread = 0; iThread < NumThreads; iThread++, ThreadIndex++ ) {
              // cuCtxDestroy called on current context or a floating context
              if ( CUDA_SUCCESS != cuCtxDestroy( g_ThreadParams[ThreadIndex].hcuContext ) )
                 return 1;
           }
        }
        printf( "\nTest PASSED\n" );
        return 0;
    }
    return 0;
}

int 
main(int argc, char **argv)
{
	runTest(argc, argv);

    if( cutCheckCmdLineFlag( argc, (const char**) argv, "qatest") ||
        cutCheckCmdLineFlag( argc, (const char**) argv, "noprompt"))
    {
        exit(0);
    }

	CUT_EXIT(argc, argv);
}

int
runTest(int argc, char **argv)
{
    printf("threadMigration API test...\n" );
#ifdef _WIN32
    InitializeCriticalSection( &g_cs );
#else
    pthread_mutex_init(&g_mutex, NULL);
#endif
    if ( argc != 1 && argc != 2 ) {
        printf("Usage: \"threadMigration -n=<threads>\", <threads> ranges 1-15\n" );
        return 1;
    }
	// By default, we will launch 2 CUDA threads for each device
    NumThreads = 2;
    if ( argc == 2 ) {
        cutGetCmdLineArgumenti(argc, (const char**) argv, "n", &NumThreads);
        if ( NumThreads < 1 || NumThreads > 15 ) {
            printf("Usage: \"threadMigration -n=<threads>\", <threads> ranges 1-15\n" );
            return 1;
        }
    }

    int cDevices;
    int hcuDevice = 0;
    CUresult status;
    status = cuInit(0);
    if ( CUDA_SUCCESS != status )
        return 1;
    status = cuDeviceGetCount( &cDevices );
    if ( CUDA_SUCCESS != status )
        return 1;

	printf( "%d CUDA devices detected, %d Threads to launch\n\n", cDevices, NumThreads );
    if ( cDevices == 0 ) {
       return 1;
    }

    int ihThread = 0;
    int ThreadIndex = 0;
    for ( int iDevice = 0; iDevice < cDevices; iDevice++ ) {
        char szName[256];
        status = cuDeviceGet( &hcuDevice, iDevice );
        if ( CUDA_SUCCESS != status )
            return 1;

        status = cuDeviceGetName( szName, 256, hcuDevice );
        if ( CUDA_SUCCESS != status )
            return 1;

        CUdevprop devProps;
        if ( CUDA_SUCCESS == cuDeviceGetProperties( &devProps, hcuDevice ) ) {
            printf("Device %d: %s\n", iDevice, szName );
            printf("\tsharedMemPerBlock: %d\n", devProps.sharedMemPerBlock );
            printf("\tconstantMemory   : %d\n", devProps.totalConstantMemory );
            printf("\tregsPerBlock     : %d\n", devProps.regsPerBlock );
            printf("\tclockRate        : %d\n", devProps.clockRate );
        }

        for ( int iThread = 0; iThread < NumThreads; iThread++, ihThread++ ) {
            g_ThreadParams[ThreadIndex].threadNum = iThread;

            if ( CUDA_SUCCESS != InitCUDAContext( &g_ThreadParams[ThreadIndex], hcuDevice, iDevice, argv[0] ) )  
            {
               return FinalErrorCheck(ThreadIndex, NumThreads, cDevices);
            }
            else 
            {
	// Launch (NumThreads) for each CUDA context
#ifdef _WIN32        
              rghThreads[ThreadIndex] = CreateThread( NULL, 0, (LPTHREAD_START_ROUTINE) ThreadProc, 
                                                      &g_ThreadParams[ThreadIndex], 0, &rgdwThreadIds[ThreadIndex] );
#else	// Assume we are running linux
              pthread_create(&rghThreads[ThreadIndex], NULL, (void *(*)(void*)) ThreadProc, &g_ThreadParams[ThreadIndex]);
#endif
              ThreadIndex += 1;
            }
        }
    }

    // Wait until all workers are done
#ifdef _WIN32
     WaitForMultipleObjects(ThreadIndex, rghThreads, TRUE, INFINITE );
#else
     for ( int i = 0; i < ThreadIndex; i++ )
        pthread_join(rghThreads[i], NULL);
#endif

    return FinalErrorCheck(ThreadIndex, NumThreads, cDevices);
}
