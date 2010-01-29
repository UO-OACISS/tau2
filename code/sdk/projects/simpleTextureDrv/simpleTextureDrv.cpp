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
* This sample demonstrates how use texture fetches in CUDA
*
* This sample takes an input PGM image (image_filename) and generates 
* an output PGM image (image_filename_out).  This CUDA kernel performs
* a simple 2D transform (rotation) on the texture coordinates (u,v).
* The results between simpleTexture and simpleTextureDrv are identical.
* The main difference is the implementation.  simpleTextureDrv makes calls
* to the CUDA driver API and demonstrates how to use cuModuleLoad to load 
* the CUDA binary (*.cubin) kernel just prior to kernel launch.
* 
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

char *image_filename = "lena_bw.pgm";
char *ref_filename   = "ref_rotated.pgm";
float angle = 0.5f;    // angle to rotate image by (in radians)

#define MIN_EPSILON_ERROR 5e-3f

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

extern "C"
void computeGold( float* reference, float* idata, const unsigned int len);

static CUresult initCUDA(int argc, char**argv, CUfunction*);

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
main( int argc, char** argv) 
{
    runTest( argc, argv);

    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    // initialize CUDA
    CUfunction transform = NULL;
    CU_SAFE_CALL(initCUDA(argc, argv, &transform));

    // load image from disk
    float* h_data = NULL;
    unsigned int width, height;
    char* image_path = cutFindFilePath(image_filename, argv[0]);
    if (image_path == 0)
        exit(EXIT_FAILURE);
    CUT_SAFE_CALL( cutLoadPGMf(image_path, &h_data, &width, &height));

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", image_filename, width, height);

    // load reference image from image (output)
    float *h_data_ref = (float*) malloc(size);
    char* ref_path = cutFindFilePath(ref_filename, argv[0]);
    if (ref_path == 0) {
        printf("Unable to find reference file %s\n", ref_filename);
        exit(EXIT_FAILURE);
    }
    CUT_SAFE_CALL( cutLoadPGMf(ref_path, &h_data_ref, &width, &height));

    // allocate device memory for result
    CUdeviceptr d_data = (CUdeviceptr)NULL;
    CU_SAFE_CALL( cuMemAlloc( &d_data, size));

    // allocate array and copy image data
    CUarray cu_array;
    CUDA_ARRAY_DESCRIPTOR desc;
    desc.Format = CU_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = width;
    desc.Height = height;
    CU_SAFE_CALL( cuArrayCreate( &cu_array, &desc ));
	CUDA_MEMCPY2D copyParam;
	memset(&copyParam, 0, sizeof(copyParam));
	copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	copyParam.dstArray = cu_array;
	copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
	copyParam.srcHost = h_data;
	copyParam.srcPitch = width * sizeof(float);
	copyParam.WidthInBytes = copyParam.srcPitch;
	copyParam.Height = height;
    CU_SAFE_CALL(cuMemcpy2D(&copyParam));

    // set texture parameters
    CUtexref cu_texref;
    CU_SAFE_CALL(cuModuleGetTexRef(&cu_texref, cuModule, "tex"));
    CU_SAFE_CALL(cuTexRefSetArray(cu_texref, cu_array, CU_TRSA_OVERRIDE_FORMAT));
    CU_SAFE_CALL(cuTexRefSetAddressMode(cu_texref, 0, CU_TR_ADDRESS_MODE_WRAP));
    CU_SAFE_CALL(cuTexRefSetAddressMode(cu_texref, 1, CU_TR_ADDRESS_MODE_WRAP));
    CU_SAFE_CALL(cuTexRefSetFilterMode(cu_texref, CU_TR_FILTER_MODE_LINEAR));
    CU_SAFE_CALL(cuTexRefSetFlags(cu_texref, CU_TRSF_NORMALIZED_COORDINATES));
    CU_SAFE_CALL(cuTexRefSetFormat(cu_texref, CU_AD_FORMAT_FLOAT, 1));

	int block_size = 8;
    CU_SAFE_CALL(cuFuncSetBlockShape( transform, block_size, block_size, 1 ));
	int offset = 0;
    CU_SAFE_CALL(cuParamSeti(transform, offset, d_data)); offset += sizeof(d_data);
    CU_SAFE_CALL(cuParamSeti(transform, offset, width));  offset += sizeof(width);
    CU_SAFE_CALL(cuParamSeti(transform, offset, height)); offset += sizeof(height);
    CU_SAFE_CALL(cuParamSetf(transform, offset, angle));  offset += sizeof(angle);
    CU_SAFE_CALL(cuParamSetSize(transform, offset));
    CU_SAFE_CALL(cuParamSetTexRef(transform, CU_PARAM_TR_DEFAULT, cu_texref));

    // warmup
    CU_SAFE_CALL(cuLaunchGrid( transform, width / block_size, height / block_size ));

    CU_SAFE_CALL( cuCtxSynchronize() );
    unsigned int timer = 0;
    CUT_SAFE_CALL( cutCreateTimer( &timer));
    CUT_SAFE_CALL( cutStartTimer( timer));

    // execute the kernel
    CU_SAFE_CALL(cuLaunchGrid( transform, width / block_size, height / block_size ));

    CU_SAFE_CALL( cuCtxSynchronize() );
    CUT_SAFE_CALL( cutStopTimer( timer));
    printf("Processing time: %f (ms)\n", cutGetTimerValue( timer));
    printf("%.2f Mpixels/sec\n", (width*height / (cutGetTimerValue( timer) / 1000.0f)) / 1e6);
    CUT_SAFE_CALL( cutDeleteTimer( timer));

    // allocate mem for the result on host side
    float* h_odata = (float*) malloc( size);
    // copy result from device to host
    CU_SAFE_CALL( cuMemcpyDtoH( h_odata, d_data, size) );

    // write result to file
    char output_filename[1024];
    strcpy(output_filename, image_path);
    strcpy(output_filename + strlen(image_path) - 4, "_out.pgm");
    CUT_SAFE_CALL( cutSavePGMf( output_filename, h_odata, width, height));
    printf("Wrote '%s'\n", output_filename);

    // write regression file if necessary
    if( cutCheckCmdLineFlag( argc, (const char**) argv, "regression")) 
    {
        // write file for regression test
        CUT_SAFE_CALL( cutWriteFilef( "./data/regression.dat", h_odata, width*height, 0.0));
    } 
    else 
    {
        // We need to reload the data from disk, because it is inverted upon output
        CUT_SAFE_CALL( cutLoadPGMf(output_filename, &h_odata, &width, &height));

        printf("Comparing files\n");
        printf("\toutput:    <%s>\n", output_filename);
        printf("\treference: <%s>\n", ref_path);
        CUTBoolean res = cutComparefe( h_odata, h_data_ref, width*height, MIN_EPSILON_ERROR );
        printf( "Test %s\n", (1 == res) ? "PASSED" : "FAILED");
    }

    // cleanup memory
    CU_SAFE_CALL(cuMemFree(d_data));
    CU_SAFE_CALL(cuArrayDestroy(cu_array));
    free(h_data);
    free(h_data_ref);
    free(h_odata);
    cutFree(image_path);
    cutFree(ref_path);

    CU_SAFE_CALL_NO_SYNC(cuCtxDetach(cuContext));

    // If we are doing the QAtest, we quite without prompting
    if( cutCheckCmdLineFlag( argc, (const char**) argv, "qatest") ||
        cutCheckCmdLineFlag( argc, (const char**) argv, "noprompt"))
    {
        exit(0);
    }

    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! This initializes CUDA, and loads the *.cubin CUDA module containing the
//! kernel function binary.  After the module is loaded, cuModuleGetFunction 
//! retrieves the CUDA function pointer "cuFunction" 
////////////////////////////////////////////////////////////////////////////////
static CUresult
initCUDA(int argc, char **argv, CUfunction* transform)
{
    CUfunction cuFunction = 0;
    char* module_path;

    CUT_DEVICE_INIT_DRV(cuDevice, argc, argv);

    CUresult status = cuCtxCreate( &cuContext, 0, cuDevice );
    if ( CUDA_SUCCESS != status )
        goto Error;

    module_path = cutFindFilePath("simpleTexture_kernel.cubin", argv[0]);
    if (module_path == 0) {
        status = CUDA_ERROR_NOT_FOUND;
        goto Error;
    }

    status = cuModuleLoad(&cuModule, module_path);
    cutFree(module_path);
    if ( CUDA_SUCCESS != status ) {
        goto Error;
    }

    status = cuModuleGetFunction( &cuFunction, cuModule, "transformKernel" );
    if ( CUDA_SUCCESS != status )
        goto Error;
    *transform = cuFunction;
    return CUDA_SUCCESS;
Error:
    cuCtxDetach(cuContext);
    return status;
}
