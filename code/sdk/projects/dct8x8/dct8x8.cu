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

/**
**************************************************************************
* \file dct8x8.cu
* \brief Contains entry point, wrappers to host and device code and benchmark.
*
* This sample implements forward and inverse Discrete Cosine Transform to blocks
* of image pixels (of 8x8 size), as in JPEG standard. The typical work flow is as 
* follows:
* 1. Run CPU version (Host code) and measure execution time;
* 2. Run CUDA version (Device code) and measure execution time;
* 3. Output execution timings and calculate CUDA speedup.
*/

#include "Common.h"


/**
*  The number of DCT kernel calls
*/
#ifdef __DEVICE_EMULATION__
#define BENCHMARK_SIZE	1
#else
#define BENCHMARK_SIZE	10
#endif


/**
*  Texture reference that is passed through this global variable into device code.
*  This is done because any conventional passing through argument list way results 
*  in compiler internal error. 2008.03.11
*/
texture<float, 2, cudaReadModeElementType> TexSrc;


// includes kernels
#include "dct8x8_kernel1.cu"
#include "dct8x8_kernel2.cu"


/**
**************************************************************************
*  Wrapper function for 1st gold version of DCT, quantization and IDCT implementations
*
* \param ImgSrc			[IN] - Source byte image plane
* \param ImgDst			[IN] - Quantized result byte image plane
* \param Stride			[IN] - Stride for both source and result planes
* \param Size			[IN] - Size of both planes
*  
* \return Execution time in milliseconds
*/
float WrapperGold1(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size)
{
	//allocate float buffers for DCT and other data
	int StrideF;
	float *ImgF1 = MallocPlaneFloat(Size.width, Size.height, &StrideF);
	float *ImgF2 = MallocPlaneFloat(Size.width, Size.height, &StrideF);

	//convert source image to float representation
	CopyByte2Float(ImgSrc, Stride, ImgF1, StrideF, Size);
	AddFloatPlane(-128.0f, ImgF1, StrideF, Size);

	//create and start CUDA timer
	unsigned int timerGold = 0;
	CUT_SAFE_CALL(cutCreateTimer(&timerGold));
	CUT_SAFE_CALL(cutResetTimer(timerGold));

	//perform block-wise DCT processing and benchmarking
	for (int i=0; i<BENCHMARK_SIZE; i++)
	{
		CUT_SAFE_CALL(cutStartTimer(timerGold));
		computeDCT8x8Gold1(ImgF1, ImgF2, StrideF, Size);
		CUT_SAFE_CALL(cutStopTimer(timerGold));
	}

	//stop and destroy CUDA timer
	float TimerGoldSpan = cutGetAverageTimerValue(timerGold);
	CUT_SAFE_CALL(cutDeleteTimer(timerGold));

	//perform quantization
	quantizeGold(ImgF2, StrideF, Size);

	//perform block-wise IDCT processing
	computeIDCT8x8Gold1(ImgF2, ImgF1, StrideF, Size);

	//convert image back to byte representation
	AddFloatPlane(128.0f, ImgF1, StrideF, Size);
	CopyFloat2Byte(ImgF1, StrideF, ImgDst, Stride, Size);

	//free float buffers
	FreePlane(ImgF1);
	FreePlane(ImgF2);

	//return time taken by the operation
	return TimerGoldSpan;
}


/**
**************************************************************************
*  Wrapper function for 2nd gold version of DCT, quantization and IDCT implementations
*
* \param ImgSrc			[IN] - Source byte image plane
* \param ImgDst			[IN] - Quantized result byte image plane
* \param Stride			[IN] - Stride for both source and result planes
* \param Size			[IN] - Size of both planes
*  
* \return Execution time in milliseconds
*/
float WrapperGold2(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size)
{
	//allocate float buffers for DCT and other data
	int StrideF;
	float *ImgF1 = MallocPlaneFloat(Size.width, Size.height, &StrideF);
	float *ImgF2 = MallocPlaneFloat(Size.width, Size.height, &StrideF);

	//convert source image to float representation
	CopyByte2Float(ImgSrc, Stride, ImgF1, StrideF, Size);
	AddFloatPlane(-128.0f, ImgF1, StrideF, Size);

	//create and start CUDA timer
	unsigned int timerGold = 0;
	CUT_SAFE_CALL(cutCreateTimer(&timerGold));
	CUT_SAFE_CALL(cutResetTimer(timerGold));

	//perform block-wise DCT processing and benchmarking
	for (int i=0; i<BENCHMARK_SIZE; i++)
	{
		CUT_SAFE_CALL(cutStartTimer(timerGold));
		computeDCT8x8Gold2(ImgF1, ImgF2, StrideF, Size);
		CUT_SAFE_CALL(cutStopTimer(timerGold));
	}

	//stop and destroy CUDA timer
	float TimerGoldSpan = cutGetAverageTimerValue(timerGold);
	CUT_SAFE_CALL(cutDeleteTimer(timerGold));

	//perform quantization
	quantizeGold(ImgF2, StrideF, Size);

	//perform block-wise IDCT processing
	computeIDCT8x8Gold2(ImgF2, ImgF1, StrideF, Size);

	//convert image back to byte representation
	AddFloatPlane(128.0f, ImgF1, StrideF, Size);
	CopyFloat2Byte(ImgF1, StrideF, ImgDst, Stride, Size);

	//free float buffers
	FreePlane(ImgF1);
	FreePlane(ImgF2);

	//return time taken by the operation
	return TimerGoldSpan;
}


/**
**************************************************************************
*  Wrapper function for 1st CUDA version of DCT, quantization and IDCT implementations
*
* \param ImgSrc			[IN] - Source byte image plane
* \param ImgDst			[IN] - Quantized result byte image plane
* \param Stride			[IN] - Stride for both source and result planes
* \param Size			[IN] - Size of both planes
*  
* \return Execution time in milliseconds
*/
float WrapperCUDA1(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size)
{
	//prepare channel format descriptor for passing texture into kernels
	cudaChannelFormatDesc floattex = cudaCreateChannelDesc<float>();

	//allocate device memory
	cudaArray *Src;
	float *Dst;
	size_t DstStride;
	CUDA_SAFE_CALL(cudaMallocArray(&Src, &floattex, Size.width, Size.height));
	CUDA_SAFE_CALL(cudaMallocPitch((void **)(&Dst), &DstStride, Size.width * sizeof(float), Size.height));
	DstStride /= sizeof(float);

	//convert source image to float representation
	int ImgSrcFStride;
	float *ImgSrcF = MallocPlaneFloat(Size.width, Size.height, &ImgSrcFStride);
	CopyByte2Float(ImgSrc, Stride, ImgSrcF, ImgSrcFStride, Size);
	AddFloatPlane(-128.0f, ImgSrcF, ImgSrcFStride, Size);

	//copy from host memory to device
	CUDA_SAFE_CALL(cudaMemcpy2DToArray(Src, 0, 0,
									   ImgSrcF, ImgSrcFStride * sizeof(float), 
									   Size.width * sizeof(float), Size.height,
									   cudaMemcpyHostToDevice) );

	//setup execution parameters
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

	//create and start CUDA timer
	unsigned int timerCUDA = 0;
	CUT_SAFE_CALL(cutCreateTimer(&timerCUDA));
	CUT_SAFE_CALL(cutResetTimer(timerCUDA));

	//execute DCT kernel and benchmark
	CUDA_SAFE_CALL(cudaBindTextureToArray(TexSrc, Src));
	for (int i=0; i<BENCHMARK_SIZE; i++)
	{
		CUT_SAFE_CALL(cutStartTimer(timerCUDA));
		CUDAkernel1DCT<<< grid, threads >>>(Dst, (int) DstStride, 0, 0);
		cudaThreadSynchronize();
		CUT_SAFE_CALL(cutStopTimer(timerCUDA));
	}
	CUDA_SAFE_CALL(cudaUnbindTexture(TexSrc));
	CUT_CHECK_ERROR("Kernel execution failed");

	// finalize CUDA timer
	float TimerCUDASpan = cutGetAverageTimerValue(timerCUDA);
	CUT_SAFE_CALL(cutDeleteTimer(timerCUDA));

	// execute Quantization kernel
	CUDAkernelQuantizationEmulator<<< grid, threads >>>(Dst, (int) DstStride);
	CUT_CHECK_ERROR("Kernel execution failed");

	//copy quantized coefficients from host memory to device array
	CUDA_SAFE_CALL(cudaMemcpy2DToArray(Src, 0, 0,
									   Dst, DstStride * sizeof(float),
									   Size.width * sizeof(float), Size.height,
									   cudaMemcpyDeviceToDevice) );

	// execute IDCT kernel
	CUDA_SAFE_CALL(cudaBindTextureToArray(TexSrc, Src));
	CUDAkernel1IDCT<<< grid, threads >>>(Dst, (int) DstStride, 0, 0);
	CUDA_SAFE_CALL(cudaUnbindTexture(TexSrc));
	CUT_CHECK_ERROR("Kernel execution failed");

	//copy quantized image block to host
	CUDA_SAFE_CALL(cudaMemcpy2D(ImgSrcF, ImgSrcFStride * sizeof(float), 
								Dst, DstStride * sizeof(float), 
								Size.width * sizeof(float), Size.height,
								cudaMemcpyDeviceToHost) );

	//convert image back to byte representation
	AddFloatPlane(128.0f, ImgSrcF, ImgSrcFStride, Size);
	CopyFloat2Byte(ImgSrcF, ImgSrcFStride, ImgDst, Stride, Size);

	//clean up memory
	CUDA_SAFE_CALL(cudaFreeArray(Src));
	CUDA_SAFE_CALL(cudaFree(Dst));
	FreePlane(ImgSrcF);

	//return time taken by the operation
	return TimerCUDASpan;
}


/**
**************************************************************************
*  Wrapper function for 2nd CUDA version of DCT, quantization and IDCT implementations
*
* \param ImgSrc			[IN] - Source byte image plane
* \param ImgDst			[IN] - Quantized result byte image plane
* \param Stride			[IN] - Stride for both source and result planes
* \param Size			[IN] - Size of both planes
*  
* \return Execution time in milliseconds
*/
float WrapperCUDA2(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size)
{
	//setup CUDA execution parameters
	int CudaDeviceNum;
	cudaDeviceProp DeviceProperties;
	cudaGetDevice(&CudaDeviceNum);
	cudaGetDeviceProperties(&DeviceProperties, CudaDeviceNum);
	const int NumThreadsInWarp = DeviceProperties.warpSize;
	const int NumWarpsInBlock = 4; //chosen according to Occupancy Calculator
	const int NumThreadsInBlock = NumThreadsInWarp * NumWarpsInBlock;
	const int NumThreadsInBlock8x8 = BLOCK_SIZE;	//Each thread processes single row of 8x8 block, then column (row-column 1D DCT or IDCT)
	const int NumBlocks8x8InBlock = NumThreadsInBlock / NumThreadsInBlock8x8;
	const int NumBlocks8x8InImage = Size.width * Size.height / BLOCK_SIZE2;
	const int NumBlocksInImage   = NumBlocks8x8InImage / NumBlocks8x8InBlock;
	const int NumEndianBlocks8x8 = NumBlocks8x8InImage % NumBlocks8x8InBlock;
	size_t SharedMemAmount = NumBlocks8x8InBlock * BLOCK_SIZE2 * sizeof(float);
	if (SharedMemAmount > DeviceProperties.sharedMemPerBlock)
	{
		return -1;
	}
	int WidthInBlocks = Size.width / BLOCK_SIZE;
	int HeightInBlocks = Size.height / BLOCK_SIZE;
	float InvWidthInBlocksF = 1.0f / WidthInBlocks;

	//Configure parameters that handle case when not all blocks8x8 will be processed by Full Warps kernel
	//In this case the unprocessed area is processed by the previous kernel
	int LastUnprocessedBlockNum = NumBlocksInImage * NumBlocks8x8InBlock;
	int LastUnprocOffsetYInBlocks = LastUnprocessedBlockNum / WidthInBlocks;
	int LastUnprocOffsetXInBlocks = LastUnprocessedBlockNum % WidthInBlocks;;
	if (HeightInBlocks - LastUnprocOffsetYInBlocks > 1)
	{
		//ensuring we overlap the whole unprocessed region
		LastUnprocOffsetXInBlocks = 0;
	}
	dim3 ThreadsEndianBlocks(BLOCK_SIZE, BLOCK_SIZE);
	dim3 GridEndianBlocks(WidthInBlocks - LastUnprocOffsetXInBlocks, HeightInBlocks - LastUnprocOffsetYInBlocks);

	//setup execution parameters for quantization
	dim3 ThreadsSmallBlocks(BLOCK_SIZE, BLOCK_SIZE);
	dim3 GridSmallBlocks(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

	//prepare channel format descriptor for passing texture into kernels
	cudaChannelFormatDesc floattex = cudaCreateChannelDesc<float>();

	//allocate device memory
	cudaArray *Src;
	float *Dst;
	size_t DstStride;
	CUDA_SAFE_CALL(cudaMallocArray(&Src, &floattex, Size.width, Size.height));
	CUDA_SAFE_CALL(cudaMallocPitch((void **)(&Dst), &DstStride, Size.width * sizeof(float), Size.height));
	DstStride /= sizeof(float);

	//convert source image to float representation
	int ImgSrcFStride;
	float *ImgSrcF = MallocPlaneFloat(Size.width, Size.height, &ImgSrcFStride);
	CopyByte2Float(ImgSrc, Stride, ImgSrcF, ImgSrcFStride, Size);
	AddFloatPlane(-128.0f, ImgSrcF, ImgSrcFStride, Size);

	//copy from host memory to device
	CUDA_SAFE_CALL(cudaMemcpy2DToArray( Src, 0, 0,
										ImgSrcF, ImgSrcFStride * sizeof(float), 
										Size.width * sizeof(float), Size.height,
										cudaMemcpyHostToDevice) );

	//create and start CUDA timer
	unsigned int timerCUDA = 0;
	CUT_SAFE_CALL(cutCreateTimer(&timerCUDA));
	CUT_SAFE_CALL(cutResetTimer(timerCUDA));

	//setup execution parameters
	dim3 ThreadsFullWarps(NumThreadsInBlock);
	dim3 GridFullWarps(NumBlocksInImage);

	//execute DCT kernel and benchmark
	CUDA_SAFE_CALL(cudaBindTextureToArray(TexSrc, Src));
	for (int i=0; i<BENCHMARK_SIZE; i++)
	{
		if (NumBlocksInImage > 0)
		{
			CUT_SAFE_CALL(cutStartTimer(timerCUDA));
			CUDAkernel2DCT<<< GridFullWarps, ThreadsFullWarps, SharedMemAmount >>>(Dst, (int) DstStride, NumBlocks8x8InBlock, WidthInBlocks, InvWidthInBlocksF);
			cudaThreadSynchronize();
			CUT_SAFE_CALL(cutStopTimer(timerCUDA));
		}
		//if the number of image blocks is multiple of 4 then this kernel call can be omitted
		if (NumEndianBlocks8x8 > 0)
		{
			CUDAkernel1DCT<<< GridEndianBlocks, ThreadsEndianBlocks >>>(Dst, (int) DstStride, LastUnprocOffsetXInBlocks, LastUnprocOffsetYInBlocks);
		}
	}
	CUDA_SAFE_CALL(cudaUnbindTexture(TexSrc));
	CUT_CHECK_ERROR("Kernel execution failed");

	// finalize CUDA timer
	float TimerCUDASpan = cutGetAverageTimerValue(timerCUDA);
	CUT_SAFE_CALL(cutDeleteTimer(timerCUDA));

	// execute Quantization kernel
	CUDAkernelQuantizationEmulator<<< GridSmallBlocks, ThreadsSmallBlocks >>>(Dst, (int) DstStride);
	CUT_CHECK_ERROR("Kernel execution failed");

	//copy quantized coefficients from host memory to device array
	CUDA_SAFE_CALL(cudaMemcpy2DToArray( Src, 0, 0,
										Dst, DstStride * sizeof(float),
										Size.width * sizeof(float), Size.height,
										cudaMemcpyDeviceToDevice) );

	// execute IDCT kernel
	CUDA_SAFE_CALL(cudaBindTextureToArray(TexSrc, Src));
	if (NumBlocksInImage > 0)
	{
		CUDAkernel2IDCT<<< GridFullWarps, ThreadsFullWarps, SharedMemAmount >>>(Dst, (int) DstStride, NumBlocks8x8InBlock, WidthInBlocks, InvWidthInBlocksF);
	}
	if (NumEndianBlocks8x8 > 0)
	{
		CUDAkernel1IDCT<<< GridEndianBlocks, ThreadsEndianBlocks >>>(Dst, (int) DstStride, LastUnprocOffsetXInBlocks, LastUnprocOffsetYInBlocks);
	}
	CUDA_SAFE_CALL(cudaUnbindTexture(TexSrc));
	CUT_CHECK_ERROR("Kernel execution failed");

	//copy quantized image block to host
	CUDA_SAFE_CALL(cudaMemcpy2D(ImgSrcF, ImgSrcFStride * sizeof(float), 
								Dst, DstStride * sizeof(float), 
								Size.width * sizeof(float), Size.height,
								cudaMemcpyDeviceToHost) );

	//convert image back to byte representation
	AddFloatPlane(128.0f, ImgSrcF, ImgSrcFStride, Size);
	CopyFloat2Byte(ImgSrcF, ImgSrcFStride, ImgDst, Stride, Size);

	//clean up memory
	CUDA_SAFE_CALL(cudaFreeArray(Src));
	CUDA_SAFE_CALL(cudaFree(Dst));
	FreePlane(ImgSrcF);

	//return time taken by the operation
	return TimerCUDASpan;
}


/**
**************************************************************************
*  Program entry point
*
* \param argc		[IN] - Number of command-line arguments
* \param argv		[IN] - Array of command-line arguments
*  
* \return Status code
*/
int main(int argc, char** argv)
{
	//
	// Sample initialization
	//

	//initialize CUDA
	CUT_DEVICE_INIT(argc, argv);

	//source and results image filenames
	char SampleImageFname[] = "barbara.bmp";
	char SampleImageFnameResGold1[] = "barbara_gold1.bmp";
	char SampleImageFnameResGold2[] = "barbara_gold2.bmp";
	char SampleImageFnameResCUDA1[] = "barbara_cuda1.bmp";
	char SampleImageFnameResCUDA2[] = "barbara_cuda2.bmp";

	char *pSampleImageFpath = cutFindFilePath(SampleImageFname, argv[0]);

	//preload image (acquire dimensions)
	int ImgWidth, ImgHeight;
	ROI ImgSize;
	int res = PreLoadBmp(pSampleImageFpath, &ImgWidth, &ImgHeight);
	ImgSize.width = ImgWidth;
	ImgSize.height = ImgHeight;

	//CONSOLE INFORMATION: saying hello to user
	printf("CUDA sample DCT/IDCT implementation\n");
	printf("===================================\n");
	printf("Loading test image: %s... ", SampleImageFname);

	if (res)
	{
		printf("\nError: Image file not found or invalid!\n");
		printf("Press ENTER to exit...\n");
		getchar();

		//finalize
		CUT_EXIT(argc, argv);
		return 1;
	}

	//check image dimensions are multiples of BLOCK_SIZE
	if (ImgWidth % BLOCK_SIZE != 0 || ImgHeight % BLOCK_SIZE != 0)
	{
		printf("\nError: Input image dimensions must be multiples of 8!\n");
		printf("Press ENTER to exit...\n");
		getchar();

		//finalize
		CUT_EXIT(argc, argv);
		return 1;
	}
	printf("[%d x %d]... ", ImgWidth, ImgHeight);

	//allocate image buffers
	int ImgStride;
	byte *ImgSrc = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
	byte *ImgDstGold1 = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
	byte *ImgDstGold2 = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
	byte *ImgDstCUDA1 = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
	byte *ImgDstCUDA2 = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);

	//load sample image
	LoadBmpAsGray(pSampleImageFpath, ImgStride, ImgSize, ImgSrc);

	//
	// RUNNING WRAPPERS
	//

	//compute Gold 1 version of DCT/quantization/IDCT
	printf("Success\nRunning Gold 1 (CPU) version... ");
	float TimeGold1 = WrapperGold1(ImgSrc, ImgDstGold1, ImgStride, ImgSize);

	//compute Gold 2 version of DCT/quantization/IDCT
	printf("Success\nRunning Gold 2 (CPU) version... ");
	float TimeGold2 = WrapperGold2(ImgSrc, ImgDstGold2, ImgStride, ImgSize);

	//compute CUDA 1 version of DCT/quantization/IDCT
	printf("Success\nRunning CUDA 1 (GPU) version... ");
	float TimeCUDA1 = WrapperCUDA1(ImgSrc, ImgDstCUDA1, ImgStride, ImgSize);

	//compute CUDA 2 version of DCT/quantization/IDCT
	printf("Success\nRunning CUDA 2 (GPU) version... ");
	float TimeCUDA2 = WrapperCUDA2(ImgSrc, ImgDstCUDA2, ImgStride, ImgSize);

	//
	// Execution statistics, result saving and validation
	//

	//dump result of Gold 1 processing
	printf("Success\nDumping result to %s... ", SampleImageFnameResGold1);
	DumpBmpAsGray(SampleImageFnameResGold1, ImgDstGold1, ImgStride, ImgSize);

	//dump result of Gold 2 processing
	printf("Success\nDumping result to %s... ", SampleImageFnameResGold2);
	DumpBmpAsGray(SampleImageFnameResGold2, ImgDstGold2, ImgStride, ImgSize);

	//dump result of CUDA 1 processing
	printf("Success\nDumping result to %s... ", SampleImageFnameResCUDA1);
	DumpBmpAsGray(SampleImageFnameResCUDA1, ImgDstCUDA1, ImgStride, ImgSize);

	//dump result of CUDA 2 processing
	printf("Success\nDumping result to %s... ", SampleImageFnameResCUDA2);
	DumpBmpAsGray(SampleImageFnameResCUDA2, ImgDstCUDA2, ImgStride, ImgSize);

	//print speed info
	printf("Success\n\n");

#ifdef __DEVICE_EMULATION__
	printf("Processing time : not relevant in CUDA emulation mode\n");
#else
	printf("Processing time (CUDA 1) : %f ms \n", TimeCUDA1);
	printf("Processing time (CUDA 2) : %f ms \n", TimeCUDA2);
#endif

	//calculate PSNR between each pair of images
	float PSNR_Src_DstGold1      = CalculatePSNR(ImgSrc, ImgDstGold1, ImgStride, ImgSize);
	float PSNR_Src_DstGold2      = CalculatePSNR(ImgSrc, ImgDstGold2, ImgStride, ImgSize);
	float PSNR_Src_DstCUDA1      = CalculatePSNR(ImgSrc, ImgDstCUDA1, ImgStride, ImgSize);
	float PSNR_Src_DstCUDA2      = CalculatePSNR(ImgSrc, ImgDstCUDA2, ImgStride, ImgSize);
	float PSNR_DstGold1_DstCUDA1 = CalculatePSNR(ImgDstGold1, ImgDstCUDA1, ImgStride, ImgSize);
	float PSNR_DstGold2_DstCUDA2 = CalculatePSNR(ImgDstGold2, ImgDstCUDA2, ImgStride, ImgSize);

	printf("PSNR Original    <---> CPU(Gold 1) : %f\n", PSNR_Src_DstGold1);
	printf("PSNR Original    <---> CPU(Gold 2) : %f\n", PSNR_Src_DstGold2);
	printf("PSNR Original    <---> GPU(CUDA 1) : %f\n", PSNR_Src_DstCUDA1);
	printf("PSNR Original    <---> GPU(CUDA 2) : %f\n", PSNR_Src_DstCUDA2);
	printf("PSNR CPU(Gold 1) <---> GPU(CUDA 1) : %f\n", PSNR_DstGold1_DstCUDA1);
	printf("PSNR CPU(Gold 2) <---> GPU(CUDA 2) : %f\n", PSNR_DstGold2_DstCUDA2);

	if (PSNR_DstGold1_DstCUDA1 > 45 && PSNR_DstGold2_DstCUDA2 > 45)
	{
		printf("\nTEST PASSED!\n");
	}
	else
	{
		printf("\nTEST FAILED! (CPU and GPU results differ too much)\n");
	}

	//
	// Finalization
	//

	//release byte planes
	FreePlane(ImgSrc);
	FreePlane(ImgDstGold1);
	FreePlane(ImgDstGold2);
	FreePlane(ImgDstCUDA1);
	FreePlane(ImgDstCUDA2);

	//finalize
	CUT_EXIT(argc, argv);
	return 0;
}
