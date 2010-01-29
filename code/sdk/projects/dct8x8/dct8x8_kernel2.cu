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
* \file dct8x8_kernel2.cu
* \brief Contains 2nd kernel implementations of DCT and IDCT routines, used in 
*        JPEG internal data processing. Device code.
*
* This code implements traditional approach to forward and inverse Discrete 
* Cosine Transform to blocks of image pixels (of 8x8 size), as in JPEG standard. 
* The data processing is done using floating point representation.
* The routine that performs quantization of coefficients can be found in 
* dct8x8_kernel1.cu file.
*/

#pragma once

#include "Common.h"


__constant__ float C_a = 1.387039845322148f; //!< a = (2^0.5) * cos(    pi / 16);  Used in forward and inverse DCT.  
__constant__ float C_b = 1.306562964876377f; //!< b = (2^0.5) * cos(    pi /  8);  Used in forward and inverse DCT.  
__constant__ float C_c = 1.175875602419359f; //!< c = (2^0.5) * cos(3 * pi / 16);  Used in forward and inverse DCT.  
__constant__ float C_d = 0.785694958387102f; //!< d = (2^0.5) * cos(5 * pi / 16);  Used in forward and inverse DCT.  
__constant__ float C_e = 0.541196100146197f; //!< e = (2^0.5) * cos(3 * pi /  8);  Used in forward and inverse DCT.  
__constant__ float C_f = 0.275899379282943f; //!< f = (2^0.5) * cos(7 * pi / 16);  Used in forward and inverse DCT.  


/**
*  Normalization constant that is used in forward and inverse DCT
*/
__constant__ float C_norm = 0.3535533905932737f; // 1 / (8^0.5)


/**
*  Block8x8 size. Used for quick GPU indexation without integer
*  arithmetic and any divisions.
*/
__constant__ float Block8x8SizeF = (1.0f * BLOCK_SIZE);


/**
*  Inverse of block8x8 size. Used for quick GPU indexation without integer
*  arithmetic and any divisions.
*/
__constant__ float InvBlock8x8SizeF = (1.0f / BLOCK_SIZE);


/**
*  Epsilon that is used for GPU indexation without integer arithmetic.
*  Problems with indexation may occur when loading images more than 
*  8*10^6 pixels width.
*/
__constant__ float Eps = 0.000001f;


/**
*  Declaration of pointer to array of floats with size specified at runtime from wrapper
*/
extern __shared__ float SharedBlocks[];


/**
**************************************************************************
*  Performs in-place DCT of vector of 8 elements.
*
* \param Vect0			[IN] - Pointer to the first element of vector
* \param Step			[IN] - Value to add to ptr to access other elements 
*  
* \return None
*/
__device__ void CUDAsubroutineInplaceDCTvector(float *Vect0, int Step)
{
	float *Vect1 = Vect0 + Step;
	float *Vect2 = Vect1 + Step;
	float *Vect3 = Vect2 + Step;
	float *Vect4 = Vect3 + Step;
	float *Vect5 = Vect4 + Step;
	float *Vect6 = Vect5 + Step;
	float *Vect7 = Vect6 + Step;

	float X07P = (*Vect0) + (*Vect7);
	float X16P = (*Vect1) + (*Vect6);
	float X25P = (*Vect2) + (*Vect5);
	float X34P = (*Vect3) + (*Vect4);

	float X07M = (*Vect0) - (*Vect7);
	float X61M = (*Vect6) - (*Vect1);
	float X25M = (*Vect2) - (*Vect5);
	float X43M = (*Vect4) - (*Vect3);

	float X07P34PP = X07P + X34P;
	float X07P34PM = X07P - X34P;
	float X16P25PP = X16P + X25P;
	float X16P25PM = X16P - X25P;

	(*Vect0) = C_norm * (X07P34PP + X16P25PP);
	(*Vect2) = C_norm * (C_b * X07P34PM + C_e * X16P25PM);
	(*Vect4) = C_norm * (X07P34PP - X16P25PP);
	(*Vect6) = C_norm * (C_e * X07P34PM - C_b * X16P25PM);

	(*Vect1) = C_norm * (C_a * X07M - C_c * X61M + C_d * X25M - C_f * X43M);
	(*Vect3) = C_norm * (C_c * X07M + C_f * X61M - C_a * X25M + C_d * X43M);
	(*Vect5) = C_norm * (C_d * X07M + C_a * X61M + C_f * X25M - C_c * X43M);
	(*Vect7) = C_norm * (C_f * X07M + C_d * X61M + C_c * X25M + C_a * X43M);
}


/**
**************************************************************************
*  Performs in-place IDCT of vector of 8 elements.
*
* \param Vect0			[IN] - Pointer to the first element of vector
* \param Step			[IN] - Value to add to ptr to access other elements 
*  
* \return None
*/
__device__ void CUDAsubroutineInplaceIDCTvector(float *Vect0, int Step)
{
	float *Vect1 = Vect0 + Step;
	float *Vect2 = Vect1 + Step;
	float *Vect3 = Vect2 + Step;
	float *Vect4 = Vect3 + Step;
	float *Vect5 = Vect4 + Step;
	float *Vect6 = Vect5 + Step;
	float *Vect7 = Vect6 + Step;

	float Y04P   = (*Vect0) + (*Vect4);
	float Y2b6eP = C_b * (*Vect2) + C_e * (*Vect6);

	float Y04P2b6ePP = Y04P + Y2b6eP;
	float Y04P2b6ePM = Y04P - Y2b6eP;
	float Y7f1aP3c5dPP = C_f * (*Vect7) + C_a * (*Vect1) + C_c * (*Vect3) + C_d * (*Vect5);
	float Y7a1fM3d5cMP = C_a * (*Vect7) - C_f * (*Vect1) + C_d * (*Vect3) - C_c * (*Vect5);

	float Y04M   = (*Vect0) - (*Vect4);
	float Y2e6bM = C_e * (*Vect2) - C_b * (*Vect6);

	float Y04M2e6bMP = Y04M + Y2e6bM;
	float Y04M2e6bMM = Y04M - Y2e6bM;
	float Y1c7dM3f5aPM = C_c * (*Vect1) - C_d * (*Vect7) - C_f * (*Vect3) - C_a * (*Vect5);
	float Y1d7cP3a5fMM = C_d * (*Vect1) + C_c * (*Vect7) - C_a * (*Vect3) + C_f * (*Vect5);

	(*Vect0) = C_norm * (Y04P2b6ePP + Y7f1aP3c5dPP);
	(*Vect7) = C_norm * (Y04P2b6ePP - Y7f1aP3c5dPP);
	(*Vect4) = C_norm * (Y04P2b6ePM + Y7a1fM3d5cMP);
	(*Vect3) = C_norm * (Y04P2b6ePM - Y7a1fM3d5cMP);

	(*Vect1) = C_norm * (Y04M2e6bMP + Y1c7dM3f5aPM);
	(*Vect5) = C_norm * (Y04M2e6bMM - Y1d7cP3a5fMM);
	(*Vect2) = C_norm * (Y04M2e6bMM + Y1d7cP3a5fMM);
	(*Vect6) = C_norm * (Y04M2e6bMP - Y1c7dM3f5aPM);
}


/**
**************************************************************************
*  Performs 8x8 block-wise Forward Discrete Cosine Transform of the given 
*  image plane and outputs result to the array of coefficients. 2nd implementation.
*  This kernel is designed to process image by blocks of blocks8x8 that 
*  utilize maximum warps capacity, assuming that it is enough of 8 threads 
*  per block8x8.
*
* \param Dst						[OUT] - Coefficients plane
* \param ImgStride					[IN] - Stride of Dst
* \param NumBlocks8x8InBlock		[IN] - Number of blocks8x8 in block of threads
* \param WidthInBlocks				[IN] - Width of image in blocks8x8
* \param InvWidthInBlocksF			[IN] - Inverse of width of image in blocks8x8
*  
* \return None
*/
__global__ void CUDAkernel2DCT(float *Dst, int ImgStride, int NumBlocks8x8InBlock, int WidthInBlocks, float InvWidthInBlocksF)
{
	//////////////////////////////////////////////////////////////////////////
	// Initialization
	//

	// Floating thread and block numbers
	int ThreadNum = threadIdx.x;
	int BlockNum = blockIdx.x;

	// Current block8x8 in current block
	int CurBlock8x8InBlock = ThreadNum >> BLOCK_SIZE_LOG2;

	// Current column in current block8x8
	int CurThreadInBlock8x8 = ThreadNum - (CurBlock8x8InBlock << BLOCK_SIZE_LOG2);

	// Current block8x8 coordinates in blocks
	int CurBlock8x8OffsetDescriptor = FAST_INT_MUL(BlockNum, NumBlocks8x8InBlock) + CurBlock8x8InBlock;
	int CurBlock8x8OffsetYInBlocks = (int)(((float)CurBlock8x8OffsetDescriptor) * InvWidthInBlocksF + Eps);
	int CurBlock8x8OffsetXInBlocks = CurBlock8x8OffsetDescriptor - FAST_INT_MUL(CurBlock8x8OffsetYInBlocks, WidthInBlocks);

	// Current block8x8 coordinates in pixels (in image)
	int CurBlock8x8OffsetXInImage = CurBlock8x8OffsetXInBlocks << BLOCK_SIZE_LOG2;
	int CurBlock8x8OffsetYInImage = CurBlock8x8OffsetYInBlocks << BLOCK_SIZE_LOG2;

	// Current block8x8 offset in shared memory
	int CurBlock8x8OffsetInShared = CurBlock8x8InBlock << BLOCK_SIZE2_LOG2;
	float *CurBlock8x8 = SharedBlocks + CurBlock8x8OffsetInShared;

	//////////////////////////////////////////////////////////////////////////
	// Copying into shared memory
	//

	// Texture coordinates
	float tex_x = (float)(CurBlock8x8OffsetXInImage + CurThreadInBlock8x8) + 0.5f;
	float tex_y = (float)(CurBlock8x8OffsetYInImage) + 0.5f;
	int CurBlock8x8Index = 0 * BLOCK_SIZE + CurThreadInBlock8x8;

	#pragma unroll
	for (int i=0; i<BLOCK_SIZE; i++)
	{
		CurBlock8x8[CurBlock8x8Index] = tex2D(TexSrc, tex_x, tex_y);
		tex_y += 1.0f;
		CurBlock8x8Index += BLOCK_SIZE;
	}

	//////////////////////////////////////////////////////////////////////////
	// Processing
	//

	//process rows
	__syncthreads();
	CUDAsubroutineInplaceDCTvector(CurBlock8x8 + (CurThreadInBlock8x8<<BLOCK_SIZE_LOG2), 1);

	//process columns
	__syncthreads();
	CUDAsubroutineInplaceDCTvector(CurBlock8x8 + CurThreadInBlock8x8, BLOCK_SIZE);


	//////////////////////////////////////////////////////////////////////////
	// Copying from shared memory
	//

	float *DstAddress = Dst + FAST_INT_MUL(CurBlock8x8OffsetYInImage, ImgStride) + CurBlock8x8OffsetXInImage + CurThreadInBlock8x8;
	CurBlock8x8Index = 0 * BLOCK_SIZE + CurThreadInBlock8x8;
	
	#pragma unroll
	for (int i=0; i<BLOCK_SIZE; i++)
	{
		__syncthreads();
		*DstAddress = CurBlock8x8[CurBlock8x8Index];
		DstAddress += ImgStride;
		CurBlock8x8Index += BLOCK_SIZE;
	}
}


/**
**************************************************************************
*  Performs 8x8 block-wise Inverse Discrete Cosine Transform of the given 
*  coefficients plane and outputs result to the image. 2nd implementation.
*  This kernel is designed to process image by blocks of blocks8x8 that 
*  utilize maximum warps capacity, assuming that it is enough of 8 threads 
*  per block8x8.
*
* \param Dst						[OUT] - Coefficients plane
* \param ImgStride					[IN] - Stride of Dst
* \param NumBlocks8x8InBlock		[IN] - Number of blocks8x8 in block of threads
* \param WidthInBlocks				[IN] - Width of image in blocks8x8
* \param InvWidthInBlocksF			[IN] - Inverse of width of image in blocks8x8
*  
* \return None
*/
__global__ void CUDAkernel2IDCT(float *Dst, int ImgStride, int NumBlocks8x8InBlock, int WidthInBlocks, float InvWidthInBlocksF)
{
	//////////////////////////////////////////////////////////////////////////
	// Initialization
	//

	// Floating thread and block numbers
	int ThreadNum = threadIdx.x;
	int BlockNum = blockIdx.x;

	// Current block8x8 in current block
	int CurBlock8x8InBlock = ThreadNum >> BLOCK_SIZE_LOG2;

	// Current column in current block8x8
	int CurThreadInBlock8x8 = ThreadNum - (CurBlock8x8InBlock << BLOCK_SIZE_LOG2);

	// Current block8x8 coordinates in blocks
	int CurBlock8x8OffsetDescriptor = FAST_INT_MUL(BlockNum, NumBlocks8x8InBlock) + CurBlock8x8InBlock;
	int CurBlock8x8OffsetYInBlocks = (int)(((float)CurBlock8x8OffsetDescriptor) * InvWidthInBlocksF + Eps);
	int CurBlock8x8OffsetXInBlocks = CurBlock8x8OffsetDescriptor - FAST_INT_MUL(CurBlock8x8OffsetYInBlocks, WidthInBlocks);

	// Current block8x8 coordinates in pixels (in image)
	int CurBlock8x8OffsetXInImage = CurBlock8x8OffsetXInBlocks << BLOCK_SIZE_LOG2;
	int CurBlock8x8OffsetYInImage = CurBlock8x8OffsetYInBlocks << BLOCK_SIZE_LOG2;

	// Current block8x8 offset in shared memory
	int CurBlock8x8OffsetInShared = CurBlock8x8InBlock << BLOCK_SIZE2_LOG2;
	float *CurBlock8x8 = SharedBlocks + CurBlock8x8OffsetInShared;

	//////////////////////////////////////////////////////////////////////////
	// Copying into shared memory
	//

	// Texture coordinates
	float tex_x = (float)(CurBlock8x8OffsetXInImage + CurThreadInBlock8x8) + 0.5f;
	float tex_y = (float)(CurBlock8x8OffsetYInImage) + 0.5f;

	#pragma unroll
	for (int i=0; i<BLOCK_SIZE; i++)
	{
		CurBlock8x8[(i<<BLOCK_SIZE_LOG2)+CurThreadInBlock8x8] = tex2D(TexSrc, tex_x, tex_y);
		tex_y += 1.0f;
	}

	//////////////////////////////////////////////////////////////////////////
	// Processing
	//

	//process rows
	__syncthreads();
	CUDAsubroutineInplaceIDCTvector(CurBlock8x8 + (CurThreadInBlock8x8<<BLOCK_SIZE_LOG2), 1);

	//process columns
	__syncthreads();
	CUDAsubroutineInplaceIDCTvector(CurBlock8x8 + CurThreadInBlock8x8, BLOCK_SIZE);


	//////////////////////////////////////////////////////////////////////////
	// Copying from shared memory
	//

	float *DstAddress = Dst + FAST_INT_MUL(CurBlock8x8OffsetYInImage, ImgStride) + CurBlock8x8OffsetXInImage + CurThreadInBlock8x8;
	#pragma unroll
	for (int i=0; i<BLOCK_SIZE; i++)
	{
		__syncthreads();
		*DstAddress = CurBlock8x8[(i<<BLOCK_SIZE_LOG2)+CurThreadInBlock8x8];
		DstAddress += ImgStride;
	}
}
