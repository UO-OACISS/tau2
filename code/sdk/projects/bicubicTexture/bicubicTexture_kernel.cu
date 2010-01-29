/*
 * Copyright 1993-2008 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/*
    Bicubic filtering
    See GPU Gems 2: "Fast Third-Order Texture Filtering", Sigg & Hadwiger

    Reformulation thanks to Keenan Crane
*/

#ifndef _BICUBIC_KERNEL_H_
#define _BICUBIC_KERNEL_H_

#include "cutil_math.h"

texture<uchar, 2, cudaReadModeNormalizedFloat> tex;

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__device__ float w0(float a)
{
    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
}

__device__ float w1(float a)
{
    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
}

__device__ float w2(float a)
{
    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
}

__device__ float w3(float a)
{
    return (1.0f/6.0f)*(a*a*a);
}

// g0 and g1 are the two amplitude functions
__device__ float g0(float a)
{
    return w0(a) + w1(a);
}

__device__ float g1(float a)
{
    return w2(a) + w3(a);
}

// h0 and h1 are the two offset functions
__device__ float h0(float a)
{
    // note +0.5 offset to compensate for CUDA linear filtering convention
    return -0.5f + w1(a) / (w0(a) + w1(a));
}

__device__ float h1(float a)
{
    return 1.5f + w3(a) / (w2(a) + w3(a));
}

// perform bicubic texture lookup
template<class T>
__device__
T tex2DBicubic(float x, float y)
{
    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    // note: we could store these functions in a lookup table texture,
    // but maths is cheap these days
    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);

    T r = g0(fy) * ( g0x * tex2D(tex, px + h0x, py + h0y)   +
                     g1x * tex2D(tex, px + h1x, py + h0y) ) +
          g1(fy) * ( g0x * tex2D(tex, px + h0x, py + h1y)   +
                     g1x * tex2D(tex, px + h1x, py + h1y) );
    return r;
}

// render image using normal bilinear texture lookup
__global__ void
d_render(uchar *d_output, uint width, uint height, float tx, float ty, float scale, float cx, float cy)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint i = __umul24(y, width) + x;

    float u = (x-cx)*scale+cx + tx;
    float v = (y-cy)*scale+cy + ty;

    if ((x < width) && (y < height)) {
        // write output color
        d_output[i] = tex2D(tex, u, v) * 0xff;
    }
}

// render image using bicubic texture lookup
__global__ void
d_renderBicubic(uchar *d_output, uint width, uint height, float tx, float ty, float scale, float cx, float cy)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    uint i = __umul24(y, width) + x;

    float u = (x-cx)*scale+cx + tx;
    float v = (y-cy)*scale+cy + ty;

    if ((x < width) && (y < height)) {
        // write output color
        d_output[i] = tex2DBicubic<float>(u, v) * 0xff;
    }
}

#endif // _BICUBIC_KERNEL_H_
