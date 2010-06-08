/*
* Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and 
* proprietary rights in and to this software and related documentation and 
* any modifications thereto.  Any use, reproduction, disclosure, or distribution 
* of this software and related documentation without an express license 
* agreement from NVIDIA Corporation is strictly prohibited.
* 
*/

#ifndef __CUDA_TOOLSAPI_H__
#define __CUDA_TOOLSAPI_H__

//  The CUDA Tools API prefix is cuToolsApi_
//
//  Clients of this API can only use the abstractions presented by the
//  public CUDA headers.  Dependencies on internal driver data types
//  are not allowed.

#include <cuda.h>
#include "nvtypes.h"        //  All fundamental types are defined here.

#define cuToolsApi_UUID_SIZE_IN_BYTES (16)

#ifdef GUID_DEFINED
typedef GUID cuToolsApi_UUID;
#else
/// The standard UUID definition.  sizeof(tag_cuToolsApi_UUID) *needs* to be 16.
/// If it doesn't on a needed compilation platform, we need nasty Work-ARounds.
typedef struct tag_cuToolsApi_UUID {
    NvU32 Data1;
    NvU16 Data2;
    NvU16 Data3;
    NvU8  Data4[8];
} __attribute__((packed)) cuToolsApi_UUID;

bool operator==(const cuToolsApi_UUID first, const cuToolsApi_UUID second)
{
	if(first.Data1!=second.Data1)return false; 
	if(first.Data2!=second.Data2)return false; 
	if(first.Data3!=second.Data3)return false;
	for(int i=0;i<8;i++)
		if(first.Data4[i]!=second.Data4[i]) return false;
	return true; 
}

#endif

#if defined(cuToolsApi_INITGUID)
    // MSVC seems to require the use of "extern" here, whereas every other
    // compiler seems require omitting it.
    #ifdef _WIN32
        #define cuToolsApi_DEFINE_GUID(x__, a, b, c, d0,d1,d2,d3,d4,d5,d6,d7) \
            extern \
            const cuToolsApi_UUID x__ = {a, b, c, {d0,d1,d2,d3,d4,d5,d6,d7}}
    #else
        #define cuToolsApi_DEFINE_GUID(x__, a, b, c, d0,d1,d2,d3,d4,d5,d6,d7) \
            const cuToolsApi_UUID x__ = {a, b, c, {d0,d1,d2,d3,d4,d5,d6,d7}}
    #endif
#else
    #define cuToolsApi_DEFINE_GUID(x__, a, b, c, d0,d1,d2,d3,d4,d5,d6,d7) \
        extern const cuToolsApi_UUID x__
#endif


#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// This is deprecated.  When video interop code in other branches dependent
// on it has has changed, and those changes make it to cuda_a, this definition
// should be removed.
typedef struct cuToolsApi_Root_st {
    NvBool (CUDAAPI *QueryForExportTable)(
        const cuToolsApi_UUID* exportTableId,
        const void** ppExportTable);
} cuToolsApi_Root;
// End deprecated.

typedef CUresult (CUDAAPI *cuDriverGetExportTable_pfn)(
    const cuToolsApi_UUID* exportTableId,
    const void** ppExportTable);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
