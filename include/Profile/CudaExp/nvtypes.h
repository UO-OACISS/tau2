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

#ifndef NVTYPES_INCLUDED
#define NVTYPES_INCLUDED

typedef unsigned char      NvU8;  /* 0 to 255                                */
typedef unsigned short     NvU16; /* 0 to 65535                              */
//typedef unsigned long      NvU32; /* 0 to 4294967295                         */
typedef unsigned int NvU32;
typedef unsigned long long NvU64; /* 0 to 18446744073709551615               */
typedef signed char        NvS8;  /* -128 to 127                             */
typedef signed short       NvS16; /* -32768 to 32767                         */
typedef signed int         NvS32; /* -2147483648 to 2147483647               */
typedef signed long long   NvS64; /* 2^-63 to 2^63-1                         */
typedef float              NvF32; /* IEEE Single Precision (S1E8M23)         */
typedef double             NvF64; /* IEEE Double Precision (S1E11M52)        */

enum { NV_FALSE = 0, NV_TRUE = 1 };
typedef NvU8 NvBool;

#endif /* NVTYPES_INCLUDED */
