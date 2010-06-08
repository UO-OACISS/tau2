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

#pragma once

#include <stdio.h>
#include "cuda_toolsapi.h"
#include "cuda_toolsapi_tau.h"

extern "C" CUresult CUDAAPI cuDriverGetExportTable(const cuToolsApi_UUID* exportTableId, const void** ppExportTable);


static struct {
	cuToolsApi_Core* coreTable;
	//cuToolsApi_Device* deviceTable;
	//cuToolsApi_Context* contextTable;
} gs_toolsapi;

inline int InitializeToolsApi(void)
{
	CUresult status;
	
	fprintf(stderr, "Before initializing Core::  %x \n", gs_toolsapi.coreTable);	
	status = cuDriverGetExportTable(&cuToolsApi_ETID_Core, (const void**) &gs_toolsapi.coreTable);
	if (status != CUDA_SUCCESS)
	{
		fprintf(stderr, "Failed to load core table Core %x \n", gs_toolsapi.coreTable);	
		return 1;
	}

	if (!gs_toolsapi.coreTable->Construct())
	{
		fprintf(stderr, "Failed to initialize tools API\n");
		return 1;
	}
	return 0;
}

inline int ShutdownToolsApi()
{
	if (gs_toolsapi.coreTable)
	{
		gs_toolsapi.coreTable->Destruct();
	}
	return 0;
}

inline cuToolsApi_Core* GetCoreTable(void)
{
	return gs_toolsapi.coreTable;
}

