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
#include<dlfcn.h>

#include <cuda_toolsapi.h>
#include <cuda_toolsapi_tau.h>

static struct {
	void * handle;
	cuToolsApi_Core* coreTable;
	cuToolsApi_Device* deviceTable;
} gs_toolsapi;

inline int InitializeToolsApi(void)
{
	CUresult status;

	gs_toolsapi.handle = dlopen("libcuda.so.190.36", RTLD_GLOBAL | RTLD_NOW);
	//gs_toolsapi.handle = dlopen("/home/shangkar/NEXUS/NVIDIA-Linux-x86_64-190.36-pkg2/usr/lib/libcuda.so.190.36", RTLD_GLOBAL | RTLD_LAZY);
	//gs_toolsapi.handle = dlopen("/home/shangkar/NEXUS/NVIDIA-Linux-x86_64-190.36-pkg2/usr/lib/libcuda.so.190.36", RTLD_LOCAL | RTLD_NOW);
	if (!gs_toolsapi.handle) 
	{
		fprintf(stderr, "Failed to load libcuda.so >> %s\n", dlerror());
		return 1;
	}

	cuDriverGetExportTable_pfn getExportTable;
	getExportTable = (cuDriverGetExportTable_pfn) dlsym(gs_toolsapi.handle, "cuDriverGetExportTable");
	if (!getExportTable) 
	{
		fprintf(stderr, "Failed to load function 'cuDriverGetExportTable' from libcuda.so\n");
		return 1;
	}

        status = getExportTable(&cuToolsApi_ETID_Device, (const void**) &gs_toolsapi.deviceTable);
        if (status != CUDA_SUCCESS)
        {
                fprintf(stderr, "Failed to load device table\n");
                return 1;
        }


	status = getExportTable(&cuToolsApi_ETID_Core, (const void**) &gs_toolsapi.coreTable);
	if (status != CUDA_SUCCESS)
	{
		fprintf(stderr, "Failed to load core table %X\n", gs_toolsapi.coreTable);
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
	if (gs_toolsapi.handle)
	{
		dlclose(gs_toolsapi.handle);
	}

	return 0;
}

inline cuToolsApi_Core* GetCoreTable(void)
{
	return gs_toolsapi.coreTable;
}

inline cuToolsApi_Device* GetDeviceTable(void)
{
        return gs_toolsapi.deviceTable;
}

