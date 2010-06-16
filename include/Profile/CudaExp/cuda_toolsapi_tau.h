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

#ifndef __CUDA_TOOLSAPI_UNRELEASED_H__
#define __CUDA_TOOLSAPI_UNRELEASED_H__

#include "cuda_toolsapi.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

///------------------------------------------------------------------------
//  CUDA ToolsApi Summary
///------------------------------------------------------------------------
/// 
/// CUDA ToolsApi provides callbacks and entry-points into the CUDA driver;
/// these form an abstraction layer between driver implementation and tools.
/// 
/// Tools gain access to ToolsApi via the definitions in cuda_toolsapi.h:
/// 
///     cuToolsApi_pfnGetRootExportTable pfnGetRootExportTable = ...;
///     cuToolsApi_Root* pApiRoot = 0;
///     pfnGetRootExportTable(&pApiRoot);
///     
/// Once cuToolsApi_Root* has been obtained, it is possible to query for
/// individual function-export-tables from the ToolsApi.  Each version of 
/// each export-table is assigned a UUID.  If the driver supports that
/// version of an export-table, it is returned to the cllaer; otherwise,
/// that version is not supported by the driver, and the caller must handle
/// the error as appropriate.
///     
///     const cuToolsApi_Core* pApiCore = 0;
///     const cuToolsApi_Context* pApiContext = 0;
///     pApiRoot->QueryForExportTable(
///         cuToolsApi_ETID_Core,
///         (const void**)&pApiCore);
///     pApiRoot->QueryForExportTable(
///         cuToolsApi_ETID_Context,
///         (const void**)&pApiContext);
///         
/// From here, it is possible to register for callbacks from the driver, or
/// directly query for information from the driver.
/// 
/// ToolsApi supports multiple clients.
/// cuToolsApi_Core::Construct() must be called once per client of ToolsApi,
/// before any other exported function is called.  Individual exported
/// functions may have additional pre-requisite requirements, which are
/// documented on a case-by-case basis.
/// 
/// 
/// Lifetime Management
/// -------------------
/// 
/// The intended sequence for the lifetime of a single client is:
/// 
///     /* before cuInit(), or during cuiInit()'s one-time-initialization */
///     /* see Initialization Constraints below for details */
///     NvU64 subscriptionId = 0;
///     pApiCore->Construct();
///     pApiCore->SubscribeCallbacks(
///         ToolsApiCallbackHandler, 
///         0 /* pUserData */,
///         &subscriptionId);
///     pApiCore->EnableCallbacks(NV_TRUE);
///     
///     /* ... now CUDA application runs ... */
///     
///     /* during Tool shutdown or CUDA driver shutdown */
///     /* see caveat below for details */
///     pApiCore->EnableCallbacks(NV_FALSE);
///     pApiCore->UnsubscribeCallbacks(subscriptionId);
///     pApiCore->Destruct();
///     
/// Caveat: 
///     At the time of writing, there are problems with CUDA driver 
///     shutdown on some platforms.  Thus, Tools are written such that they
///     permanently initialize the ToolsApi, and permanently subscribe for 
///     callbacks.
///     The main issue within ToolsApi implementation is the thread-safety 
///     of ApiCore -- currently it is not, since one-time-initialization 
///     meets the immediate requirements for today's Tools.
///     Performance impact of making ToolsApi thread-safe is unknown.
/// 
/// Initialization Constraints:
///     At the time of writing, clients are required to call Construct(),
///     SubscribeCallbacks(), and EnableCallbacks() at one of two times:
///     -   Before cuInit() has been called for the first time.
///     -   In the body of cuiInit()'s one-time initialization.
///         This is an easier place to perform initialization, since the
///         CUDA driver guarantees the thread-safety of cuiInit().
/// 
/// 
/// Callbacks
/// ---------
/// 
/// When a client subscribes for callbacks, they subscribe for ALL
/// callbacks.  The callback's location is specified via the UUID parameter
/// 'callbackId', which the client should use to determine what struct to
/// cast the 'inParams' to.
/// A really simple callback that traces global memory alloc/free to stdout
/// would like this:
/// 
///     int UuidsEqual(cuToolsApi_UUID* lhs, cuToolsApi_UUID* rhs)
///     {
///         return (memcmp(lhs, rhs, sizeof(cuToolsApi_UUID) == 0);
///     }
/// 
///     static void CUDAAPI ToolsApiCallbackHandler(
///         void* pUserData,
///         const cuToolsApi_UUID* callbackId,
///         const void* inParams)
///     {
///         if (UuidsEqual(callbackId, &cuToolsApi_CBID_GlobalMemAlloc))
///         {
///             cuToolsApi_GlobalMemAllocInParams* pInParams =
///                 (cuToolsApi_GlobalMemAllocInParams*)inParams;
///             printf("global mem alloc : devptr = %p\n", pInParams->address);     
///         }
///         else
///         if (UuidsEqual(callbackId, &cuToolsApi_CBID_GlobalMemFree))
///         {
///             cuToolsApi_GlobalMemFreeInParams* pInParams =
///                 (cuToolsApi_GlobalMemFreeInParams*)inParams;
///             printf("global mem free  : devptr = %p\n", pInParams->address);     
///         }
///     }
/// 
/// 
/// Semantically, there are several categories of callbacks.  Clients
/// receive all of these callbacks through the same handler.
/// 
///     1.  API Interception -- These occur at the entry and exit of
///         each API-function.
///         Example:
///             cuToolsApi_CBID_EnterGeneric
///             cuToolsApi_CBID_ExitGeneric
///     
///     2.  Resource Tracking -- These occur at strategically placed
///         locations in the cui layer.  They are used by clients to track
///         allocation/free of CUDA driver resources, such as context,
///         module, and memory.
///         Example:
///             cuToolsApi_CBID_ModuleLoaded    
///             cuToolsApi_CBID_ModuleUnloadStarting
///             cuToolsApi_CBID_ModuleUnloadCompleted
///         Note:
///             Unload is split into Starting and Completed, so that a
///             Tool can be aware of inner resources being deallocated as
///             a side-effect of a Module loading.  For example, a trace
///             may look like:
///             cuToolsApi_CBID_ModuleUnloadStarting
///             cuToolsApi_CBID_GlobalMemFree  // some __device__ variable
///             cuToolsApi_CBID_GlobalMemFree  // some __device__ variable
///             cuToolsApi_CBID_GlobalMemFree  // some __device__ variable
///             cuToolsApi_CBID_ModuleUnloadCompleted
///     
///     3.  Debug-Specific -- Currently, cuiLaunchGrid() is the single
///         cui function through which all kernel-launches are funneled.
///         A hardware-kernel-debugger needs callbacks at specific points
///         in that function (based on its current implementation) in
///         order to modify launch-properties, GPU-state, etc.
///         Example:
///             cuToolsApi_CBID_BeforeFunctionSetup
///             cuToolsApi_CBID_BeforeGridLaunched
///             cuToolsApi_CBID_AfterGridLaunched         
///     
///     4.  Profiler-Specific -- Currently, profiler records are set up and 
///         managed internally in the driver.  Once results are obtained,
///         callbacks are made on each profiler record so that a client
///         can trace or make the data available in a client-specific way.
///         Example:
///             cuToolsApi_CBID_ProfileLaunch
///             cuToolsApi_CBID_ProfileMemcpy
/// 
///------------------------------------------------------------------------

//--------------------------------------------------------------------------
//  "Core" Export-Table
//  Contains functions for setting up the entire API, and callbacks.
//--------------------------------------------------------------------------

/// \brief This is the only callback-signature the client implements.
typedef void (CUDAAPI *cuToolsApi_pfnCallback)(
    void* pUserData,
    const cuToolsApi_UUID* callbackId,
    const void* inParams);

//--------------------------------------------------------------------------

/// \brief ApiCore is used to init the debug logic, enable driver callbacks,
/// and subscribe.
cuToolsApi_DEFINE_GUID(cuToolsApi_ETID_Core,
    0xa76e3e92, 0x2454, 0x44fe, 0xb0, 0xf8, 0xfe, 0x48, 0xdb, 0x93, 0x9d, 0x1b);
typedef struct {
    /// \brief Every client should call Construct before using the ToolsApi.
    /// \return NV_TRUE on success, NV_FALSE on failure.
    NvBool (CUDAAPI *Construct)(void);

    /// \brief Every client should call Destruct once they're done using the ToolsApi.
    /// \return NV_TRUE on success.
    /// Currently should not be called.  Refer to API summary for details.
    NvBool (CUDAAPI *Destruct)(void);

    /// \brief Returns NV_TRUE if callbacks are enabled.
    /// All callbacks are either enabled or disabled.  There is not fine
    /// grain control of callbacks.
    NvBool (CUDAAPI *CallbacksEnabled)(void);

    /// \brief Enable or disable driver extensions for debugging.
    /// \return NV_TRUE or NV_FALSE to indicate whether or not callbacks are
    /// enabled after the function returns.  This is reference-counted, to
    /// allow multiple clients.        
    NvBool (CUDAAPI *EnableCallbacks)(NvBool enable);

    /// \brief Subscribe for synchronous callbacks from within CUDA API functions.
    /// \param pUserData is passed into the callbacks
    /// \param subscriptionId receives the resultant subscription-ID
    /// Threading: Not thread-safe by design.  Only call this at startup.
    NvBool (CUDAAPI *SubscribeCallbacks)(
        cuToolsApi_pfnCallback pfnHandler,
        void* pUserData,
        NvU64* subscriptionId);

    /// \brief Unsubscribe from callbacks, using the subscriptionId generated
    ///     from a previous call to SubscribeCallbacks.
    /// Currently should not be called.  Refer to API summary for details.
    /// Threading: Not thread-safe by design.  
    NvBool (CUDAAPI *UnsubscribeCallbacks)(const NvU64 subscriptionId);
} cuToolsApi_Core;


//------------------------------------------------------------------
//  Callbacks from the driver
//------------------------------------------------------------------

cuToolsApi_DEFINE_GUID(cuToolsApi_CBID_CudaInitialized,
    0x81645ce0, 0xa834, 0x46dc, 0xa7, 0x2b, 0x08, 0xdf, 0x8c, 0x67, 0xdd, 0xb4);
//  no InParams
//  Should be called at the end of cuInit() if successful.  Guaranteed to
//  be in the global critical section at the end of init.  The callback
//  can safely assume everything has been initialized successfully, and
//  that nothing else can happen until it returns.

//
//  CUcontext-related
//

//  ContextCreated
cuToolsApi_DEFINE_GUID(cuToolsApi_CBID_ContextCreated,
    0xe04d62de, 0xf164, 0x4dbb, 0x85, 0x1c, 0x35, 0xe6, 0x73, 0xa7, 0xb1, 0x80);
typedef struct {
    CUcontext ctx;
} cuToolsApi_ContextCreatedInParams;
//  Should be called at the end of cuictx.c :: cuiCtxCreate(), if successful.
//  [note: same area as current gpudbgRegisterClient() ]

//  ContextDestroyStarting
cuToolsApi_DEFINE_GUID(cuToolsApi_CBID_ContextDestroyStarting,
    0x3cb1c116, 0xd98b, 0x4364, 0x96, 0x01, 0xae, 0xcd, 0xa5, 0x1a, 0x39, 0xc3);
typedef struct {
    CUcontext ctx;
} cuToolsApi_ContextDestroyStartingInParams;
//  Should be called at the beginning of cuictx.c :: cuiCtxDestroy().

//  ContextDestroyCompleted
cuToolsApi_DEFINE_GUID(cuToolsApi_CBID_ContextDestroyCompleted,
    0x70e1968f, 0x777c, 0x4e5b, 0xb6, 0x47, 0x60, 0x30, 0x4e, 0xa6, 0x4e, 0xae);
typedef struct {
    CUcontext ctx;
} cuToolsApi_ContextDestroyCompletedInParams;
//  Should be called at the end of cuiCtx.c :: cuiCtxDestroy()

//
//  CuStream-related
//

//  StreamCreated
cuToolsApi_DEFINE_GUID(cuToolsApi_CBID_StreamCreated,
    0xf4f61fa3, 0x8520, 0x4dd2, 0x88, 0xda, 0x4a, 0xb8, 0xa7, 0x10, 0x1a, 0xdd);
typedef struct {
    CUcontext ctx;
    CUstream stream;
} cuToolsApi_StreamCreatedInParams;

//  StreamDestroyStarting
cuToolsApi_DEFINE_GUID(cuToolsApi_CBID_StreamDestroyStarting,
    0x66ca06f6, 0xd3c7, 0x489b, 0xb5, 0x47, 0xbe, 0x4c, 0x31, 0xce, 0x3a, 0xa7);
typedef struct {
    CUcontext ctx;
    CUstream stream;
} cuToolsApi_StreamDestroyStartingInParams;

//  StreamDestroyCompleted
cuToolsApi_DEFINE_GUID(cuToolsApi_CBID_StreamDestroyCompleted,
    0x40da92a2, 0x4ea0, 0x4fc0, 0x81, 0xd3, 0xd0, 0x31, 0xf0, 0xa0, 0x9f, 0xca);
typedef struct {
    CUcontext ctx;
    CUstream stream;
} cuToolsApi_StreamDestroyCompletedInParams;


//
//  API enter and exit
//

//  ApiEnter_Generic
cuToolsApi_DEFINE_GUID(cuToolsApi_CBID_EnterGeneric,
    0x9f55ccc7, 0xeaa6, 0x4097, 0x8f, 0x3d, 0x4e, 0x45, 0x2e, 0xed, 0xc, 0xa3);
typedef struct {
    CUcontext ctx;
    CUstream stream;
    NvU32 functionIndex;
    const char* functionName;
    const void* params;
    NvU64 apiCallId;
    NvU64* pCallId; // This is an out parameter!
} cuToolsApi_EnterGenericInParams;

//  ApiExit_Generic
cuToolsApi_DEFINE_GUID(cuToolsApi_CBID_ExitGeneric,
    0x409ce0b2, 0xc84c, 0x4be9, 0xb7, 0x23, 0x3b, 0x1a, 0x5e, 0xab, 0xc3, 0x16);
typedef struct {
    CUcontext ctx;
    CUstream stream;
    NvU32 functionIndex;
    const char* functionName;
    const void* params;
    NvU64 apiCallId;
    NvU64 callId; // This is an in parameter!
    CUresult status;
} cuToolsApi_ExitGenericInParams;

//
//  CUDA Profiler related
//
//  When configured to use callbacks, the profiler will not log records
//  to a file, but instead call these callbacks for each record.

//  ProfileLaunch
cuToolsApi_DEFINE_GUID(cuToolsApi_CBID_ProfileLaunch,
    0x7d9bb8f4, 0x80d3, 0x4be7, 0xa8, 0x60, 0xe1, 0x4, 0x5d, 0x63, 0xfa, 0x7b);
typedef struct
{
    CUcontext ctx;              // Handle, not unique over application lifetime
    CUstream stream;            // Handle, not unique over application lifetime
    CUmodule mod;               // Handle, not unique over application lifetime
    CUfunction func;            // Handle, not unique over application lifetime
    NvU64 contextId;            // Incrementing ID, unique per process
    NvU64 streamId;             // Incrementing ID, unique per context
    NvU32 moduleId;             // Incrementing ID, unique per context
    NvU32 functionId;           // Incrementing ID, unique per module
    NvU64 apiCallId;            // Incrementing ID, unique per context
    NvU64 startTime;            // NV01TIMER timestamp, nanoseconds since reset
    NvU64 endTime;              // NV01TIMER timestamp, nanoseconds since reset
    const char* methodName;
    NvF32 occupancy;
    NvU32 blockSizeX;
    NvU32 blockSizeY;
    NvU32 blockSizeZ;
    NvU32 gridSizeX;
    NvU32 gridSizeY;
    NvU32 registerPerThread;
    NvU32 staticSharedMemPerBlock;
    NvU32 dynamicSharedMemPerBlock;
    NvU32 counters;                 // Number of attached hardware counters
    const NvU32* counterValues;     // List of counter data
    const char** counterNames;      // List of counter names
    NvU32 underLoaderLock;          // Called under DllMain on Windows
} cuToolsApi_ProfileLaunchInParams;

//  GpuMethodDmaTransfer
cuToolsApi_DEFINE_GUID(cuToolsApi_CBID_ProfileMemcpy,
    0x6b78cec, 0x7132, 0x4fad, 0x93, 0x8e, 0xd0, 0x67, 0xa7, 0x62, 0xe5, 0xc5);
typedef struct
{
    CUcontext ctx;               // Handle, not unique over application lifetime
    CUstream stream;             // Handle, not unique over application lifetime
    NvU64 contextId;             // Incrementing ID, unique per process
    NvU64 streamId;              // Incrementing ID, unique per context
    NvU64 apiCallId;             // Incrementing ID, unique per context
    NvU64 startTime;             // NV01TIMER timestamp, nanoseconds since reset
    NvU64 endTime;               // NV01TIMER timestamp, nanoseconds since reset
    NvU64 memTransferSize;       // Number of bytes transferred
    NvU32 memTransferSrcType;    // Use CUmemorytype + pinned flag (0x100)
    NvU32 memTransferDstType;    // Use CUmemorytype + pinned flag (0x100)
    NvBool memTransferAsyncCall; // True if memcpy was expected to support overlap
    NvBool memTransferAsyncExec; // True if memcpy actually supported overlap
    NvU32 underLoaderLock;       // Called under DllMain on Windows
} cuToolsApi_ProfileMemcpyInParams;

//------------------------------------------------------------------
//  Export tables - interfaces into the driver for callbacks to use
//------------------------------------------------------------------


// =============================================================================
// DEVICE
// =============================================================================

// DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE
// Base enumeration value for attributes that can be retrieved from the CUDA API
// using a method other than cuDeviceGetAttribute.  (e.g. cuDeviceGetName)
//
// DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE
// Base enumeration value for attributes that cannot be retrieved using the
// CUDA Driver API.
//
// DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE
// Base enumeration value for directly mapping to the internal CUdeviceLimits.
//
// These base numbers should be sufficiently high as to ensure that there is
// never a conflict between CUdevice_attribute and cuToolsApi_DeviceAttribute.
//
// All enumeration values should be explicitly set.  Do not rely on auto-increment.

enum
{
    cuToolsApi_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE            = 0x10000000UL,
    cuToolsApi_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_BASE           = 0x20000000UL,
    cuToolsApi_DEVICE_ATTRIBUTE_INTERNAL_PRIVATE_LIMITS_BASE    = 0x30000000UL
};

// Add attributes that map to CUdevice_attribute_enum 
#define cuToolsApi_DEVICE_ATTRIBUTES_PUBLIC \
    cuToolsApi_DEFINE_ATTRIBUTE(MAX_THREADS_PER_BLOCK,          CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),        \
    cuToolsApi_DEFINE_ATTRIBUTE(MAX_BLOCK_DIM_X,                CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),              \
    cuToolsApi_DEFINE_ATTRIBUTE(MAX_BLOCK_DIM_Y,                CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),              \
    cuToolsApi_DEFINE_ATTRIBUTE(MAX_BLOCK_DIM_Z,                CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z),              \
    cuToolsApi_DEFINE_ATTRIBUTE(MAX_GRID_DIM_X,                 CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),               \
    cuToolsApi_DEFINE_ATTRIBUTE(MAX_GRID_DIM_Y,                 CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y),               \
    cuToolsApi_DEFINE_ATTRIBUTE(MAX_GRID_DIM_Z,                 CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z),               \
    cuToolsApi_DEFINE_ATTRIBUTE(MAX_SHARED_MEMORY_PER_BLOCK,    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK),  \
    cuToolsApi_DEFINE_ATTRIBUTE(TOTAL_CONSTANT_MEMORY,          CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY),        \
    cuToolsApi_DEFINE_ATTRIBUTE(WARP_SIZE,                      CU_DEVICE_ATTRIBUTE_WARP_SIZE),                    \
    cuToolsApi_DEFINE_ATTRIBUTE(MAX_PITCH,                      CU_DEVICE_ATTRIBUTE_MAX_PITCH),                    \
    cuToolsApi_DEFINE_ATTRIBUTE(MAX_REGISTERS_PER_BLOCK,        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK),      \
    cuToolsApi_DEFINE_ATTRIBUTE(CLOCK_RATE,                     CU_DEVICE_ATTRIBUTE_CLOCK_RATE),                   \
    cuToolsApi_DEFINE_ATTRIBUTE(TEXTURE_ALIGNMENT,              CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT),            \
    cuToolsApi_DEFINE_ATTRIBUTE(GPU_OVERLAP,                    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP),                  \
    cuToolsApi_DEFINE_ATTRIBUTE(MULTIPROCESSOR_COUNT,           CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),         \
    cuToolsApi_DEFINE_ATTRIBUTE(KERNEL_EXEC_TIMEOUT,            CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT),          \
    cuToolsApi_DEFINE_ATTRIBUTE(INTEGRATED,                     CU_DEVICE_ATTRIBUTE_INTEGRATED),                   \
    cuToolsApi_DEFINE_ATTRIBUTE(CAN_MAP_HOST_MEMORY,            CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY),          \
    cuToolsApi_DEFINE_ATTRIBUTE(COMPUTE_MODE,                   CU_DEVICE_ATTRIBUTE_COMPUTE_MODE),                 \

// Add attributes that can be queried through functions in cuda.h
#define cuToolsApi_DEVICE_ATTRIBUTES_INTERNAL_PUBLIC \
    cuToolsApi_DEFINE_ATTRIBUTE(DISPLAY_NAME,                   cuToolsApi_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 0), \
    cuToolsApi_DEFINE_ATTRIBUTE(COMPUTE_CAPABILITY_MAJOR,       cuToolsApi_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 1), \
    cuToolsApi_DEFINE_ATTRIBUTE(COMPUTE_CAPABILITY_MINOR,       cuToolsApi_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 2), \
    cuToolsApi_DEFINE_ATTRIBUTE(TOTAL_MEMORY,                   cuToolsApi_DEVICE_ATTRIBUTE_INTERNAL_PUBLIC_BASE + 3), \


// cuToolsApi_DEVICE_ATTRIBUTE_TABLE
// This macros combines the attribute tables.
#define cuToolsApi_DEVICE_ATTRIBUTE_TABLE                  \
    cuToolsApi_DEVICE_ATTRIBUTES_PUBLIC                     \
    cuToolsApi_DEVICE_ATTRIBUTES_INTERNAL_PUBLIC            \


// Generate cuToolsApi_DeviceAttribute enumeration.  This is done by
// defining cuToolsApi_DEFINE_ATTRIBUTE to generate the line
//      cuToolsApi_DEVICE_ATTRIBUTE_<NAME> = <VALUE>
// The comma is provided by the cuToolsApi_DEFINE_ATTRIBUTE definition in
// the previous tables.
#ifdef cuToolsApi_DEFINE_ATTRIBUTE
#undef cuToolsApi_DEFINE_ATTRIBUTE
#endif
#define cuToolsApi_DEFINE_ATTRIBUTE(NAME, VALUE) cuToolsApi_DEVICE_ATTRIBUTE_ ## NAME = (VALUE)

/// Device Attributes
typedef enum cuToolsApi_DeviceAttribute
{
    cuToolsApi_DEVICE_ATTRIBUTE_TABLE
} cuToolsApi_DeviceAttribute;

#undef cuToolsApi_DEFINE_ATTRIBUTE

/// Device Attribute Properties.
typedef enum cuToolsApi_DeviceAttributeProperty
{
    cuToolsApi_DEVICE_ATTRIBUTE_PROPERTY_VALUE = 0,         /// Value of the attribute
    cuToolsApi_DEVICE_ATTRIBUTE_PROPERTY_VISIBILITY = 1,    /// Visibility (public or private) or the attribute
} cuToolsApi_DeviceAttributeProperty;

typedef enum cuToolsApi_DeviceAttributeVisibility
{
    cuTools_API_DEVICE_ATTRIBUTE_VISIBILITY_PUBLIC = 0,     /// Visible through the CUDA driver API
    cuTools_API_DEVICE_ATTRIBUTE_VISIBILITY_PRIVATE = 1     /// Only visible through the tools API
} cuToolsApi_DeviceAttributeVisibility;

/// Type of the value returned in a cuToolsApi_Variant.
typedef enum cuToolsApi_VariantType
{
    cuToolsApi_VariantType_Unknown      = 0,
    cuToolsApi_VariantType_S64          = 1,    // signed 64-bit integer
    cuToolsApi_VariantType_U64          = 2,    // unsigned 64-bit integer
    cuToolsApi_VariantType_ConstString  = 3,    // const char* with a lifetime equal to the driver
} cuToolsApi_VariantType;

/// Flexible structure for returning variable data types.
/// \p type contains a cuToolsApi_VariantType defining which member of the
/// \p data union the value is contained in.
/// \p data contains the value.
// NOTE: pchar is only valid for the lifetime of the CUDA driver.  It is
// recommended that the caller immediately copy strings returned by
// cuToolsApi_DeviceGetAttributeProperty.
typedef struct cuToolsApi_Variant
{
    NvU32 type;           // assign a cuToolsApi_VariantType
    NvU32 reserved;
    union
    {
        NvS64 s64;
        NvU64 u64;
        const char* pchar;
    } data;
} cuToolsApi_Variant;

/// \brief The device table provides functions to retrieve state from CUdevice
cuToolsApi_DEFINE_GUID(cuToolsApi_ETID_Device,
    0x275fc221, 0x8b23, 0x438a, 0x85, 0xc9, 0x04, 0x90, 0x54, 0xf3, 0x6c, 0x10);
typedef struct
{
    /// \brief Returns a handle to a compute device
    /// Returns in \p *device a device handle given an ordinal in the range <b>[0,
    /// ::cuDeviceGetCount()-1]</b>.
    ///
    /// This is a wrapper for cuDeviceGet.
    ///
    /// \param device  - Returned device handle
    /// \param ordinal - Device number to get handle for
    /// \sa ::cuDeviceGet
    NvBool (CUDAAPI *DeviceGet)(
        CUdevice* device,
        int ordinal);

    /// \brief Returns the number of compute-capable devices
    /// 
    /// /Returns in \p *count the number of devices with compute capability greater
    /// than or equal to 1.0 that are available for execution. If there is no such
    /// device, ::cuDeviceGetCount() returns 0.
    ///
    /// This is a wrapper for cuDeviceGetCount.
    ///
    /// \param count - Returns number of compute-capable devices
    /// \sa ::cuDeviceGetCount
    NvBool (CUDAAPI *DeviceGetCount)(int* count);

    /// \brief Returns the number of attributes for the device \p dev in \p *count.
    /// \param dev   - device handle
    /// \param count - Returns the number of attributes for the device.
    NvBool (CUDAAPI *DeviceGetAttributeCount)(CUdevice dev, int* count);

    /// \brief Returns a list of device attribute keys
    /// On input \p count contains the number of items allocated in \p *keys.
    /// \param dev   - device handle
    /// \param count - specifies the number of elements in \p *keys
    /// \param keys  - caller allocated memory in which \p *count elements
    ///                keys are returned.
    ///
    /// /Returns in \p *keys an array of \p count keys
    NvBool (CUDAAPI *DeviceGetAttributeKeys)(
        CUdevice dev,
        int count,
        cuToolsApi_DeviceAttribute* keys);

    /// \brief Returns a property of the attribute
    /// \param dev   - device handle
    /// \param key   - attribute identifier
    /// \param prop  - property identifier
    /// \param value - value of property
    NvBool (CUDAAPI *DeviceGetAttributeProperty)(
        CUdevice dev,
        cuToolsApi_DeviceAttribute key,
        cuToolsApi_DeviceAttributeProperty prop,
        cuToolsApi_Variant* value);

    /// \brief Acquire a timestamp (nanoseconds since reset) from a specific CUDA device.
    NvBool (CUDAAPI *DeviceGetTimestamp)(
        CUdevice dev,
        NvU64* timestamp);

    /// \brief Get OS-specific device ordinal.
    /// On WINLH (Vista), the osDeviceOrdinal == iDevNum parameter of EnumDisplayDevices
    /// On other platforms, currently it fails.
    NvBool (CUDAAPI *DeviceGetOsDeviceOrdinal)(CUdevice dev, NvU32* pOsDeviceOrdinal);

} cuToolsApi_Device;


/// \brief ApiContext contains functions for controlling a CUcontext
cuToolsApi_DEFINE_GUID(cuToolsApi_ETID_Context,
    0xd9acf7c0, 0x7b67, 0x4a8a, 0x91, 0x52, 0x5c, 0x4e, 0x41, 0xba, 0xb9, 0xb3);
typedef struct {
    void* reserved1;
    void* reserved2;
    void* reserved3;
    void* reserved4;
    void* reserved5;
    void* reserved6;
    void* reserved7;

    NvBool (CUDAAPI *CtxGetDevice)(
        CUcontext ctx,
        NvU32* deviceIndex);

    /// \brief Gets the unique ID for the context.
    /// Context IDs are linearly increasing.  Unique IDs should be used in
    /// place of the CUcontext address as the address can be re-used.
    NvBool (CUDAAPI *CtxGetId)(
        CUcontext ctx,
        NvU64* pContextId);

    /// \brief Gets the unique ID for the stream.
    /// Stream IDs are linearly increasing.  Unique IDs should be used in
    /// place of the CUstream address as the address can be re-used.
    NvBool (CUDAAPI *StreamGetId)(
        CUcontext ctx,
        CUstream stream,
        NvU64* pStreamId);

} cuToolsApi_Context;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
