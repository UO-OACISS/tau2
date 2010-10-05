/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__CUPTI_CALLBACKS_H__)
#define __CUPTI_CALLBACKS_H__

/* Driver api types */
#include <cuda.h>

/* Runtime api types */
#include <builtin_types.h>         

/* define size_t */
#include <string.h>

/* define standard integer types */
#include <cuda_stdint.h>

/** Maximum number of subscriber callbacks to the cuda driver. */
#define MAX_SUBSCRIBERS 1

#ifndef CUPTIAPI
#ifdef _WIN32
#define CUPTIAPI __stdcall
#else 
#define CUPTIAPI
#endif 
#endif

#if defined(__cplusplus)
extern "C" {
#endif 

/* -------------------------------------------------------------------------- */
/**
 * \defgroup CUPTI_API CUDA Profiler Tools Interface
 *
 * This section describes the CUDA profiler tools interface
 *
 * @{
 */

/** 
 * \defgroup CUPTI_CALLBACK_TYPES Data types used for CUPTI callback management 
 * @{
 */

/**
 * Specifies at what point in an api call a callback was issued.
 */
typedef enum CUpti_ApiTraceIssueSite_enum {
    CUPTI_API_ENTER                 = 0,
    CUPTI_API_EXIT                  = 1,
    CUPTI_API_ENTRY_POINT_FORCE_INT = 0x7fffffff
} CUpti_ApiTraceIssueSite;

/**
 * Container passed into the callback which describes the source runtime api call. 
 */
typedef struct CUpti_RuntimeTraceApi_st {
    /** 
     * Current size of struct - can be used for limited versioning.  
     */
    uint32_t struct_size;     

    /** 
     * Unused. 
     */
    uint32_t reserved0;       
   
    /**
     * Sequential id number given to contexts created by the runtime which 
     * is unique per process. Set to zero when there is no context current 
     * to the thread, or during runtime-driver interop when a driver 
     * context has not yet been attached to the runtime thread.            
     */
    uint64_t contextId;  

    /**
     * Sequential id number assigned to streams created by the runtime which is
     * unique per context. This is set to zero when no stream is passed or when
     * no context has been initialized by the runtime.
     */
    uint64_t streamId;

    /**
     * Sequential id number given to each runtime api call which is unique per 
     * context. This id is only valid in api exit callbacks and is zero
     * when no context is current to the thread. 
     */ 
    uint64_t apiCallId;

    /**
     * Pointer to data shared between entry and exit callbacks of a 
     * runtime api function. Arbitrary 64 bit values can be passed from
     * entry to exit callbacks to correlate the callbacks from a single
     * api call. 
     */
    uint64_t *pCorrelationId;
 
    /**
     * Pointer to return status of a runtime api call. Status is only 
     * guaranteed to be valid within the exit callback. 
     */
    cudaError_t *pStatus;

    /**
     * Name of the runtime api function which caused the callback.
     */ 
    const char* functionName;

    /**  
     * Pointer to struct wrapping all arguments passed to the runtime api call.
     */
    const void* params;

    /** 
     * Driver context used by the runtime api. If no context is current
     * to the runtime thread, this will be null. This value can change 
     * from the entry to exit callback of a function if the runtime 
     * initializes a context.
     */ 
    CUcontext ctx;
    
    /** 
     * Stream used by runtime api call. If no stream is passed to a runtime                           
     * function, this will be null. 
     */
    cudaStream_t stream;

    /**
     * Unique id of the function in the runtime api trace domain which
     * issued the callback.
     */
    uint32_t functionId;

    /**
     * Point in the runtime function from where the callback was issued. 
     */ 
    CUpti_ApiTraceIssueSite callbacksite;

} CUpti_RuntimeTraceApi;

/**
 * Container passed into the callback which describes driver api call. 
 */
typedef struct CUpti_DriverTraceApi_st
{
    /** 
     * Current size of struct - can be used for limited versioning.  
     */
    uint32_t struct_size;
 
    /** 
     * Unused. 
     */
    uint32_t reserved0;
  
    /**
     * Sequential id number given to contexts created by the driver which is 
     * unique per process. Id is set to zero when there is no context current
     * to the thread.
     */
    uint64_t contextId;

    /** 
     * Sequential id number given to streams created by the driver 
     * which is unique per context. Set to zer if no stream is passed 
     * to the driver function. 
     */
    uint64_t streamId;

    /**
     * Sequential id number for each driver api call which is unique per 
     * context. Set to zero if no context is current to the thread.
     */
    uint64_t apiCallId; 

    /**
     * Pointer to data shared between entry and exit callbacks of a 
     * runtime api function. Arbitrary 64 bit values can be passed from
     * entry to exit callbacks to correlate the callbacks from a single 
     * single api call.
     */
    uint64_t *pCorrelationId;

    /**
     * Pointer to the return status of the driver api call. Status is valid
     * only in an exit callback.
     */
    uint32_t *pStatus;

    /**
     * Name of the runtime api function which caused the callback.
     */ 
    const char* functionName;

    /**  
     * Pointer to struct wrapping all arguments passed to the driver api call.
     */
    const void* params;

    /** 
     * Provides context current to the thread, null otherwise.
     */ 
    CUcontext ctx;

    /** 
     * Provides stream passed as argument to the function, null if function
     * takes no stream argument.
     */
    CUstream stream;

    /**
     * Contains unique id of function in the driver api trace domain which
     * issued the callback.
     */
    uint32_t functionId;

    /**
     * Point in the driver function from where the callback was issued. 
     */ 
    CUpti_ApiTraceIssueSite callbacksite;

} CUpti_DriverTraceApi;
 

/**
 * Available domains for callback subscriptions. 
 */
typedef enum CUpti_CallbackDomain_enum
{
    CUPTI_CB_DOMAIN_INVALID                 = 0x000,
    CUPTI_CB_DOMAIN_DRIVER_API_TRACE        = 0x006,
    CUPTI_CB_DOMAIN_RUNTIME_API_TRACE       = 0x007,
    CUPTI_CB_DOMAIN_FORCE_INT               = 0x7fffffff
} CUpti_CallbackDomain;

typedef CUpti_CallbackDomain *CUpti_DomainTable;

/**
 * Unique number which identifies the api function which issued the callback.
 */ 
typedef uint32_t CUpti_CallbackId;

/**
 * \brief Callback function type
 * \param userdata - Arbitrary data given by user at subscription
 * \param domain   - domain of the callback provided by driver/runtime
 * \param cbid     - callback id of callback provided by driver/runtime
 * \param params   - packed information struct from callsite, the type of 
 * which is dependent on the domain - e.g., for runtime trace
 * a CUpti_RuntimeTraceApi* structure will be passed in. 
 */
typedef void (CUPTIAPI *CUpti_CallbackFunc)(
    void *userdata,
    CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid,
    const void *params); 

typedef struct CUpti_Subscriber_st *CUpti_SubscriberHandle;

/** @} */ /* END CUPTI_CALLBACK_TYPES */ 

/* ------------------------------------------------------------------------- */

/** 
 * \defgroup CUPTI_CALLBACK_API Callback management
 * @{
 */

/**
 * \brief Returns callbacks domains supported by driver.
 * 
 * Returns in \p *domainTable a pointer to an array of size \p *domainCount of all
 * the supported callback domains. A callback domain is a group of related callback
 * points that one to subscribe a callback function to, e.g. all return api entry/exit
 * points. 
 *
 * \param domainCount - Returned number of callback domains
 * \param domainTable - Returned pointer to array of supported callback domains 
 * 
 * \return 
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 */
CUresult CUPTIAPI cuptiSupportedDomains(size_t *domainCount, 
                                        CUpti_DomainTable *domainTable);

/** 
 * \brief Register a user function for callbacks with the cuda driver.
 * 
 * Registers callback function pointer to cuda driver/runtime, returning a 
 * handle to this subscription which can be used to enable/disable use of the callback
 * by the driver and to unsubscribe the callback when finished. After a function has been
 * subscribed, no callbacks are enabled by default. Enabling the callback is controlled 
 * via the cuToolsEnable(Callback/Domain/AllDomains) functions. ::cuptiSubscribe will 
 * fail to register a callback if the driver subscriber limit (MAX_SUBSCRIBERS) is exceeded. 
 *
 * threadsafety: safe to call any time.  
 *
 * \param subscriber - Returned handle to callback subscription.
 * \param callback   - Client's callback function pointer.
 * \param userdata   - When the callback is invoked, this value is passed in as
 * the callback's 'userdata' parameter.  Analogous to the client's 'this' pointer.
 *
 * \return 
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE,
 * ::CUDA_ERROR_UNKNOWN
 */
CUresult CUPTIAPI cuptiSubscribe(CUpti_SubscriberHandle *subscriber, 
                                 CUpti_CallbackFunc callback, 
                                 void *userdata);

/**
 * \brief Unregister a user function with the cuda driver.  
 * 
 * Removes subscription of callback from the driver/runtime, and guarantees
 * that no future callbacks will be issued to the subscription on return.
 * 
 * threadsafety: safe to call any time.  
 *
 * \param subscriber - Handle to callback subscription.
 *
 * \return 
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_HANDLE
 */
CUresult CUPTIAPI cuptiUnsubscribe(CUpti_SubscriberHandle subscriber); 

/**
 * \brief Get current enabled/disabled state of callback for a specific
 * domain and callback id. 
 * 
 * Returns in pEnable whether the callback specified by domain and callback id
 * is activated (enabled if *enable != 0) for a certain subscription. 
 *
 * threadsafety: Multiple subscribers may call concurrently, but each
 * subscriber must serialize access to Get and Set.  In other words:
 * If ::GetCallbackEnabled(sub, d, c) and ::SetCallbackEnabled(sub, d, c) are
 * called concurrently, the results are undefined.
 * 
 * \param enable - Returned status of callback (enabled if *pEnable != 0)
 * \param handle - Subsciption handle to callback
 * \param domain - Callback domain (e.g. driver api trace)
 * \param cbid   - Callback id (specifies certain callback issue point in domain)
 *
 * \return 
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 */
CUresult CUPTIAPI cuptiGetCallbackState(uint32_t *enable,               
                                        CUpti_SubscriberHandle handle,
                                        CUpti_CallbackDomain domain,
                                        CUpti_CallbackId cbid);

/**
 * \brief Enable or disable callbacks for the specified subscriber from {domain, 
 * callback-id}.
 *
 * Enable or disable callbacks for the specified subscriber from {domain, callback-id}.
 *
 * threadsafety: Multiple subscribers may call concurrently, but each
 * subscriber must serialize access to Get and Set.  In other words:
 * If ::GetCallbackEnabled(sub, d, c) and ::SetCallbackEnabled(sub, d, c) are
 * called concurrently, the results are undefined.
 * 
 * \param enable - Value to set enable status to (enabled if nonzero)
 * \param handle - Handle to callback subscription 
 * \param domain - Callback domain (e.g. driver api trace)
 * \param cbid   - Callback id (specifies certain callback issue point in domain)
 *
 * \return 
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 */
CUresult CUPTIAPI cuptiEnableCallback(uint32_t enable,
                                      CUpti_SubscriberHandle handle, 
                                      CUpti_CallbackDomain domain,
                                      CUpti_CallbackId cbid);

/**
 * \brief Enable or disable callbacks for the specified subscriber from all callback 
 * id's in domain. 
 *
 * Enable or disable callbacks for the specified subscriber from all callback id's in
 * domain. 
 *
 * threadsafety: Multiple subscribers may call concurrently, but each
 * subscriber must serialize access to Get and Set. In other words:
 * If ::GetCallbackEnabled(sub, d, c) and ::SetCallbackEnabled(sub, d, c) are
 * called concurrently, the results are undefined.
 *
 * \param enable - Value to set enabled status(es) to (enabled if nonzero)
 * \param handle - Handle to callback subscription 
 * \param domain - Callback domain (e.g. driver api trace) where all id's will be set
 *
 * \return 
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 */
CUresult CUPTIAPI cuptiEnableDomain(uint32_t enable,
                                    CUpti_SubscriberHandle handle, 
                                    CUpti_CallbackDomain domain);

/**
 * \brief Enable or disable all possible callbacks for the specified subscriber (All available
 * domains and callback id's).
 *
 * Enable or disable all possible callbacks for the specified subscriber (All available
 * domains and callback id's).
 * 
 * threadsafety: Multiple subscribers may call concurrently, but each
 * subscriber must serialize access to Get and Set.  In other words:
 * If GetCallbackEnabled(sub, d, c) and SetCallbackEnabled(sub, d, c) are
 * called concurrently, the results are undefined.
 *
 * \param enable - Value to set enabled status(es) to (enabled if nonzero)
 * \param handle - Handle to callback subscription 
 *
 * \return 
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_HANDLE
 */
CUresult CUPTIAPI cuptiEnableAllDomains(uint32_t enable, 
                                        CUpti_SubscriberHandle handle);

/** @} */ /* END CUPTI_CALLBACK_API */ 
/** @} */ /* END CUPTI_API */ 

// ----------------------------------------------------------------------------
#if defined(__cplusplus)
}
#endif 

#endif  // file guard

