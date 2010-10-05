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

#if !defined(_CUPTI_EVENTS_H_)
#define _CUPTI_EVENTS_H_

/* Driver api types */
#include <cuda.h>

/* define size_t */
#include <string.h>

/* define standard integer types */
#include <cuda_stdint.h>

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

/**
 * @addtogroup CUPTI_API CUDA Profiler Tools Interface
 *
 * This section describes the CUDA profiler tools interface
 *
 * @{
 */

/**
 * \defgroup CUPTI_EVENT_TYPES Data types used for CUPTI event management
 * @{
 */

typedef uint32_t CUpti_EventDomainID; /**< Event Domain ID */
typedef uint32_t CUpti_EventID;       /**< Event ID */
typedef void *   CUpti_EventGroup;    /**< Event Group */

/**
 * Error codes
 */
typedef enum CUpti_Error_enum {
    /**
     * The API call returned with no errors.
     */
    CUPTI_SUCCESS                                       = 0,
    /**
     * This indicates that one or more of the parameters passed to the API call
     * is invalid.
     */
    CUPTI_ERROR_INVALID_PARAMETER                       = 1,
    /**
     * This indicates that event domain id passed to the API call is invalid.
     */
    CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID                 = 2,   
    /**
     * This indicates that event id passed to the API call is invalid.
     */
    CUPTI_ERROR_INVALID_EVENT_ID                        = 3,
    /**
     * This indicates that event name passed to the API call is invalid.
     */
    CUPTI_ERROR_INVALID_EVENT_NAME                      = 4,
    /**
     * This indicates that event could not be added to event group because 
     * the event is from a different domain than the existing events in the
     * event group.
     */
    CUPTI_ERROR_EVENT_NOT_ADDED_DIFFERENT_DOMAIN        = 5,
    /**
     * This indicates that event could not be added to event group because 
     * the new event cannot be grouped with the existing events in the 
     * eventgroup due to hardware limitations.
     */
    CUPTI_ERROR_EVENT_NOT_ADDED_NOT_COMPATIBLE          = 6,
    /**
     * This indicates that event could not be added to event group because 
     * the eventgroup already has maximum events that can be profiled together.
     */
    CUPTI_ERROR_EVENT_NOT_ADDED_MAX_LIMIT_REACHED       = 7,
    /**
     * This indicates that an unknown internal error has occurred.
     */
    CUPTI_ERROR_UNKNOWN                                 = 8,
    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid CUDA device.
     */
    CUPTI_ERROR_INVALID_DEVICE                          = 9,
    /**
     * This indicates that the context passed to the API call is not a 
     * valid handle
     */
    CUPTI_ERROR_INVALID_CONTEXT                         = 10,
    /**
     * This indicates that an error occured while cupti attempted 
     * to initialize its connection to the cuda driver.
     */
    CUPTI_ERROR_NOT_INITIALIZED                         = 11,
    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    CUPTI_ERROR_OUT_OF_MEMORY                           = 13,
    /**
     * This is returned by the GetAttribute APIs if the input buffer size is 
     * not sufficient to return all requested data. 
     */
    CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT           = 14,
    /**
     * This indicates the specific API is not implemented. 
     */
    CUPTI_ERROR_API_NOT_IMPLEMENTED                     = 15,
    /**
     * This indicates that either the performance monitoring hardware could  
     * not be reserved or some other hardware error occurred.
     */
    CUPTI_ERROR_HARDWARE                                = 16,
    CUPTI_ERROR_FORCE_INT                               = 0x7fffffff,
} CUptiResult;


/**
 * Attributes queryable for a device
 */
typedef enum CUpti_DeviceAttribute_enum {
    CUPTI_DEVICE_ATTR_MAX_EVENT_ID        = 1,  /**< Maximum event ids for a device */
    CUPTI_DEVICE_ATTR_MAX_EVENT_DOMAIN_ID = 2,  /**< Maximum event domain ids for a device */
    CUPTI_DEVICE_ATTR_FORCE_INT           = 0x7fffffff,    
} CUpti_DeviceAttribute;


/**
 * Attributes queryable for an event domain
 */
typedef enum CUpti_EventDomainAttribute_enum {
    CUPTI_EVENT_DOMAIN_ATTR_NAME           = 0, /**< Event domain name, value is a null terminated const c-string */
    CUPTI_EVENT_DOMAIN_ATTR_INSTANCE_COUNT = 1, /**< Number of instances of the domain, value is an integer */
    CUPTI_EVENT_DOMAIN_MAX_EVENTS          = 2, /**< Max events available in this domain, value is an integer */
    CUPTI_EVENT_DOMAIN_ATTR_FORCE_INT      = 0x7fffffff,    
} CUpti_EventDomainAttribute;

/**
 * Attributes queryable for an event
 */
typedef enum CUpti_EventAttribute_enum {
    CUPTI_EVENT_ATTR_NAME              = 0,     /**< Event name, value is a null terminated const c-string */
    CUPTI_EVENT_ATTR_SHORT_DESCRIPTION = 1,     /**< Short description of event, value is a null terminated const c-string */
    CUPTI_EVENT_ATTR_LONG_DESCRIPTION  = 2,     /**< Long description of event, value is a null terminated const c-string */
    CUPTI_EVENT_ATTR_FORCE_INT         = 0x7fffffff,    
} CUpti_EventAttribute;

/**
 * Attributes for an event group some are read-only, others are read/write where noted.
 */
typedef enum CUpti_EventGroupAttribute_enum {
    CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID              = 0, 
    /**< 
     * [rw] The domain to which the event group is bound, set by the first event added. 
     * May be set on a new event group prior to adding any events to limit which events may be added.
     */
    CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES = 1, 
    /**< 
     * [rw] Profile all the instances of the domain for this eventgroup. This feature can be used to 
     * get load balancing across all instances of a domain 
     */
    CUPTI_EVENT_GROUP_ATTR_USER_DATA                    = 2, 
    /**< 
     * [rw] opaque user data 
     */
    CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS                   = 3, 
    /**< 
     * [ro] Number of events present in the group 
     */
    CUPTI_EVENT_GROUP_ATTR_FORCE_INT                    = 0x7fffffff,    
} CUpti_EventGroupAttribute;

/**
 * Flags for counter reading APIs
 */
typedef enum CUpti_ReadEventFlags_enum {
    CUPTI_EVENT_READ_FLAG_NONE          = 0,   /**< Default value */
    CUPTI_EVENT_READ_FLAG_ZERO_COUNTERS = 0x1, /**< Causes counters to be set to zero after read */
    CUPTI_EVENT_READ_FLAG_FORCE_INT     = 0x7fffffff,    
} CUpti_ReadEventFlags;

/** @} */ /* END CUPTI_COUNTER_TYPES */


/**
 * \defgroup CUPTI_EVENTS_API Event Management
 *
 * This section describes the Event Management APIs for CUDA profiler tools interface
 *
 * @{
 */

/**
 * \brief Returns the requested attribute for a device.
 *
 * Returns the requested attribute \p attrib in the variable pointed to by \p value for a device \p device.
 *
 * \param device  - in_param   CUDA device to query
 * \param attrib  - in_param   attribute to query
 * \param value   - out_param  value for requested attribute
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_NOT_INITIALIZED,
 * ::CUPTI_ERROR_INVALID_DEVICE,
 * ::CUPTI_ERROR_INVALID_PARAMETER
 *
 *
 */
CUptiResult CUPTIAPI cuptiDeviceGetAttribute(CUdevice device, 
                                             CUpti_DeviceAttribute attrib, 
                                             uint64_t *value);

/**
 * \brief Returns a timestamp for the device in nanoseconds since reset.
 *
 * Returns the timestamp in \p timestamp in nanoseconds since reset for the 
 * device. A context handle \p context needs to be passed to get the device 
 * timestamp.
 *
 * \param context    - in_param  CUDA context to query
 * \param timestamp  - out_param  timestamp value
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_CONTEXT,
 * ::CUPTI_ERROR_INVALID_PARAMETER
 *
 *
 */
CUptiResult CUPTIAPI cuptiDeviceGetTimestamp(CUcontext context, 
                                             uint64_t *timestamp);

/**
 * \brief Returns number of domains for a device.
 *
 * Returns the number of domains in \p numdomains for a device \p device.
 *
 * \param device     - in_param  CUDA device to query
 * \param numdomains - out_param requested number of domains
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_DEVICE,
 * ::CUPTI_ERROR_INVALID_PARAMETER
 *
 *
 */
CUptiResult CUPTIAPI cuptiDeviceGetNumEventDomains(CUdevice device, 
                                                   uint32_t *numdomains);

/**
 * \brief Enumerates the event domains for a device.
 *
 * Enumerates the event domains ids in \p domainArray for a device \p device. 
 * User has to allocate the buffer for \p domainArray and specify the 
 * size of buffer in \p arraySizeBytes. Recommended size of buffer is 
 * numdomains * sizeof(CUpti_EventDomainID). The output value in \p arraySizeBytes
 * contains the valid bytes available in the array. 
 *
 *
 * \param device         - in_param     CUDA device to query
 * \param arraySizeBytes - in_out_param size of domainArray in bytes
 * \param domainArray    - out_param    array of event domain ids
 *
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_DEVICE,
 * ::CUPTI_ERROR_INVALID_PARAMETER
 *
 *
 */
CUptiResult CUPTIAPI cuptiDeviceEnumEventDomains(CUdevice device, 
                                                 size_t *arraySizeBytes, 
                                                 CUpti_EventDomainID *domainArray);

/**
 * \brief Returns attribute for a event domain.
 *
 * Returns the requested attribute \p attrib in \p value for a event domain  
 * \p eventDomain for a device \p device. User has to allocate the buffer for 
 * \p value and specify the size of buffer in \p attribSize. The output value 
 * in \p attribSize contains the valid bytes available in the \p value. 
 *
 *
 * \param device        - in_param     CUDA device to query
 * \param eventDomain   - in_param     eventdomanin id for which attribute is queried
 * \param attrib        - in_param     attribute to query
 * \param attribSize    - in_out_param size of value in bytes
 * \param value         - out_param    value of requested attribute
 *
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_DEVICE,
 * ::CUPTI_ERROR_INVALID_PARAMETER,
 * ::CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID,
 * ::CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT
 *
 *
 */
CUptiResult CUPTIAPI cuptiEventDomainGetAttribute(CUdevice device, 
                                                  CUpti_EventDomainID eventDomain, 
                                                  CUpti_EventDomainAttribute attrib, 
                                                  size_t *attribSize, 
                                                  void *value);

/**
 * \brief Returns number of events for a domain.
 *
 * Returns the number of events in \p numevents for a domain \p eventDomain 
 * for a device \p device. 
 *
 *
 * \param device        - in_param     CUDA device to query
 * \param eventDomain   - in_param     eventdomanin id for which number of events is queried
 * \param numevents     - out_param    number of events supported in the domain
 *
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_DEVICE,
 * ::CUPTI_ERROR_INVALID_PARAMETER,
 * ::CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID
 *
 *
 */
CUptiResult CUPTIAPI cuptiEventDomainGetNumEvents(CUdevice device, 
                                                  CUpti_EventDomainID eventDomain, 
                                                  uint32_t *numevents);

/**
 * \brief Enumerates events for a domain.
 *
 * Enumerates the event ids in \p eventArray for a device \p device. 
 * User has to allocate the buffer for \p eventArray and specify the 
 * size of buffer in \p arraySizeBytes. Recommended size of buffer is 
 * numevents * sizeof(CUpti_EventID). The output value in \p arraySizeBytes
 * contains the valid bytes available in the array. 
 *
 *
 * \param device         - in_param      CUDA device to query
 * \param eventDomain    - in_param      eventdomain id for which number of events are enumerated
 * \param arraySizeBytes - in_out_param  size of eventArray in bytes
 * \param eventArray     - out_param     array of event ids
 *
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_DEVICE,
 * ::CUPTI_ERROR_INVALID_PARAMETER,
 * ::CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID
 *
 *
 */
CUptiResult CUPTIAPI cuptiEventDomainEnumEvents(CUdevice device, 
                                                CUpti_EventDomainID eventDomain, 
                                                size_t *arraySizeBytes, 
                                                CUpti_EventID *eventArray);

/**
 * \brief Returns attribute for an event.
 *
 * Returns the requested attribute \p attrib in \p value for an event   
 * \p eventID for a device \p device. User has to allocate the buffer for 
 * \p value and specify the size of buffer in \p attribSize. The output value 
 * in \p attribSize contains the valid bytes available in the \p value. 
 *
 *
 * \param device        - in_param     CUDA device to query
 * \param eventID       - in_param     event id for which attribute is queried
 * \param attrib        - in_param     attribute to query
 * \param attribSize    - in_out_param size of value in bytes
 * \param value         - out_param    requested attribute
 *
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_DEVICE,
 * ::CUPTI_ERROR_INVALID_PARAMETER,
 * ::CUPTI_ERROR_INVALID_EVENT_ID,
 * ::CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT
 *
 *
 */
CUptiResult CUPTIAPI cuptiEventGetAttribute(CUdevice device, 
                                            CUpti_EventID eventID, 
                                            CUpti_EventAttribute attrib, 
                                            size_t *attribSize, 
                                            void *value);

/**
 * \brief Returns event id for event name.
 *
 * Returns the event id in \p evenID for an event name \p eventName for   
 * a device \p device. 
 *
 *
 * \param device        - in_param     CUDA device to query
 * \param eventName     - in_param     event name for which eventid is queried
 * \param eventID       - out_param    event id 
 *
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_DEVICE,
 * ::CUPTI_ERROR_INVALID_PARAMETER,
 * ::CUPTI_ERROR_INVALID_EVENT_NAME
 *
 *
 */
CUptiResult CUPTIAPI cuptiEventGetIdFromName(CUdevice device, 
                                             const char *eventName, 
                                             CUpti_EventID *eventID);

/**
 * \brief Creates event group for a context.
 *
 * Creates event group for a context \p context and returns handle in \p eventGroup. 
 * \p flags must be 0. 
 *
 * \param context       - in_param     Context for which eventgroup is created
 * \param eventGroup    - out_param    Handle for eventgroup
 * \param flags         - in_param     flags should be 0
 *
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_CONTEXT,
 * ::CUPTI_ERROR_INVALID_PARAMETER,
 * ::CUPTI_ERROR_OUT_OF_MEMORY
 *
 *
 */
CUptiResult CUPTIAPI cuptiEventGroupCreate(CUcontext context, 
                                           CUpti_EventGroup *eventGroup, 
                                           uint32_t flags);

/**
 * \brief Destroys an event group.
 *
 * Destroys an event group \p eventGroup freeing its resources. Eventgroup 
 * will not be destroyed if it is enabled, it will return an error 
 * ::CUPTI_ERROR_UNKNOWN.
 *
 * \param eventGroup    - in_param    Handle for eventgroup
 *
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_PARAMETER,
 * ::CUPTI_ERROR_UNKNOWN
 *
 */
CUptiResult CUPTIAPI cuptiEventGroupDestroy(CUpti_EventGroup eventGroup);


/**
 * \brief Returns attribute for an event group.
 *
 * Returns the requested attribute \p attrib in \p value for an event group   
 * \p eventGroup. 
 *
 * \param eventGroup    - in_param     event group to query
 * \param attrib        - in_param     attribute to query
 * \param value         - out_param    value for requested attribute
 *
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_PARAMETER
 *
 */
CUptiResult CUPTIAPI cuptiEventGroupGetAttribute(CUpti_EventGroup eventGroup, 
                                                 CUpti_EventGroupAttribute attrib, 
                                                 uint64_t *value);

/**
 * \brief Sets attribute for an event group.
 *
 * Sets the value \p value for requested attribute \p attrib for the event group 
 * \p eventGroup
 *
 * \param eventGroup    - in_param     event group for which attribute is to be set
 * \param attrib        - in_param     attribute to set
 * \param value         - out_param    attribute value to set
 *
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_PARAMETER
 *
 *
 */
CUptiResult CUPTIAPI cuptiEventGroupSetAttribute(CUpti_EventGroup eventGroup, 
                                                 CUpti_EventGroupAttribute attrib, 
                                                 uint64_t value);


/**
 * \brief Adds event to event group.
 *
 * Attempts to add the event \p eventID to the event group \p eventGroup. 
 * Event cannot be added if the eventgroup is enabled.
 *
 * \param eventGroup    - in_param     event group to which the event is to be added
 * \param eventID       - in_param     event to be added
 *
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_EVENT_ID,
 * ::CUPTI_ERROR_EVENT_NOT_ADDED_DIFFERENT_DOMAIN,
 * ::CUPTI_ERROR_EVENT_NOT_ADDED_NOT_COMPATIBLE,
 * ::CUPTI_ERROR_EVENT_NOT_ADDED_MAX_LIMIT_REACHED,
 * ::CUPTI_ERROR_INVALID_PARAMETER,
 * ::CUPTI_ERROR_UNKNOWN
 *
 *
 */
CUptiResult CUPTIAPI cuptiEventGroupAddEvent(CUpti_EventGroup eventGroup, 
                                             CUpti_EventID eventID);

/**
 * \brief Removes an event from the event group.
 *
 * Removes the event \p eventID from the event group \p eventGroup. Event 
 * cannot be removed if the eventgroup is enabled.
 *
 * \param eventGroup    - in_param     event group from which the event is to be removed
 * \param eventID       - in_param     event to be removed
 *
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_EVENT_ID,
 * ::CUPTI_ERROR_INVALID_PARAMETER,
 * ::CUPTI_ERROR_UNKNOWN
 *
 *
 */
CUptiResult CUPTIAPI cuptiEventGroupRemoveEvent(CUpti_EventGroup eventGroup, 
                                                CUpti_EventID eventID);


/**
 * \brief Removes all events from the event group.
 *
 * Removes all events from the event group \p eventGroup. Event group 
 * should not be enabled while removing events.
 *
 * \param eventGroup    - in_param     event group from which events are to be removed
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_PARAMETER,
 * ::CUPTI_ERROR_UNKNOWN
 *
 *
 */
CUptiResult CUPTIAPI cuptiEventGroupRemoveAllEvents(CUpti_EventGroup eventGroup);


/**
 * \brief Resets all events in the event group to zero. 
 *
 * Resets all events from the event group \p eventGroup to 0.
 *
 * \param eventGroup    - in_param     event group from which the events are to be reset
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_PARAMETER
 *
 *
 */
CUptiResult CUPTIAPI cuptiEventGroupResetAllEvents(CUpti_EventGroup eventGroup);

/**
 * \brief Enables the event group. 
 *
 * Enables event group \p eventGroup. This api reserves the perfmon hardware  
 * for the current context if it is not already reserved, configures the events  
 * in event group, and starts collection of performance events. The counter 
 * values are reset while enabling the eventgroup.
 *
 *
 * \param eventGroup    - in_param     event group to be enabled
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_PARAMETER,
 * ::CUPTI_ERROR_HARDWARE
 *
 *
 */
CUptiResult CUPTIAPI cuptiEventGroupEnable(CUpti_EventGroup eventGroup);

/**
 * \brief Disables the event group. 
 *
 * Disables an event group \p eventGroup. This api frees the perfmon 
 * hardware if this is the last eventgroup to be disabled for the current  
 * context and stops collection of performance events. Counters will preserve
 * the values until they are explicitly reset or eventgroup is enabled again. 
 * 
 * 
 * \param eventGroup    - in_param     event group to be disabled
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_PARAMETER
 *
 *
 */
CUptiResult CUPTIAPI cuptiEventGroupDisable(CUpti_EventGroup eventGroup);

/**
 * \brief Reads counter data for the specified event ID value from the event group.  
 *
 * Reads counter value in \p counterData for the event \p eventID from the event
 * group \p eventGroup. 
 * - \p flags specifies the action to be taken for the counter values.  
 * - Input value of \p bufferSizeBytes indicates the size allocated
 *   by user, output contains the valid bytes available in array.
 *   A buffer of size = (uint64) should be allocated if 
 *   ::CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES is not set.
 *   A buffer of size = (uint64 * no of domain instances) should be allocated
 *   if ::CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES is set to get the 
 *   value for the event for all domain instances.
 * 
 * \param eventGroup      - in_param        Event group to read counter data from
 * \param flags           - in_param        Flags modifying the reading of the counters
 * \param eventID         - in_param        The event ID of the counter to read
 * \param bufferSizeBytes - in_out_param    The size of the counter storage buffer in bytes
 * \param counterData     - out_param       Pointer to a buffer for storage of the value of the event's counter
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_PARAMETER
 *
 *
 */
CUptiResult CUPTIAPI cuptiEventGroupReadEvent(CUpti_EventGroup       eventGroup, 
                                              CUpti_ReadEventFlags   flags, 
                                              CUpti_EventID          eventID, 
                                              size_t                 *bufferSizeBytes,
                                              uint64_t               *counterData);

/**
 * \brief Reads counter data for all the events from the event group.  
 *
 * Reads counter values in \p counterDataBuffer for all the events from the 
 * event group \p eventGroup. 
 * - \p flags specifies the action to be taken for  
 *   counter values. 
 * - Input value of \p bufferSizeBytes indicates the size 
 *   allocated by user, output contains the valid bytes available in array.
 *   A buffer of size = (uint64 * no of counters in the event group) should be 
 *   allocated if ::CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES is not set.
 *   A buffer of size = (uint64 * no of counters * no of domain instances) 
 *   should be allocated if ::CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES 
 *   is set to get the value for all events for all domain instances..
 *   Values will be arranged as follows:
 *    - instance 0: event0 event 1 ... event N
 *    - instance 1: event0 event 1 ... event N
 *    - ... 
 *    - instance M: event0 event 1 ... event N
 * - The counter values in \p counterDatabuffer should be read for events in 
 *   order in which events are specified in \p eventIDArray. This order does not
 *   change unless new events are added and events are removed from the event 
 *   group. 
 * - The buffer for \p eventIDArray should be allocated by user and the size 
 *   should be specified in \p arraySizeBytes. 
 * - For \p arraySizeBytes, input value indicates the size allocated by user, 
 *   output contains the valid bytes available in array. The size of buffer should
 *   be (sizeof(CUpti_EventID) * no of counters in the event group). 
 * - The actual number of counters returned in eventIDArray will be speicified in 
 *   \p numCountersRead.
 * 
 * 
 * \param eventGroup        - in_param        Event group to read counter data from
 * \param flags             - in_param        Flags modifying the reading of the counters
 * \param bufferSizeBytes   - in_out_param    The size of the counter storage buffer in bytes
 * \param counterDataBuffer - out_param       Pointer to a buffer for storage of the value of
 *                                            the event's counter. 
 * \param arraySizeBytes    - in_out_param    The size of the even id array in bytes. 
 * \param eventIDArray      - out_param       An array of event IDs in corresponding order to 
 *                                            the written counter values.
 * \param numCountersRead   - out_param       The number of counters actually read into the data buffer
 *
 * \return
 * ::CUPTI_SUCCESS,
 * ::CUPTI_ERROR_INVALID_PARAMETER
 *
 *
 */
CUptiResult CUPTIAPI cuptiEventGroupReadAllEvents(CUpti_EventGroup       eventGroup, 
                                                  CUpti_ReadEventFlags   flags,
                                                  size_t                 *bufferSizeBytes,
                                                  uint64_t               *counterDataBuffer,
                                                  size_t                 *arraySizeBytes,
                                                  CUpti_EventID          *eventIDArray,
                                                  size_t                 *numCountersRead);

/** @} */ /* END CUPTI_EVENTS_API */

/** @} */ /* END CUPTI_API */

#if defined(__cplusplus)
}
#endif 

#endif /*_CUPTI_EVENTS_H_*/


