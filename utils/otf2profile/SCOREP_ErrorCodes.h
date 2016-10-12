/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2012,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2012,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2012, 2014,
 * Technische Universitaet Dresden, Germany
 *
 * Copyright (c) 2009-2012,
 * University of Oregon, Eugene, USA
 *
 * Copyright (c) 2009-2012,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2009-2012,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2009-2012,
 * Technische Universitaet Muenchen, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license.  See the COPYING file in the package base
 * directory for details.
 *
 */

#ifndef SCOREP_ERROR_CODES_H
#define SCOREP_ERROR_CODES_H


/**
 * @file            SCOREP_ErrorCodes.h
 * @ingroup         SCOREP_Exception_module
 *
 * @brief           Error codes and error handling.
 *
 */

#include <errno.h>
#include <stdint.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * This is the list of error codes for SCOREP.
 */
typedef enum
{
    /** Special marker for error messages which indicates an deprecation. */
    SCOREP_DEPRECATED    = -3,

    /** Special marker when the application will be aborted. */
    SCOREP_ABORT         = -2,

    /** Special marker for error messages which are only warnings. */
    SCOREP_WARNING       = -1,

    /** Operation successful */
    SCOREP_SUCCESS       = 0,

    /** Invalid error code
     *
     * Should only be used internally and not as an actual error code.
     */
    SCOREP_ERROR_INVALID = 1,

    /* These are the internal implementation of POSIX error codes. */
    /** The list of arguments is to long */
    SCOREP_ERROR_E2BIG,
    /** Not enough rights */
    SCOREP_ERROR_EACCES,
    /** Address is not available */
    SCOREP_ERROR_EADDRNOTAVAIL,
    /** Address family is not supported */
    SCOREP_ERROR_EAFNOSUPPORT,
    /** Resource temporaly not available */
    SCOREP_ERROR_EAGAIN,
    /** Connection is already processed */
    SCOREP_ERROR_EALREADY,
    /** Invalid file pointer */
    SCOREP_ERROR_EBADF,
    /** Invalid message */
    SCOREP_ERROR_EBADMSG,
    /** Resource or device is busy */
    SCOREP_ERROR_EBUSY,
    /** Operation was aborted */
    SCOREP_ERROR_ECANCELED,
    /** No child process available */
    SCOREP_ERROR_ECHILD,
    /** Connection was refused */
    SCOREP_ERROR_ECONNREFUSED,
    /** Connection was reset */
    SCOREP_ERROR_ECONNRESET,
    /** Resolved deadlock */
    SCOREP_ERROR_EDEADLK,
    /** Destination address was expected */
    SCOREP_ERROR_EDESTADDRREQ,
    /** Domain error */
    SCOREP_ERROR_EDOM,
    /** Reserved */
    SCOREP_ERROR_EDQUOT,
    /** File does already exist */
    SCOREP_ERROR_EEXIST,
    /** Invalid Address */
    SCOREP_ERROR_EFAULT,
    /** File is to big */
    SCOREP_ERROR_EFBIG,
    /** Operation is work in progress */
    SCOREP_ERROR_EINPROGRESS,
    /** Interuption of an operating system call */
    SCOREP_ERROR_EINTR,
    /** Invalid argument */
    SCOREP_ERROR_EINVAL,
    /** Generic I/O error */
    SCOREP_ERROR_EIO,
    /** Socket is already connected */
    SCOREP_ERROR_EISCONN,
    /** Target is a directory */
    SCOREP_ERROR_EISDIR,
    /** To many layers of symbolic links */
    SCOREP_ERROR_ELOOP,
    /** To many opened files */
    SCOREP_ERROR_EMFILE,
    /** To many links */
    SCOREP_ERROR_EMLINK,
    /** Message buffer is to small */
    SCOREP_ERROR_EMSGSIZE,
    /** Reserved */
    SCOREP_ERROR_EMULTIHOP,
    /** Filename is to long */
    SCOREP_ERROR_ENAMETOOLONG,
    /** Network is down */
    SCOREP_ERROR_ENETDOWN,
    /** Connection was reset from the network */
    SCOREP_ERROR_ENETRESET,
    /** Network is not reachable */
    SCOREP_ERROR_ENETUNREACH,
    /** To much opened files */
    SCOREP_ERROR_ENFILE,
    /** No buffer space available */
    SCOREP_ERROR_ENOBUFS,
    /** No more data left in the queue */
    SCOREP_ERROR_ENODATA,
    /** This device does not support this function */
    SCOREP_ERROR_ENODEV,
    /** File or Directory does not exist */
    SCOREP_ERROR_ENOENT,
    /** Cannot execute binary */
    SCOREP_ERROR_ENOEXEC,
    /** Locking failed */
    SCOREP_ERROR_ENOLCK,
    /** Reserved */
    SCOREP_ERROR_ENOLINK,
    /** Not enough main memory available */
    SCOREP_ERROR_ENOMEM,
    /** Message has not the expected type */
    SCOREP_ERROR_ENOMSG,
    /** This protocol is not available */
    SCOREP_ERROR_ENOPROTOOPT,
    /** No space left on device */
    SCOREP_ERROR_ENOSPC,
    /** No stream available */
    SCOREP_ERROR_ENOSR,
    /** This is not a stream */
    SCOREP_ERROR_ENOSTR,
    /** Requested function is not implemented */
    SCOREP_ERROR_ENOSYS,
    /** Socket is not connected */
    SCOREP_ERROR_ENOTCONN,
    /** This is not an directory */
    SCOREP_ERROR_ENOTDIR,
    /** This directory is not empty */
    SCOREP_ERROR_ENOTEMPTY,
    /** No socket */
    SCOREP_ERROR_ENOTSOCK,
    /** This operation is not supported */
    SCOREP_ERROR_ENOTSUP,
    /** This IOCTL is not supported by the device */
    SCOREP_ERROR_ENOTTY,
    /** Device is not yet configured */
    SCOREP_ERROR_ENXIO,
    /** Operation is not supported by this socket */
    SCOREP_ERROR_EOPNOTSUPP,
    /** Value is to long for the datatype */
    SCOREP_ERROR_EOVERFLOW,
    /** Operation is not permitted */
    SCOREP_ERROR_EPERM,
    /** Broken pipe */
    SCOREP_ERROR_EPIPE,
    /** Protocoll error */
    SCOREP_ERROR_EPROTO,
    /** Protocoll is not supported */
    SCOREP_ERROR_EPROTONOSUPPORT,
    /** Wrong protocoll type for this socket */
    SCOREP_ERROR_EPROTOTYPE,
    /** Value is out of range */
    SCOREP_ERROR_ERANGE,
    /** Filesystem is read only */
    SCOREP_ERROR_EROFS,
    /** This seek is not allowed */
    SCOREP_ERROR_ESPIPE,
    /** No matching process found */
    SCOREP_ERROR_ESRCH,
    /** Reserved */
    SCOREP_ERROR_ESTALE,
    /** Timout in file stream or IOCTL */
    SCOREP_ERROR_ETIME,
    /** Connection timed out */
    SCOREP_ERROR_ETIMEDOUT,
    /** File couldn't be executed while it is opened */
    SCOREP_ERROR_ETXTBSY,
    /** Operation would be blocking */
    SCOREP_ERROR_EWOULDBLOCK,
    /** Invalid link between devices */
    SCOREP_ERROR_EXDEV,

    /* These are the error codes specific to the Score-P package */

    /** Unintentional reached end of function */
    SCOREP_ERROR_END_OF_FUNCTION,
    /** Function call not allowed in current state */
    SCOREP_ERROR_INVALID_CALL,
    /** Parameter value out of range */
    SCOREP_ERROR_INVALID_ARGUMENT,
    /** The given size cannot be used */
    SCOREP_ERROR_INVALID_SIZE_GIVEN,
    /** The given type is not known */
    SCOREP_ERROR_UNKNOWN_TYPE,
    /** The structural integrity is not given */
    SCOREP_ERROR_INTEGRITY_FAULT,
    /** This could not be done with the given memory */
    SCOREP_ERROR_MEM_FAULT,
    /** Memory allocation failed */
    SCOREP_ERROR_MEM_ALLOC_FAILED,
    /** An error appeared when data was processed */
    SCOREP_ERROR_PROCESSED_WITH_FAULTS,
    /** Index out of bounds */
    SCOREP_ERROR_INDEX_OUT_OF_BOUNDS,
    /** Fortran and C integer size mismatch */
    SCOREP_ERROR_F2C_INT_SIZE_MISMATCH,
    /** Unknown region type */
    SCOREP_ERROR_UNKNOWN_REGION_TYPE,
    /** Invalid source code line number */
    SCOREP_ERROR_INVALID_LINENO,
    /** End of buffer/file reached */
    SCOREP_ERROR_END_OF_BUFFER,
    /** Cannot find MPI window */
    SCOREP_ERROR_MPI_NO_WINDOW,
    /** Cannot find MPI communicator */
    SCOREP_ERROR_MPI_NO_COMM,
    /** Too many MPI windows are created */
    SCOREP_ERROR_MPI_TOO_MANY_WINDOWS,
    /** Too many MPI communicators are created */
    SCOREP_ERROR_MPI_TOO_MANY_COMMS,
    /** Too many MPI groups are created */
    SCOREP_ERROR_MPI_TOO_MANY_GROUPS,
    /** Cannot find MPI group */
    SCOREP_ERROR_MPI_NO_GROUP,
    /** Too many concurrent window accesses */
    SCOREP_ERROR_MPI_TOO_MANY_WINACCS,
    /** Cannot find window access entry */
    SCOREP_ERROR_MPI_NO_WINACC,
    /** Missing termination in tracking data structure. */
    SCOREP_ERROR_MPI_NO_LAST_REQUEST,
    /** Usage of an invalid region handle */
    SCOREP_ERROR_USER_INVALID_REGION,
    /** Usage of an invalid metric group handle */
    SCOREP_ERROR_USER_INVALID_MGROUP,
    /** Invalid file operation */
    SCOREP_ERROR_FILE_INTERACTION,
    /** Unable to open file */
    SCOREP_ERROR_FILE_CAN_NOT_OPEN,
    /** Error while parsing: Unknown token */
    SCOREP_ERROR_PARSE_UNKNOWN_TOKEN,
    /** Error while parsing: Unexpected end of string */
    SCOREP_ERROR_PARSE_UNEXPECTED_END,
    /** Error while parsing: Missing seperator */
    SCOREP_ERROR_PARSE_NO_SEPARATOR,
    /** Error while parsing: Missing key */
    SCOREP_ERROR_PARSE_NO_KEY,
    /** Error while parsing: Missing value */
    SCOREP_ERROR_PARSE_NO_VALUE,
    /** Error while parsing: Invalid value */
    SCOREP_ERROR_PARSE_INVALID_VALUE,
    /** Error while parsing: Syntax error */
    SCOREP_ERROR_PARSE_SYNTAX,
    /** Expect location format 'filename:lineNo1:lineNo2' */
    SCOREP_ERROR_POMP_SCL_BROKEN,
    /** Unknown schedule specified */
    SCOREP_ERROR_POMP_UNKNOWN_SCHEDULE,
    /** Invalid number of sections. Exspected > 0 sections */
    SCOREP_ERROR_POMP_INVALID_SECNUM,
    /** Name of user region missing */
    SCOREP_ERROR_POMP_NO_NAME,
    /** Inconsistent profile. Stop profiling */
    SCOREP_ERROR_PROFILE_INCONSISTENT,
    /** Profiling accessed uninitialized thread */
    SCOREP_ERROR_PROFILE_UNINITIALIZED_THREAD,
    /** Could not initialize PAPI library */
    SCOREP_ERROR_PAPI_INIT,
    /** No free memory page available */
    SCOREP_ERROR_MEMORY_OUT_OF_PAGES,
    /** Switching recording on or off in a parallel region is not allowed. Please consult the manual for information, where recording can be switched on/off */
    SCOREP_ERROR_SWITCH_IN_PARALLEL,
    /** Could not parse MRI request */
    SCOREP_ERROR_OA_PARSE_MRI,
    /** Error in execution of system command */
    SCOREP_ERROR_ON_SYSTEM_CALL,
    /** Unknown POMP region type specified in CTC string */
    SCOREP_ERROR_POMP_UNKNOWN_REGION_TYPE,
    /** Cannot open library handle */
    SCOREP_ERROR_DLOPEN_FAILED,
    /** Could not find symbol */
    SCOREP_ERROR_DLSYM_FAILED,
    /** Cannot close library handle */
    SCOREP_ERROR_DLCLOSE_FAILED,
    /** An error occurred while initializing Linux PERF infrastructure */
    SCOREP_ERROR_PERF_INIT
} SCOREP_ErrorCode;


/**
 * Returns the name of an error code.
 *
 * @param errorCode : Error Code
 *
 * @returns Returns the name of a known error code, and "INVALID_ERROR" for
 *          invalid or unknown error IDs.
 * @ingroup SCOREP_Exception_module
 */
const char*
SCOREP_Error_GetName( SCOREP_ErrorCode errorCode );


/**
 * Returns the description of an error code.
 *
 * @param errorCode : Error Code
 *
 * @returns Returns the description of a known error code.
 * @ingroup SCOREP_Exception_module
 */
const char*
SCOREP_Error_GetDescription( SCOREP_ErrorCode errorCode );


/**
 * Signature of error handler callback functions. The error handler can be set
 * with @ref SCOREP_Error_RegisterCallback.
 *
 * @param userData        : Data passed to this function as given by the registry call.
 * @param file            : Name of the source-code file where the error appeared
 * @param line            : Line number in the source-code where the error appeared
 * @param function        : Name of the function where the error appeared
 * @param errorCode       : Error Code
 * @param msgFormatString : Format string like it is used at the printf family.
 * @param va              : Variable argument list
 *
 * @returns Should return the errorCode
 */
typedef SCOREP_ErrorCode
( *SCOREP_ErrorCallback )( void*          userData,
                         const char*    file,
                         uint64_t       line,
                         const char*    function,
                         SCOREP_ErrorCode errorCode,
                         const char*    msgFormatString,
                         va_list        va );


/**
 * Register a programmers callback function for error handling.
 *
 * @param errorCallbackIn : Fucntion will be called instead of printing a
 *                          default message to standard error.
 * @param userData :        Data pointer passed to the callback.
 *
 * @returns Function pointer to the former error handling function.
 * @ingroup SCOREP_Exception_module
 *
 */
SCOREP_ErrorCallback
SCOREP_Error_RegisterCallback( SCOREP_ErrorCallback errorCallbackIn,
                             void*              userData );


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SCOREP_ERROR_CODES_H */
