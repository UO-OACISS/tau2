/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2012,
 *    RWTH Aachen University, Germany
 *    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *    Technische Universitaet Dresden, Germany
 *    University of Oregon, Eugene, USA
 *    Forschungszentrum Juelich GmbH, Germany
 *    German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *    Technische Universitaet Muenchen, Germany
 *
 * See the COPYING file in the package base directory for details.
 *
 */

#ifndef OTF2_ERROR_CODES_H
#define OTF2_ERROR_CODES_H


/**
 * @file            OTF2_ErrorCodes.h
 * @maintainer      Daniel Lorenz <d.lorenz@fz-juelich.de>
 * @status          ALPHA
 * @ingroup         OTF2_Exception_module
 *
 * @brief           Error codes and error handling.
 *
 * @author          Dominic Eschweiler <d.eschweiler@fz-juelich.de>
 */

#include <errno.h>
#include <stdint.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * This is the list of error codes for OTF2.
 */
typedef enum
{
    /** Special marker for error messages which indicates an deprecation. */
    OTF2_DEPRECATED    = -3,

    /** Special marker when the application will be aborted. */
    OTF2_ABORT         = -2,

    /** Special marker for error messages which are only warnings. */
    OTF2_WARNING       = -1,

    /** Operation successful */
    OTF2_SUCCESS       = 0,

    /** Invalid error code
     *
     * Should only be used internally and not as an actual error code.
     */
    OTF2_ERROR_INVALID = 1,

    /* These are the internal implementation of POSIX error codes. */
    /** The list of arguments is to long */
    OTF2_ERROR_E2BIG,
    /** Not enough rights */
    OTF2_ERROR_EACCES,
    /** Address is not available */
    OTF2_ERROR_EADDRNOTAVAIL,
    /** Address family is not supported */
    OTF2_ERROR_EAFNOSUPPORT,
    /** Resource temporaly not available */
    OTF2_ERROR_EAGAIN,
    /** Connection is already processed */
    OTF2_ERROR_EALREADY,
    /** Invalid file pointer */
    OTF2_ERROR_EBADF,
    /** Invalid message */
    OTF2_ERROR_EBADMSG,
    /** Resource or device is busy */
    OTF2_ERROR_EBUSY,
    /** Operation was aborted */
    OTF2_ERROR_ECANCELED,
    /** No child process available */
    OTF2_ERROR_ECHILD,
    /** Connection was refused */
    OTF2_ERROR_ECONNREFUSED,
    /** Connection was reset */
    OTF2_ERROR_ECONNRESET,
    /** Resolved deadlock */
    OTF2_ERROR_EDEADLK,
    /** Destination address was expected */
    OTF2_ERROR_EDESTADDRREQ,
    /** Domain error */
    OTF2_ERROR_EDOM,
    /** Reserved */
    OTF2_ERROR_EDQUOT,
    /** File does already exist */
    OTF2_ERROR_EEXIST,
    /** Invalid Address */
    OTF2_ERROR_EFAULT,
    /** File is to big */
    OTF2_ERROR_EFBIG,
    /** Operation is work in progress */
    OTF2_ERROR_EINPROGRESS,
    /** Interuption of an operating system call */
    OTF2_ERROR_EINTR,
    /** Invalid argument */
    OTF2_ERROR_EINVAL,
    /** Generic I/O error */
    OTF2_ERROR_EIO,
    /** Socket is already connected */
    OTF2_ERROR_EISCONN,
    /** Target is a directory */
    OTF2_ERROR_EISDIR,
    /** To many layers of symbolic links */
    OTF2_ERROR_ELOOP,
    /** To many opened files */
    OTF2_ERROR_EMFILE,
    /** To many links */
    OTF2_ERROR_EMLINK,
    /** Message buffer is to small */
    OTF2_ERROR_EMSGSIZE,
    /** Reserved */
    OTF2_ERROR_EMULTIHOP,
    /** Filename is to long */
    OTF2_ERROR_ENAMETOOLONG,
    /** Network is down */
    OTF2_ERROR_ENETDOWN,
    /** Connection was reset from the network */
    OTF2_ERROR_ENETRESET,
    /** Network is not reachable */
    OTF2_ERROR_ENETUNREACH,
    /** To much opened files */
    OTF2_ERROR_ENFILE,
    /** No buffer space available */
    OTF2_ERROR_ENOBUFS,
    /** No more data left in the queue */
    OTF2_ERROR_ENODATA,
    /** This device does not support this function */
    OTF2_ERROR_ENODEV,
    /** File or Directory does not exist */
    OTF2_ERROR_ENOENT,
    /** Can not execute binary */
    OTF2_ERROR_ENOEXEC,
    /** Locking failed */
    OTF2_ERROR_ENOLCK,
    /** Reserved */
    OTF2_ERROR_ENOLINK,
    /** Not enough main memory available */
    OTF2_ERROR_ENOMEM,
    /** Message has not the expected type */
    OTF2_ERROR_ENOMSG,
    /** This protocol is not available */
    OTF2_ERROR_ENOPROTOOPT,
    /** No space left on device */
    OTF2_ERROR_ENOSPC,
    /** No stream available */
    OTF2_ERROR_ENOSR,
    /** This is not a stream */
    OTF2_ERROR_ENOSTR,
    /** Requested function is not implemented */
    OTF2_ERROR_ENOSYS,
    /** Socket is not connected */
    OTF2_ERROR_ENOTCONN,
    /** This is not an directory */
    OTF2_ERROR_ENOTDIR,
    /** This directory is not empty */
    OTF2_ERROR_ENOTEMPTY,
    /** No socket */
    OTF2_ERROR_ENOTSOCK,
    /** This operation is not supported */
    OTF2_ERROR_ENOTSUP,
    /** This IOCTL is not supported by the device */
    OTF2_ERROR_ENOTTY,
    /** Device is not yet configured */
    OTF2_ERROR_ENXIO,
    /** Operation is not supported by this socket */
    OTF2_ERROR_EOPNOTSUPP,
    /** Value is to long for the datatype */
    OTF2_ERROR_EOVERFLOW,
    /** Operation is not permitted */
    OTF2_ERROR_EPERM,
    /** Broken pipe */
    OTF2_ERROR_EPIPE,
    /** Protocoll error */
    OTF2_ERROR_EPROTO,
    /** Protocoll is not supported */
    OTF2_ERROR_EPROTONOSUPPORT,
    /** Wrong protocoll type for this socket */
    OTF2_ERROR_EPROTOTYPE,
    /** Value is out of range */
    OTF2_ERROR_ERANGE,
    /** Filesystem is read only */
    OTF2_ERROR_EROFS,
    /** This seek is not allowed */
    OTF2_ERROR_ESPIPE,
    /** No matching process found */
    OTF2_ERROR_ESRCH,
    /** Reserved */
    OTF2_ERROR_ESTALE,
    /** Timout in file stream or IOCTL */
    OTF2_ERROR_ETIME,
    /** Connection timed out */
    OTF2_ERROR_ETIMEDOUT,
    /** File couldn't be executed while it is opened */
    OTF2_ERROR_ETXTBSY,
    /** Operation would be blocking */
    OTF2_ERROR_EWOULDBLOCK,
    /** Invalid link between devices */
    OTF2_ERROR_EXDEV,

    /* These are the error codes specific to the OTF2 package */

    /** Unintentional reached end of function */
    OTF2_ERROR_END_OF_FUNCTION,
    /** Function call not allowed in current state */
    OTF2_ERROR_INVALID_CALL,
    /** Parameter value out of range */
    OTF2_ERROR_INVALID_ARGUMENT,
    /** Invalid definition or event record */
    OTF2_ERROR_INVALID_RECORD,
    /** Invalid or inconsistent record data */
    OTF2_ERROR_INVALID_DATA,
    /** The given size can not be used */
    OTF2_ERROR_INVALID_SIZE_GIVEN,
    /** The given type is not known */
    OTF2_ERROR_UNKNOWN_TYPE,
    /** The structural integrity is not given */
    OTF2_ERROR_INTEGRITY_FAULT,
    /** This could not be done with the given memory */
    OTF2_ERROR_MEM_FAULT,
    /** Memory allocation failed */
    OTF2_ERROR_MEM_ALLOC_FAILED,
    /** An error appeared when data was processed */
    OTF2_ERROR_PROCESSED_WITH_FAULTS,
    /** Index out of bounds */
    OTF2_ERROR_INDEX_OUT_OF_BOUNDS,
    /** Invalid source code line number */
    OTF2_ERROR_INVALID_LINENO,
    /** End of buffer/file reached */
    OTF2_ERROR_END_OF_BUFFER,
    /** Invalid file operation */
    OTF2_ERROR_FILE_INTERACTION,
    /** Unable to open file */
    OTF2_ERROR_FILE_CAN_NOT_OPEN,
    /** Record reading interrupted by reader callback */
    OTF2_ERROR_INTERRUPTED_BY_CALLBACK,
    /** Property name does not conform to the naming scheme */
    OTF2_ERROR_PROPERTY_NAME_INVALID,
    /** Property already exists */
    OTF2_ERROR_PROPERTY_EXISTS,
    /** Property not found found in this archive */
    OTF2_ERROR_PROPERTY_NOT_FOUND,
    /** Property value does not have the expected value */
    OTF2_ERROR_PROPERTY_VALUE_INVALID,
    /** Missing library support for requested compression mode */
    OTF2_ERROR_FILE_COMPRESSION_NOT_SUPPORTED,
    /** Multiple definitions for the same mapping type */
    OTF2_ERROR_DUPLICATE_MAPPING_TABLE
} OTF2_ErrorCode;


/**
 * Returns the name of an error code.
 *
 * @param errorCode : Error Code
 *
 * @returns Returns the name of a known error code, and "INVALID_ERROR" for
 *          invalid or unknown error IDs.
 * @ingroup OTF2_Exception_module
 */
const char*
OTF2_Error_GetName( OTF2_ErrorCode errorCode );


/**
 * Returns the description of an error code.
 *
 * @param errorCode : Error Code
 *
 * @returns Returns the description of a known error code.
 * @ingroup OTF2_Exception_module
 */
const char*
OTF2_Error_GetDescription( OTF2_ErrorCode errorCode );


/**
 * Signature of error handler callback functions. The error handler can be set
 * with @ref OTF2_Error_RegisterCallback.
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
typedef OTF2_ErrorCode
( *OTF2_ErrorCallback )( void*          userData,
                         const char*    file,
                         uint64_t       line,
                         const char*    function,
                         OTF2_ErrorCode errorCode,
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
 * @ingroup OTF2_Exception_module
 *
 */
OTF2_ErrorCallback
OTF2_Error_RegisterCallback( OTF2_ErrorCallback errorCallbackIn,
                             void*              userData );


#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OTF2_ERROR_CODES_H */
