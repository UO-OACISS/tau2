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


/**
 * @file            UTILS_Error.c
 * @ingroup         UTILS_Exception_module
 *
 * @brief           Module for error handling in PACKAGE
 */


#include <config.h>
#include <UTILS_Error.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <inttypes.h>

#include <UTILS_CStr.h>
#include <UTILS_IO.h>

#include <utils_package.h>

#include "normalize_file.h"


/*--- Internal functions -------------------------------------------*/

/**
 * Default callback function for error handling, which is used if the programmer
 * doesn't register an own callback with function PACKAGE_Error_RegisterCallback().
 *
 * @param function        : Name of the function where the error appeared
 * @param file            : Name of the source-code file where the error appeared
 * @param line            : Line number in the source-code where the error appeared
 * @param errorCode       : Error Code
 * @param msgFormatString : Format string like it is used at the printf family.
 * @param va              : Variable argument list
 *
 * @returns Should return the errorCode
 */

/**
 * This is a pointer to a function which handles errors if UTILS_ERROR(a) is
 * called.
 */
static PACKAGE_ErrorCallback utils_error_callback;
static void*                 utils_error_callback_user_data;


/*--- External visible functions ----------------------------------*/

static PACKAGE_ErrorCode
utils_error_handler_va( const char*       srcdir,
                        const char*       file,
                        const uint64_t    line,
                        const char*       function,
                        PACKAGE_ErrorCode errorCode,
                        const char*       msgFormatString,
                        va_list           va )
{
    const char* normalized_file = normalize_file( srcdir, file );

    if ( utils_error_callback )
    {
        errorCode = utils_error_callback( utils_error_callback_user_data,
                                          normalized_file,
                                          line,
                                          function,
                                          errorCode,
                                          msgFormatString,
                                          va );
    }
    else
    {
        size_t msg_format_string_length = msgFormatString ?
                                          strlen( msgFormatString ) : 0;
        const char* type               = "error";
        const char* description        = "";
        const char* description_prefix = "";

        if ( errorCode == PACKAGE_WARNING )
        {
            type = "warning";
        }
        else if ( errorCode == PACKAGE_DEPRECATED )
        {
            type = "deprecated";
        }
        else if ( errorCode == PACKAGE_ABORT )
        {
            type = "abort";
        }
        else
        {
            description        = PACKAGE_Error_GetDescription( errorCode );
            description_prefix = ": ";
        }

        fprintf( stderr, "[%s] %s:%" PRIu64 ": %s%s%s%s",
                 PACKAGE_NAME, normalized_file, line,
                 type, description_prefix, description,
                 msg_format_string_length ? ": " : "\n" );

        if ( msg_format_string_length )
        {
            vfprintf( stderr, msgFormatString, va );
            fprintf( stderr, "\n" );
        }
    }

    return errorCode;
}

PACKAGE_ErrorCode
UTILS_Error_Handler( const char*       srcdir,
                     const char*       file,
                     const uint64_t    line,
                     const char*       function,
                     PACKAGE_ErrorCode errorCode,
                     const char*       msgFormatString,
                     ... )
{
    if ( errorCode == PACKAGE_SUCCESS )
    {
        return errorCode;
    }

    va_list va;
    va_start( va, msgFormatString );

    errorCode = utils_error_handler_va( srcdir,
                                        file,
                                        line,
                                        function,
                                        errorCode,
                                        msgFormatString,
                                        va );


    va_end( va );

    return errorCode;
}

struct utils_error_decl
{
    const char*       errorName;
    const char*       errorDescription;
    PACKAGE_ErrorCode errorCode;
};


/* *INDENT-OFF* */
static const struct utils_error_decl none_error_decls[] =
{
    { "SUCCESS",    "Success",    PACKAGE_SUCCESS    },
    { "WARNING",    "Warning",    PACKAGE_WARNING    },
    { "ABORT",      "Aborting",   PACKAGE_ABORT      },
    { "DEPRECATED", "Deprecated", PACKAGE_DEPRECATED }
};
/* *INDENT-ON* */
static const size_t none_error_decls_size =
    sizeof( none_error_decls ) / sizeof( none_error_decls[ 0 ] );


/* *INDENT-OFF* */
static const struct utils_error_decl error_decls[] =
{
    #define _e( code, description ) \
    { \
        UTILS_STRINGIFY_( code ), \
        description, \
        PACKAGE_MANGLE_NAME( ERROR_ ## code ) \
    }

    /* This is the internal implementation of posix error code descriptions. */
    _e( E2BIG,           "The list of arguments is to long" ),
    _e( EACCES,          "Not enough rights" ),
    _e( EADDRNOTAVAIL,   "Address is not available" ),
    _e( EAFNOSUPPORT,    "Address family is not supported" ),
    _e( EAGAIN,          "Resource temporaly not available" ),
    _e( EALREADY,        "Connection is already processed" ),
    _e( EBADF,           "Invalid file pointer" ),
    _e( EBADMSG,         "Invalid message" ),
    _e( EBUSY,           "Resource or device is busy" ),
    _e( ECANCELED,       "Operation was aborted" ),
    _e( ECHILD,          "No child process available" ),
    _e( ECONNREFUSED,    "Connection was refused" ),
    _e( ECONNRESET,      "Connection was reset" ),
    _e( EDEADLK,         "Resolved deadlock" ),
    _e( EDESTADDRREQ,    "Destination address was expected" ),
    _e( EDOM,            "Domain error" ),
    _e( EDQUOT,          "Reserved" ),
    _e( EEXIST,          "File does already exist" ),
    _e( EFAULT,          "Invalid Address" ),
    _e( EFBIG,           "File is to big" ),
    _e( EINPROGRESS,     "Operation is work in progress" ),
    _e( EINTR,           "Interuption of an operating system call" ),
    _e( EINVAL,          "Invalid argument" ),
    _e( EIO,             "Generic I/O error" ),
    _e( EISCONN,         "Socket is already connected" ),
    _e( EISDIR,          "Target is a directory" ),
    _e( ELOOP,           "To many layers of symbolic links" ),
    _e( EMFILE,          "To many opened files" ),
    _e( EMLINK,          "To many links" ),
    _e( EMSGSIZE,        "Message buffer is to small" ),
    _e( EMULTIHOP,       "Reserved" ),
    _e( ENAMETOOLONG,    "Filename is to long" ),
    _e( ENETDOWN,        "Network is down" ),
    _e( ENETRESET,       "Connection was reset from the network" ),
    _e( ENETUNREACH,     "Network is not reachable" ),
    _e( ENFILE,          "To much opened files" ),
    _e( ENOBUFS,         "No buffer space available" ),
    _e( ENODATA,         "No more data left in the queue" ),
    _e( ENODEV,          "This device does not support this function" ),
    _e( ENOENT,          "File or Directory does not exist" ),
    _e( ENOEXEC,         "Cannot execute binary" ),
    _e( ENOLCK,          "Locking failed" ),
    _e( ENOLINK,         "Reserved" ),
    _e( ENOMEM,          "Not enough main memory available" ),
    _e( ENOMSG,          "Message has not the expected type" ),
    _e( ENOPROTOOPT,     "This protocol is not available" ),
    _e( ENOSPC,          "No space left on device" ),
    _e( ENOSR,           "No stream available" ),
    _e( ENOSTR,          "This is not a stream" ),
    _e( ENOSYS,          "Requested function is not implemented" ),
    _e( ENOTCONN,        "Socket is not connected" ),
    _e( ENOTDIR,         "This is not an directory" ),
    _e( ENOTEMPTY,       "This directory is not empty" ),
    _e( ENOTSOCK,        "No socket" ),
    _e( ENOTSUP,         "This operation is not supported" ),
    _e( ENOTTY,          "This IOCTL is not supported by the device" ),
    _e( ENXIO,           "Device is not yet configured" ),
    _e( EOPNOTSUPP,      "Operation is not supported by this socket" ),
    _e( EOVERFLOW,       "Value is to long for the datatype" ),
    _e( EPERM,           "Operation is not permitted" ),
    _e( EPIPE,           "Broken pipe" ),
    _e( EPROTO,          "Protocoll error" ),
    _e( EPROTONOSUPPORT, "Protocoll is not supported" ),
    _e( EPROTOTYPE,      "Wrong protocoll type for this socket" ),
    _e( ERANGE,          "Value is out of range" ),
    _e( EROFS,           "Filesystem is read only" ),
    _e( ESPIPE,          "This seek is not allowed" ),
    _e( ESRCH,           "No matching process found" ),
    _e( ESTALE,          "Reserved" ),
    _e( ETIME,           "Timout in file stream or IOCTL" ),
    _e( ETIMEDOUT,       "Connection timed out" ),
    _e( ETXTBSY,         "File couldn't be executed while it is opened" ),
    _e( EWOULDBLOCK,     "Operation would be blocking" ),
    _e( EXDEV,           "Invalid link between devices" ),

    #include PACKAGE_INCLUDE( error_decls.gen.h )

    #undef _e
};
/* *INDENT-ON* */
static size_t error_decls_size =
    sizeof( error_decls ) / sizeof( error_decls[ 0 ] );


/* *INDENT-OFF* */
static const struct
{
    PACKAGE_ErrorCode errorCode;
    int                posixErrno;
} posix_errno_delcs[] =
{
    #define _e( code ) \
    { \
        PACKAGE_MANGLE_NAME( ERROR_ ## code ), \
        code \
    }

#ifdef EACCES
    _e( EACCES ),           //  0
#endif
#ifdef EADDRNOTAVAIL
    _e( EADDRNOTAVAIL ),    //  1
#endif
#ifdef EAFNOSUPPORT
    _e( EAFNOSUPPORT ),     //  2
#endif
#ifdef EAGAIN
    _e( EAGAIN ),           //  3
#endif
#ifdef EALREADY
    _e( EALREADY ),         //  4
#endif
#ifdef EBADF
    _e( EBADF ),            //  5
#endif
#ifdef EBADMSG
    _e( EBADMSG ),          //  6
#endif
#ifdef EBUSY
    _e( EBUSY ),            //  7
#endif
#ifdef ECANCELED
    _e( ECANCELED ),        //  8
#endif
#ifdef ECHILD
    _e( ECHILD ),           //  9
#endif
#ifdef ECONNREFUSED
    _e( ECONNREFUSED ),     // 10
#endif
#ifdef ECONNRESET
    _e( ECONNRESET ),       // 11
#endif
#ifdef EDEADLK
    _e( EDEADLK ),          // 12
#endif
#ifdef EDESTADDRREQ
    _e( EDESTADDRREQ ),     // 13
#endif
#ifdef EDOM
    _e( EDOM ),             // 14
#endif
#ifdef EDQUOT
    _e( EDQUOT ),           // 15
#endif
#ifdef EEXIST
    _e( EEXIST ),           // 16
#endif
#ifdef EFAULT
    _e( EFAULT ),           // 17
#endif
#ifdef EFBIG
    _e( EFBIG ),            // 18
#endif
#ifdef EINPROGRESS
    _e( EINPROGRESS ),      // 19
#endif
#ifdef EINTR
    _e( EINTR ),            // 20
#endif
#ifdef EINVAL
    _e( EINVAL ),           // 21
#endif
#ifdef EIO
    _e( EIO ),              // 22
#endif
#ifdef EISCONN
    _e( EISCONN ),          // 23
#endif
#ifdef EISDIR
    _e( EISDIR ),           // 24
#endif
#ifdef ELOOP
    _e( ELOOP ),            // 25
#endif
#ifdef EMFILE
    _e( EMFILE ),           // 26
#endif
#ifdef EMLINK
    _e( EMLINK ),           // 27
#endif
#ifdef EMSGSIZE
    _e( EMSGSIZE ),         // 28
#endif
#ifdef EMULTIHOP
    _e( EMULTIHOP ),        // 29
#endif
#ifdef ENAMETOOLONG
    _e( ENAMETOOLONG ),     // 30
#endif
#ifdef ENETDOWN
    _e( ENETDOWN ),         // 31
#endif
#ifdef ENETRESET
    _e( ENETRESET ),        // 32
#endif
#ifdef ENETUNREACH
    _e( ENETUNREACH ),      // 33
#endif
#ifdef ENFILE
    _e( ENFILE ),           // 34
#endif
#ifdef ENOBUFS
    _e( ENOBUFS ),          // 35
#endif
#ifdef ENODATA
    _e( ENODATA ),          // 36
#endif
#ifdef ENODEV
    _e( ENODEV ),           // 37
#endif
#ifdef ENOENT
    _e( ENOENT ),           // 38
#endif
#ifdef ENOEXEC
    _e( ENOEXEC ),          // 39
#endif
#ifdef ENOLCK
    _e( ENOLCK ),           // 40
#endif
#ifdef ENOLINK
    _e( ENOLINK ),          // 41
#endif
#ifdef ENOMEM
    _e( ENOMEM ),           // 42
#endif
#ifdef ENOMSG
    _e( ENOMSG ),           // 43
#endif
#ifdef ENOPROTOOPT
    _e( ENOPROTOOPT ),      // 44
#endif
#ifdef ENOSPC
    _e( ENOSPC ),           // 45
#endif
#ifdef ENOSR
    _e( ENOSR ),            // 46
#endif
#ifdef ENOSTR
    _e( ENOSTR ),           // 47
#endif
#ifdef ENOSYS
    _e( ENOSYS ),           // 48
#endif
#ifdef ENOTCONN
    _e( ENOTCONN ),         // 49
#endif
#ifdef ENOTDIR
    _e( ENOTDIR ),          // 50
#endif
#ifdef ENOTEMPTY
    _e( ENOTEMPTY ),        // 51
#endif
#ifdef ENOTSOCK
    _e( ENOTSOCK ),         // 52
#endif
#ifdef ENOTSUP
    _e( ENOTSUP ),          // 53
#endif
#ifdef ENOTTY
    _e( ENOTTY ),           // 54
#endif
#ifdef ENXIO
    _e( ENXIO ),            // 55
#endif
#ifdef EOPNOTSUPP
    _e( EOPNOTSUPP ),       // 56
#endif
#ifdef EOVERFLOW
    _e( EOVERFLOW ),        // 57
#endif
#ifdef EPERM
    _e( EPERM ),            // 58
#endif
#ifdef EPIPE
    _e( EPIPE ),            // 59
#endif
#ifdef EPROTO
    _e( EPROTO ),           // 60
#endif
#ifdef EPROTONOSUPPORT
    _e( EPROTONOSUPPORT ),  // 61
#endif
#ifdef EPROTOTYPE
    _e( EPROTOTYPE ),       // 62
#endif
#ifdef ERANGE
    _e( ERANGE ),           // 63
#endif
#ifdef EROFS
    _e( EROFS ),            // 64
#endif
#ifdef ESPIPE
    _e( ESPIPE ),           // 65
#endif
#ifdef ESRCH
    _e( ESRCH ),            // 66
#endif
#ifdef ESTALE
    _e( ESTALE ),           // 67
#endif
#ifdef ETIME
    _e( ETIME ),            // 68
#endif
#ifdef ETIMEDOUT
    _e( ETIMEDOUT ),        // 69
#endif
#ifdef ETXTBSY
    _e( ETXTBSY ),          // 70
#endif
#ifdef EWOULDBLOCK
    _e( EWOULDBLOCK ),      // 71
#endif
#ifdef EXDEV
    _e( EXDEV ),            // 72
#endif

    #undef _e
};
/* *INDENT-ON* */
static size_t posix_errno_delcs_size =
    sizeof( posix_errno_delcs ) / sizeof( posix_errno_delcs[ 0 ] );

static const struct utils_error_decl*
utils_get_error_decl( PACKAGE_ErrorCode errorCode )
{
    if ( errorCode <= PACKAGE_SUCCESS )
    {
        if ( -errorCode >= none_error_decls_size )
        {
            return NULL;
        }
        return &none_error_decls[ -errorCode ];
    }

    /* real error codes start 1 behind PACKAGE_ERROR_INVALID */
    size_t index = errorCode - PACKAGE_ERROR_INVALID - 1;

    if ( errorCode == PACKAGE_ERROR_INVALID ||
         index >= error_decls_size )
    {
        return NULL;
    }
    return &error_decls[ index ];
}

const char*
PACKAGE_Error_GetName( const PACKAGE_ErrorCode errorCode )
{
    const struct utils_error_decl* decl = utils_get_error_decl( errorCode );
    if ( !decl )
    {
        return "INVALID";
    }

    /* See scorep_error_codes.h for the definition of this array */
    return decl->errorName;
}


const char*
PACKAGE_Error_GetDescription( const PACKAGE_ErrorCode errorCode )
{
    const struct utils_error_decl* decl = utils_get_error_decl( errorCode );
    if ( !decl )
    {
        return "Unknown error code";
    }

    /* See scorep_error_codes.h for the definition of this array */
    return decl->errorDescription;
}


PACKAGE_ErrorCode
UTILS_Error_FromPosix( const int posixErrno )
{
    uint64_t i;

    if ( posixErrno == 0 )
    {
        return PACKAGE_SUCCESS;
    }

    for ( i = 0; i < posix_errno_delcs_size; i++ )
    {
        if ( posix_errno_delcs[ i ].posixErrno == posixErrno )
        {
            return posix_errno_delcs[ i ].errorCode;
        }
    }

    return PACKAGE_ERROR_INVALID;
}

void
UTILS_Error_Abort( const char* srcdir,
                   const char* fileName,
                   uint64_t    line,
                   const char* functionName,
                   const char* messageFormatString,
                   ... )
{
    va_list va;
    va_start( va, messageFormatString );

    utils_error_handler_va( srcdir,
                            fileName,
                            line,
                            functionName,
                            PACKAGE_ABORT,
                            messageFormatString,
                            va );

    va_end( va );

    abort();
}


PACKAGE_ErrorCallback
PACKAGE_Error_RegisterCallback( PACKAGE_ErrorCallback errorCallbackIn,
                                void*                 userData )
{
    PACKAGE_ErrorCallback temp_pointer = utils_error_callback;
    utils_error_callback           = errorCallbackIn;
    utils_error_callback_user_data = userData;
    return temp_pointer;
}
