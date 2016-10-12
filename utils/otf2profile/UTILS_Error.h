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

#ifndef UTILS_ERROR_H
#define UTILS_ERROR_H

/**
 * @file            UTILS_Error.h
 * @ingroup         UTILS_Exception_module
 *
 * @brief           Module for error handling in SCOREP.
 *
 */

#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif
#include <stdarg.h>

/*
 * Include the package specific error codes.
 */
#include PACKAGE_ERROR_CODES_HEADER

/*
 * The package specific name for error codes.
 *
 * This should be private, but to make this header readable is here defined.
 */
#define PACKAGE_ErrorCode PACKAGE_MANGLE_NAME( ErrorCode )

UTILS_BEGIN_C_DECLS

/**
 * @def UTILS_ERROR
 * This is a prep function, which delegates error information to the error
 * callback.
 * @ingroup UTILS_Exception_module
 */
#define UTILS_ERROR( errCode, ... ) \
    UTILS_Error_Handler( AFS_PACKAGE_SRCDIR, \
                         __FILE__, \
                         __LINE__, \
                         __func__, \
                         errCode, \
                         __VA_ARGS__ )

/**
 * Emit a warning.
 */
#define UTILS_WARNING( ... ) \
    UTILS_Error_Handler( AFS_PACKAGE_SRCDIR, \
                         __FILE__, \
                         __LINE__, \
                         __func__, \
                         PACKAGE_MANGLE_NAME( WARNING ), \
                         __VA_ARGS__ )
/**
 * Emit a warning, but only on first occurrence
 */
#define UTILS_WARN_ONCE( ... ) \
    do { \
        static int utils_warn_once_##__LINE__; \
        if ( !utils_warn_once_##__LINE__ ) \
        { \
            utils_warn_once_##__LINE__ = 1; \
            UTILS_WARNING( __VA_ARGS__ ); \
        } \
    } while ( 0 )

/**
 * Inform the user about not yet implemented functions by printing the function name and the source file.
 */
#define UTILS_NOT_YET_IMPLEMENTED() UTILS_WARN_ONCE( "Not yet implemented" )

/**
 * Use this to print a deprecation message once on wirst usage of the deprecated
 * entity.
 */
#define UTILS_DEPRECATED( ... ) \
    do { \
        static int utils_deprecated_##__LINE__; \
        if ( !utils_deprecated_##__LINE__ ) \
        { \
            utils_deprecated_##__LINE__ = 1; \
            UTILS_Error_Handler( AFS_PACKAGE_SRCDIR, \
                                 __FILE__, \
                                 __LINE__, \
                                 __func__, \
                                 PACKAGE_MANGLE_NAME( DEPRECATED ), \
                                 __VA_ARGS__ ); \
        } \
    } while ( 0 )

/**
 * Delegation error handler function, which is used by the prep UTILS_ERROR to
 * to avert that external programmers use the function pointer directly.
 *
 * @param function        : Name of the function where the error appeared
 * @param file            : Name of the source-code file where the error appeared
 * @param line            : Line number in the source-code where the error appeared
 * @param errorCode       : Error Code
 * @param msgFormatString : Format string like it is used at the printf family.
 *
 * @returns Should return the ErrorCode
 * @ingroup UTILS_Exception_module
 */
#define UTILS_Error_Handler PACKAGE_MANGLE_NAME( UTILS_Error_Handler )
PACKAGE_ErrorCode
UTILS_Error_Handler( const char*       srcdir,
                     const char*       file,
                     uint64_t          line,
                     const char*       function,
                     PACKAGE_ErrorCode errorCode,
                     const char*       msgFormatString,
                     ... );

/**
 * @def UTILS_ERROR_POSIX
 * This is a prep function, which is able to handle external POSIX
 * error codes with the SCOREP error handling system.
 *
 * @param ... The first argument needs to be a string constant.
 *
 * @ingroup UTILS_Exception_module
 */
#define UTILS_ERROR_POSIX( ... ) \
    UTILS_ERROR( UTILS_Error_FromPosix( errno ), "POSIX: " __VA_ARGS__ )

/**
 * Translates a POSIX error code into a SCOREP error code.
 *
 * @param posixErrorCode : Error Code
 *
 * @returns Returns a UTILS error code (see the package depended ErrorCodes.h)
 * @ingroup UTILS_Exception_module
 */
#define UTILS_Error_FromPosix PACKAGE_MANGLE_NAME( UTILS_Error_FromPosix )
PACKAGE_ErrorCode
UTILS_Error_FromPosix( const int posixErrorCode );

#define HAVE_UTILS_NO_ASSERT UTILS_JOIN_SYMS( HAVE_, PACKAGE_MANGLE_NAME( NO_ASSERT ) )

/**
 * @def UTILS_ASSERT
 * Definition of the utils assertion macro. It evaluates an @a expression. If it is false,
 * an error message is output and the program is aborted. To use the assertion,
 * HAVE_PACKAGE_NO_ASSERT must not be defined.
 * @param expression A logical expression which should be verified. If it is zero the
 *                    assertion fails.
 */
#if !HAVE( UTILS_NO_ASSERT )

#define UTILS_ASSERT( expression ) \
    ( ( expression ) ? ( void )0 : \
      UTILS_Error_Abort( AFS_PACKAGE_SRCDIR, \
                         __FILE__, \
                         __LINE__, \
                         __func__, \
                         "Assertion '" UTILS_STRINGIFY( expression ) "' failed" ) )

#else

#define UTILS_ASSERT( expression ) do { ( void )( expression ); } while ( 0 )

#endif

/**
 * @def UTILS_FATAL
 *
 * Indicates a fatal condition. The program will abort immediately.
 *
 */
#define UTILS_FATAL( ... ) \
    UTILS_Error_Abort( AFS_PACKAGE_SRCDIR, \
                       __FILE__, \
                       __LINE__, \
                       __func__, \
                       __VA_ARGS__ )


/**
 * @def UTILS_BUG
 *
 * Indicates an error in the software, not induced by wrong usage of an user.
 * The program will abort immediately.
 *
 */
#define UTILS_BUG( ... ) \
    UTILS_Error_Abort( AFS_PACKAGE_SRCDIR, \
                       __FILE__, \
                       __LINE__, \
                       __func__, \
                       "Bug: " __VA_ARGS__ )


/**
 * @def UTILS_BUG_ON
 *
 * Same as UTILS_BUG but with an condition.
 *
 */
#define UTILS_BUG_ON( expression, ... ) \
    ( !( expression ) ? ( void )0 :     \
      UTILS_Error_Abort( AFS_PACKAGE_SRCDIR, \
                         __FILE__,      \
                         __LINE__,      \
                         __func__,      \
                         "Bug '" #expression "': " __VA_ARGS__ ) )

/**
 * This function implements the UTILS_ASSERT, UTILS_FATAL, UTILS_BUG, UTILS_BUG_ON macro.
 * It prints an error message and aborts the program. Do not insert calls to this function
 * directly, but use the macros instead.
 *  @param fileName   The file name of the file which contains the source code where the
 *                    message was created.
 *  @param line       The line number of the source code line where the debug message
 *                    was created.
 *  @param functionName  A string containing the name of the function where the debug
 *                       message was called.
 *  @param messageFormatString A printf-like format string.
 */
#define UTILS_Error_Abort PACKAGE_MANGLE_NAME( UTILS_Error_Abort )
void
UTILS_Error_Abort( const char* srcdir,
                   const char* fileName,
                   uint64_t    line,
                   const char* functionName,
                   const char* messageFormatString,
                   ... );


UTILS_END_C_DECLS

#endif /* UTILS_ERROR_H */
