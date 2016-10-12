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

#ifndef UTILS_INTERNAL_PACKAGE_H
#define UTILS_INTERNAL_PACKAGE_H

/**
 * @file            utils_package.h
 * @ingroup         UTILS_Exception_module
 *
 * @brief           Module for error handling in SCOREP.
 */

#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif
#include <stdarg.h>

#include <UTILS_Error.h>

/*
 * The angle brackets need to tighly enclose the header name, else
 * the additional spaces are taken into account for the file name
 */
/* *INDENT-OFF* */
#define PACKAGE_INCLUDE_( f ) <f>
/* *INDENT-ON* */

#define PACKAGE_INCLUDE( f )  PACKAGE_INCLUDE_( PACKAGE_MANGLE_name( f ) )

/* Package dependent symbols, are defined in the packages ErrorCodes.h header */
#define PACKAGE_DEPRECATED             PACKAGE_MANGLE_NAME( DEPRECATED )
#define PACKAGE_ABORT                  PACKAGE_MANGLE_NAME( ABORT )
#define PACKAGE_WARNING                PACKAGE_MANGLE_NAME( WARNING )
#define PACKAGE_SUCCESS                PACKAGE_MANGLE_NAME( SUCCESS )
#define PACKAGE_ERROR_INVALID          PACKAGE_MANGLE_NAME( ERROR_INVALID )
#define PACKAGE_Error_GetName          PACKAGE_MANGLE_NAME( Error_GetName )
#define PACKAGE_Error_GetDescription   PACKAGE_MANGLE_NAME( Error_GetDescription )
#define PACKAGE_ErrorCallback          PACKAGE_MANGLE_NAME( ErrorCallback )
#define PACKAGE_Error_RegisterCallback PACKAGE_MANGLE_NAME( Error_RegisterCallback )

/* Add here more convenient macros for error codes used in the utils sources */
#define PACKAGE_ERROR_END_OF_FUNCTION    PACKAGE_MANGLE_NAME( ERROR_END_OF_FUNCTION )
#define PACKAGE_ERROR_FILE_CAN_NOT_OPEN  PACKAGE_MANGLE_NAME( ERROR_FILE_CAN_NOT_OPEN )
#define PACKAGE_ERROR_MEM_ALLOC_FAILED   PACKAGE_MANGLE_NAME( ERROR_MEM_ALLOC_FAILED )
#define PACKAGE_ERROR_END_OF_BUFFER      PACKAGE_MANGLE_NAME( ERROR_END_OF_BUFFER )
#define PACKAGE_ERROR_FILE_INTERACTION   PACKAGE_MANGLE_NAME( ERROR_FILE_INTERACTION )

#endif /* UTILS_INTERNAL_PACKAGE_H */
