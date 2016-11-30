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

#ifndef CONFIG_COMMON_H
#define CONFIG_COMMON_H


/**
 * @file       config-common.h
 *
 * @brief This file gets included by config.h (resp. config-frontend.h and
 * config-backend.h) and contains supplementary macros to be used with the
 * macros in config.h.
 *
 */


/**
 * Conditionally compile on macro values that are either 0 or 1.
 *
 * E.g. if you have the macro @c HAVE_DECL_MPI_ACCUMULATE that is either 0 or
 * 1, you should use <tt>#if HAVE(DECL_MPI_ACCUMULATE)</tt> in your code
 * to conditionally compile if @c HAVE_DECL_MPI_ACCUMULATE is defined to 1.
 */
#define HAVE( H ) ( defined( HAVE_ ## H ) && HAVE_ ## H )


/**
 * Use these macros in internal headers to mark symbols as extern "C"
 *
 * @{
 */

#ifdef __cplusplus
#  define UTILS_BEGIN_C_DECLS extern "C" {
#  define UTILS_END_C_DECLS   }
#else
#  define UTILS_BEGIN_C_DECLS
#  define UTILS_END_C_DECLS
#endif

/**
 * @}
 */


/* Macros used by utilities under vendor/common */

#define UTILS_STRINGIFY_( x ) #x
#define UTILS_STRINGIFY( x )  UTILS_STRINGIFY_( x )

#define UTILS_JOIN_SYMS_( x, y )   x ## y
#define UTILS_JOIN_SYMS( x, y )    UTILS_JOIN_SYMS_( x, y )

#define UTILS_JOIN_3SYMS_( x, y, z )  x ## y ## z
#define UTILS_JOIN_3SYMS( x, y, z )   UTILS_JOIN_3SYMS_( x, y, z )

#define PACKAGE_MANGLE_name( name ) UTILS_JOIN_SYMS( AFS_PACKAGE_name, _ ## name )
#define PACKAGE_MANGLE_NAME( name ) UTILS_JOIN_SYMS( AFS_PACKAGE_NAME, _ ## name )


/**
 * Macros to fool the linker, so that the named compilation unit will always be
 * linked into the library/binary
 * @ingroup fool_linker @{
 */

/** Use this macro in the top level scope of the compilation unit, which
 *  should always be linked into the library/binary.
 *  @param name A unique name of this compilation unit, must be a valid C symbol.
 */
#define UTILS_FOOL_LINKER_DECLARE( name ) \
    bool name ## _fool_linker = false

/** Use this macro in a function from a compilation unit, which is guaranteed
 *  to be always linked into an library/binary, so that the named compilation
 *  unit is also always linked into the library/binary.
 *
 *  @param name The unique name of the compilation unit, must be a valid C symbol.
 */
#define UTILS_FOOL_LINKER( name ) \
    extern bool name ## _fool_linker; \
    name ## _fool_linker = true

/**
 * @}
 */


#endif /* CONFIG_COMMON_H */
