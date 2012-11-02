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

#ifndef CONFIG_COMMON_H
#define CONFIG_COMMON_H


/**
 * @file       config-common.h
 * @maintainer Christian R&ouml;ssel <c.roessel@fz-juelich.de>
 * @maintainer Bert Wesarg <Bert.Wesarg@tu-dresden.de>
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

#define PACKAGE_MANGLE_NAME( sym )      UTILS_JOIN_SYMS( PACKAGE_SYM, _ ## sym )
#define PACKAGE_MANGLE_NAME_CAPS( sym ) UTILS_JOIN_SYMS( PACKAGE_SYM_CAPS, _ ## sym )


#endif /* CONFIG_COMMON_H */
