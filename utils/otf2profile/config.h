/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2011,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2011,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2011, 2013,
 * Technische Universitaet Dresden, Germany
 *
 * Copyright (c) 2009-2011,
 * University of Oregon, Eugene, USA
 *
 * Copyright (c) 2009-2011,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2009-2011,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2009-2011,
 * Technische Universitaet Muenchen, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license.  See the COPYING file in the package base
 * directory for details.
 *
 */

#ifndef CONFIG_H
#define CONFIG_H

/** @internal
 *  @file
 *
 *  @brief      Provide a single config.h that hides the frontend/backend
 *              issues from the developer.
 */

#if defined CROSS_BUILD
    #if defined FRONTEND_BUILD
        #include <config-frontend.h>
        #include <config-backend-for-frontend.h>
        #define HAVE_BACKEND( H ) ( defined( HAVE_BACKEND_ ## H ) && HAVE_BACKEND_ ## H )
    #elif defined BACKEND_BUILD
        #include <config-backend.h>
        #define HAVE_BACKEND( H ) ( defined( HAVE_ ## H ) && HAVE_ ## H )
    #else
        #error "You cannot use config.h without defining either FRONTEND_BUILD or BACKEND_BUILD."
    #endif

#elif defined NOCROSS_BUILD
    #include <config-backend.h>

#else
    #error "You cannot use config.h without defining either CROSS_BUILD or NOCROSS_BUILD."
#endif

#include <config-common.h>

#include <config-custom.h>

#endif /* CONFIG_H */
