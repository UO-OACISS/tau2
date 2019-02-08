/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2011,
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

#ifndef CONFIG_H
#define CONFIG_H

/**
 * @file       config.h
 *
 * @brief      Provide a single config.h that hides the frontend/backend
 *             issues from the developer
 *
 */

#if defined CROSS_BUILD
    #if defined FRONTEND_BUILD
        #include <config-frontend.h>
    #elif defined BACKEND_BUILD
        #include <config-backend.h>
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
