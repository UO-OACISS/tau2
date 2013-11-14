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


#ifndef CONFIG_CUSTOM_H
#define CONFIG_CUSTOM_H


/** @internal
 *  @file       config-custom.h
 *  @authors    Christian R&ouml;ssel <c.roessel@fz-juelich.de>,
 *              Bert Wesarg <Bert.Wesarg@tu-dresden.de>
 *  @maintainer Christian R&ouml;ssel <c.roessel@fz-juelich.de>
 *              Bert Wesarg <Bert.Wesarg@tu-dresden.de>
 *
 *  @brief      This file gets included by config.h (resp. config-frontend.h and
 *              config-backend.h) and contains supplementary macros to be used
 *              with the macros in config.h.
 */


#define UTILS_DEBUG_MODULES \
    UTILS_DEFINE_DEBUG_MODULE( ARCHIVE,      0 ), \
    UTILS_DEFINE_DEBUG_MODULE( READER,       1 ), \
    UTILS_DEFINE_DEBUG_MODULE( ANCHOR_FILE,  2 )


#endif /* CONFIG_CUSTOM_H */
