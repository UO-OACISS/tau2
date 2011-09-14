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


/**
 * @file       config-custom.h
 * @author     Christian R&ouml;ssel <c.roessel@fz-juelich.de>
 * @author     Bert Wesarg <Bert.Wesarg@tu-dresden.de>
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
 * 1, you should use <tt>\#if HAVE(DECL_MPI_ACCUMULATE)</tt> in your code
 * to conditionally compile if @c HAVE_DECL_MPI_ACCUMULATE is defined to 1.
 */
#define HAVE( H ) ( defined( HAVE_##H ) && HAVE_##H )

#endif /* CONFIG_CUSTOM_H */
