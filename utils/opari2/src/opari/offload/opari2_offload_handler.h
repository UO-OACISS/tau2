/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2013,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license. See the COPYING file in the package base
 * directory for details.
 *
 */

/** @internal
 *
 *  @file      opari2_offload_handler.h
 *
 *  @brief
 */


#ifndef OPARI2_OFFLOAD_HANDLER_H
#define OPARI2_OFFLOAD_HANDLER_H


#include <iostream>
using std::ostream;

#include "opari2.h"
#include "opari2_directive.h"


void
h_offload_target( OPARI2_Directive* d,
                  ostream&          os );

void
h_end_offload_target( OPARI2_Directive* d,
                      ostream&          os );

void
h_offload_declspec( OPARI2_Directive* d,
                    ostream&          os );

void
h_end_offload_declspec( OPARI2_Directive* d,
                        ostream&          os );

#endif
