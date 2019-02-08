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
 *  @file      opari2_pomp_handler.h
 *
 *  @brief
 */

#ifndef OPARI2_POMP_HANDLER_H
#define OPARI2_POMP_HANDLER_H

#include <iostream>
using std::ostream;


void
h_pomp_inst( OPARI2_Directive* ptr,
             ostream&          os );

void
h_pomp_instinit( OPARI2_Directive* ptr,
                 ostream&          os );

void
h_pomp_instfinalize( OPARI2_Directive* ptr,
                     ostream&          os );

void
h_pomp_inston( OPARI2_Directive* ptr,
               ostream&          os );

void
h_pomp_instoff( OPARI2_Directive* ptr,
                ostream&          os );


void
h_pomp_instrument( OPARI2_Directive* ptr,
                   ostream&          os );


void
h_pomp_noinstrument( OPARI2_Directive* ptr,
                     ostream&          os );


void
h_pomp_instbegin( OPARI2_Directive* ptr,
                  ostream&          os );


void
h_pomp_instaltend( OPARI2_Directive* ptr,
                   ostream&          os );

void
h_pomp_instend( OPARI2_Directive* ptr,
                ostream&          os );

void
h_end_pomp_instbegin( OPARI2_Directive* ptr,
                      ostream&          os );

#endif
