/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2011,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2011,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2011,
 * Technische Universitaet Dresden, Germany
 *
 * Copyright (c) 2009-2011,
 * University of Oregon, Eugene, USA
 *
 * Copyright (c) 2009-2011, 2014
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2009-2011,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2009-2011,
 * Technische Universitaet Muenchen, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license. See the COPYING file in the package base
 * directory for details.
 *
 * Testfile for automated testing of OPARI2
 *
 *
 * @brief Tests user instrumentation directives and selective instrumentation.
 */

int main() {
  int i;
  int b = 1;

#pragma pomp inst init

#pragma pomp inst off

#pragma pomp inst begin(foo)

#pragma omp parallel
  i = 1;

#pragma pomp inst on

#pragma pomp noinstrument
#pragma omp parallel
#pragma omp for
  for ( i = 0; i < 2; ++i)
    b = b + i;

#pragma omp parallel for
  for ( i = 0; i < 2; ++i)
    b = b + i;
#pragma pomp instrument

#pragma omp parallel
  i = 3;

  if (b) {
#pragma pomp inst altend(foo)
    return 0;
  }

#pragma omp parallel
  {
#pragma pomp inst begin(phase1)
    i = 4;
#pragma omp barrier
    i = 5;
#pragma pomp inst end(phase1)
  }

  if (b) {
#pragma pomp inst altend(foo)
    return 0;
  }

#pragma pomp inst end(foo)

#pragma pomp inst finalize

  return 0;
}
