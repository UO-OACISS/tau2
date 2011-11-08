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
 * Testfile for automated testing of OPARI2
 *
 * @authors Bernd Mohr, Peter Philippen
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
  i = 2;
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
