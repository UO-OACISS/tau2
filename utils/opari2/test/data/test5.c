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
 *
 * @brief Test the splitting of combined parallel clauses.
 */

#include <stdio.h>

int t;
#pragma omp threadprivate(t)

int main() {
  int i, j, k, l;

  l=0;
  #pragma omp parallel         /* parallel */ \
      for                      /* for */ \
      lastprivate(k)           /* for */ \
      private(i,j),             /* parallel */    \
      lastprivate              /* for */ \
      (                        /* for */ \
        l                      /* for */ \
      ), schedule(dynamic      /* for */      \
      )
  for(i=0; i<4;++i) {
    printf("parallel for %d\n", i);
    k+=i;
  }

#pragma omp parallel sections if(k) num_threads(2) lastprivate(i) firstprivate(j) default(shared) copyin(t) reduction(+:l)
  {
    #pragma omp section
    {
      printf("Section 1\n");
    }
    #pragma omp section
    {
      printf("Section 2\n");
    }
    #pragma omp section
    {
      printf("Section 3\n");
    }
  }
}
