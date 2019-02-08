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
 * @brief Test the nowait and untied clauses
 */

#include <stdio.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

int main() {
  int i;

 #pragma omp parallel
 {
   printf("parallel\n");
   #pragma omp for nowait
   for(i=0; i<4; ++i) {
     printf("for nowait %d\n", i);
   }

   #pragma omp sections nowait
   {
     #pragma omp section
     printf("section nowait 1\n");
     #pragma omp section
     { printf("section nowait 2\n"); }
   }

   #pragma omp single nowait
   {
     printf("single nowait\n");
   }

   #pragma omp task untied
   {
     printf("task untied\n");
   }
 }
}
