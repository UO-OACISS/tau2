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
 * @brief Test the basic instrumentation of all directives.
 */

#include <stdio.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

int j;
#pragma omp threadprivate(j)

int main() {
  int i;
  int k = 0;

 #pragma omp parallel
 {
   printf("parallel\n");

   #pragma omp for
   for(i=0; i<4; ++i) {
     printf("for %d\n", i);
     k++;
   }

   #pragma omp flush(k)

   #pragma omp barrier

   #pragma omp for ordered
   for(i=0; i<4; ++i) {
     #pragma omp ordered
     {
       printf("for %d\n", i);
     }
   }

   #pragma omp sections
   {
     #pragma omp section
     printf("section 1\n");
     #pragma omp section
     { printf("section 2\n"); }
   }

   #pragma omp master
   {
     printf("master\n");
   }

   #pragma omp critical
   {
     printf("critical\n");
   }

   #pragma omp critical(foobar)
   {
     printf("critical(foobar)\n");
   }

   #pragma omp atomic
   /* -------------- */
   /* do this atomic */
   i += 1;
   /* -------------- */

   #pragma omp single
   {
     printf("single\n");
   }
 }

 #pragma omp parallel
 {
   #pragma omp task
   {
     printf("task\n");
   }

   #pragma omp taskwait
 }

 // #pragma omp this should be ignored by opari and the compiler
 // #pragma this too
}
