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
 * @brief Special tests for code blocks and nested parallel regions/loops.

 */

#include <stdio.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

int main() {
  int i, j;
  int k = 0;
 
  #pragma omp parallel
 {
    #pragma omp for
   for (i=0; i<4; ++i)
     {
       k++;
     }
 }

  #pragma omp parallel
    #pragma omp for
 for (i=0; i<4; ++i)
   {
     k++;
   }

  #pragma omp parallel
 {
    #pragma omp master
   for (i=0; i<4; ++i)
     k++;
 }

  #pragma omp parallel
 {
    #pragma omp single
   for (i=0; i<4; ++i) k++;
 }

  #pragma omp parallel
    #pragma omp critical
 for (i=0; i<4; ++i) k++;

 // *****************************************
 // * Testing of nested parallelism         *
 // *****************************************

#pragma omp parallel 
 {
#pragma omp parallel 
   {
#pragma omp parallel 
     {
#pragma omp for nowait
       for (i=0; i<4; ++i) {
	 printf("do %d\n", i);
       }
     }
   }
 }

 // *****************************************
 // * Testing of nested for loops           *
 // *****************************************

  #pragma omp parallel
 {
    #pragma omp for nowait
   for (i=0; i<4; ++i) {
     for (j=0; j<4; ++j) {
       printf("do %d\n", i);
     }
   }

    #pragma omp for nowait
   for (i=0; i<4; ++i) {
     for (j=0; j<4; ++j) {
       for (k=0; k<4; ++k) {
	 printf("do %d\n", i);
       }
     }
   }

    #pragma omp for nowait
   for (i=0; i<4; ++i)
     for (j=0; j<4; ++j)
       printf("do %d\n", i);

    #pragma omp for nowait
   for (i=0; i<4; ++i)
     for (j=0; j<4; ++j)
       for (k=0; k<4; ++k)
	 printf("do %d\n", i);

    #pragma omp for nowait
   for (i=0; i<4; ++i)
     for (j=0; j<4; ++j) {
       printf("do %d\n", i);
     }
 }
}
