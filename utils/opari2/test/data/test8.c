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
 * @brief Tests whether specific clauses are found and inserted into the CTC string.
 */

#include <stdio.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

int j;
#pragma omp threadprivate(j)

int
main()
{
    int i = 5;
    int k = 0;

#pragma omp parallel if(k==0) num_threads(4) reduction(+:k)
    {
        printf( "parallel\n" );

#pragma omp for reduction(+:k) schedule(dynamic, 5 ) collapse(1)
        for ( i = 0; i < 4; ++i )
        {
            printf( "for %d\n", i );
            k++;
        }

#pragma omp sections reduction(+:k)
        {
     #pragma omp section
            printf( "section 1\n" );
     #pragma omp section
            { printf( "section 2\n" );
            }
        }
    }

#pragma omp parallel default(private)
    {
#pragma omp task if(true) untied
        {
            printf( "task\n" );
        }
    }

#pragma omp parallel shared(num_threads)
    {
	printf("num_threads variable is %d\n",num_threads);
    }
}

#pragma omp parallel for  num_threads(4) reduction(+:k) schedule(static,chunkif) collapse(1) ordered if(1) default(none)
for ( i = 0; i < 4; ++i )
{
     #pragma omp ordered
    printf( "for %d\n", i );

    k++;
}

#pragma omp parallel sections if((i+k)>5) num_threads(4) reduction(+:k)
{
     #pragma omp section
    printf( "section 1\n" );
     #pragma omp section
    { printf( "section 2\n" );
    }
}
