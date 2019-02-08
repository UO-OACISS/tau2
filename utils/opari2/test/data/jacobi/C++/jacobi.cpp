/*
 ******************************************************************
 * Subroutine HelmholtzJ
 * Solves poisson equation on rectangular grid assuming :
 * (1) Uniform discretization in each direction, and
 * (2) Dirichlect boundary conditions
 *
 * Jacobi method is used in this routine
 *
 * Input : n,m   Number of grid points in the X/Y directions
 *         dx,dy Grid spacing in the X/Y directions
 *         alpha Helmholtz eqn. coefficient
 *         omega Relaxation factor
 *         f(n,m) Right hand side function
 *         u(n,m) Dependent variable/Solution
 *         tolerance Tolerance for iterative solver
 *         maxit  Maximum number of iterations
 *
 * Output : u(n,m) - Solution
 *****************************************************************
 */
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "jacobi.h"

using namespace std;

#define U( j, i ) afU[ ( ( j ) - data.iRowFirst ) * data.iCols + ( i ) ]
#define F( j, i ) afF[ ( ( j ) - data.iRowFirst ) * data.iCols + ( i ) ]
#define UOLD( j, i ) uold[ ( ( j ) - data.iRowFirst ) * data.iCols + ( i ) ]


void
Jacobi( JacobiData& data )
{
    /*use local pointers for performance reasons*/
    double* afU, * afF;
    afU = data.afU;
    afF = data.afF;
    double* uold = new double[ data.iCols * data.iRows ];

    if ( uold )
    {
        double ax       = 1.0 / ( data.fDx * data.fDx );    /* X-direction coef */
        double ay       = 1.0 / ( data.fDy * data.fDy );    /* Y_direction coef */
        double b        = -2.0 * ( ax + ay ) - data.fAlpha; /* Central coeff */
        double residual = 10.0 * data.fTolerance;

        while ( data.iIterCount < data.iIterMax&& residual > data.fTolerance )
        {
            residual = 0.0;

            /* Copy new solution into old, excluding the boundary. */
#pragma omp parallel
            {
#pragma omp for
                for ( int j = data.iRowFirst; j <= data.iRowLast; j++ )
                {
                    for ( int i = 0; i < data.iCols; i++ )
                    {
                        UOLD( j, i ) = U( j, i );
                    }
                }

                double fLRes;

                /* Compute stencil, residual and update.
                 * Update excludes the boundary */
#pragma omp for reduction(+:residual)
                for ( int j = data.iRowFirst + 1; j <= data.iRowLast - 1; j++ )
                {
                    for ( int i = 1; i < data.iCols - 1; i++ )
                    {
                        fLRes = ( ax * ( UOLD( j, i - 1 ) + UOLD( j, i + 1 ) )
                                  + ay * ( UOLD( j - 1, i ) + UOLD( j + 1, i ) )
                                  +  b * UOLD( j, i ) - F( j, i ) ) / b;

                        /* update solution */
                        U( j, i ) = UOLD( j, i ) - data.fRelax * fLRes;

                        /* accumulate residual error */
                        residual += fLRes * fLRes;
                    }
                }
            } /* end omp parallel */

            /* error check */
            data.iIterCount++;
            residual = sqrt( residual ) / ( data.iCols * data.iRows );
        } /* while */

        data.fResidual = residual;
        delete[] uold;
    }
    else
    {
        cerr << "Error: cant allocate memory\n";
        Finish( data );
        exit( 1 );
    }
}
