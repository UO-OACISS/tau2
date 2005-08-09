#include <iostream>
#include <stdlib.h>
#include <mpi.h>

#include <math.h>

// This example is adapted from "Using MPI, second edition" 
// by Gropp, Lusk, and Skellum

void initialize( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
}

void identifyRankAndSize( int& rank, int& size, MPI_Comm comm )
{
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &size);
}

void constructWorkerCommGroup( MPI_Comm& workerComm, int size )
{
    // construct the world group
    MPI_Group worldGroup;
    MPI_Comm_group( MPI_COMM_WORLD, &worldGroup );

    // construct the worker group
    MPI_Group workerGroup;
    int ranks[1];
    ranks[0] = size - 1; // exclude the server
    MPI_Group_excl( worldGroup, 1, ranks, &workerGroup );

    MPI_Comm_create( MPI_COMM_WORLD, workerGroup, &workerComm );

    MPI_Group_free( &worldGroup );
    MPI_Group_free( &workerGroup );
}

int computeRandom()
{
    return random();
}

void runServer( )
{
    const int numRands = 1000;
    int* rands = new int[ numRands ];
    MPI_Request request;

    const int c_fin = 0;
    const int c_req = 1;
    const int c_rep = 2;


    do
    {
        MPI_Status status;
        MPI_Recv( &request, 1, MPI_INT, MPI_ANY_SOURCE, c_req,
            MPI_COMM_WORLD, &status );
        if ( request )
        {
            for ( int i = 0; i < numRands; )
            {
                rands[i] = computeRandom();
                if ( rands[i] <= INT_MAX ) ++i;
            }
            MPI_Send( rands, numRands, MPI_INT, status.MPI_SOURCE, c_rep, 
                MPI_COMM_WORLD );
        }
    } while ( request > 0 );

    delete[] rands;
}

double runWorker( MPI_Comm& workerComm, int rank, int size)
{
    const int numRands = 1000;
    int* rands = new int[ numRands ];

    // request constants
    const int c_fin = 0;
    const int c_req = 1;
    const int c_rep = 2;


    double x;
    double y;
    bool notDone = true;
    int in = 0; 
    int out = 0;
    int totalin = 0;
    int totalout = 0;
    double epsilon = 0.00000001;
    double error;
    double Pi;
    int request = c_req;

    int server = size - 1;

    MPI_Status status;
    MPI_Send( &request, 1, MPI_INT, server, c_req, MPI_COMM_WORLD );
    while ( notDone )
    {
        MPI_Recv( rands, numRands, MPI_INT, MPI_ANY_SOURCE, c_rep, 
            MPI_COMM_WORLD, &status );
        for ( int i = 0; i < numRands; )
        {
            x = (((double)(rands[i++]))/(double)(INT_MAX)) * 2 - 1;
            y = (((double)(rands[i++]))/(double)(INT_MAX)) * 2 - 1;
            if (( x*x + y*y ) < 1.0 ) ++in;
            else ++out;
        }
        MPI_Allreduce(&in, &totalin, 1, MPI_INT, MPI_SUM, workerComm );
        MPI_Allreduce(&out, &totalout, 1, MPI_INT, MPI_SUM, workerComm );

        Pi = (4.0 * totalin) / ( totalin + totalout );
        error = fabs( Pi - 3.141592653589793238462643 );
        notDone = ( (error > epsilon) && (totalin + totalout) < 10000000 );
        if ( rank == 0 && !notDone )
        {
            request = c_fin;
            MPI_Send( &request, 1, MPI_INT, server, c_req, MPI_COMM_WORLD );
        }
        else
        {
            request = c_req;
            if ( notDone ) 
            {
                MPI_Send( &request, 1, MPI_INT, server, c_req,
                    MPI_COMM_WORLD );
            }
        }
    }
    MPI_Comm_free( &workerComm );

    delete[] rands;
    return Pi;
 }

int main( int argc, char* argv[] )
{
    initialize( argc, argv );

    int myRank;
    int worldSize;

    identifyRankAndSize( myRank, worldSize, MPI_COMM_WORLD );
    
    MPI_Comm workerComm;

    constructWorkerCommGroup( workerComm, worldSize );

    double pi;
    if ( myRank == worldSize - 1 )
    {
        runServer();
    }

    else
    {
        pi = runWorker( workerComm, myRank, worldSize );
    }

    if (myRank == 0 )
    {
        std::cout << "Pi is " << pi << std::endl;
    }

    MPI_Finalize();
}


