
#include "headers2/B.h"
#ifdef USE_MPI
#include "mpi.h"
#endif

int main ( int argc, char * argv[] ) {
    REPORT
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
#endif
    A a;
    B b;

    a.method1();
    b.method1();
    a.method2();
    b.method2();
    a.vmethod();
    b.vmethod();

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
