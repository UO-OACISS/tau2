#include <mpi.h>

#if defined(MPI_4)

    #if MPI_VERSION >= 4
        int main(void) { return 0; }
    #else
        #error "MPI version is lower than 4"
    #endif

#elif defined(MPI_3)

    #if MPI_VERSION >= 3
        int main(void) { return 0; }
    #else
        #error "MPI version is lower than 3"
    #endif

#else
    #error "No MPI version flag provided (use -DMPI_3 or -DMPI_4)"
#endif