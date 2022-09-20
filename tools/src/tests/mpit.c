#include <mpi.h>
int main(int argc, char **argv) {
    MPI_T_cvar_handle handle;
    MPI_T_enum enumtype;
    int provided, ncvars, npvars;

    MPI_T_init_thread(MPI_THREAD_SINGLE, &provided);
    MPI_Init(&argc, &argv);
    MPI_T_cvar_get_num(&ncvars);
    MPI_T_pvar_get_num(&npvars);

    MPI_Finalize();
}

