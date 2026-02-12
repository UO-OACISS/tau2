#include <mpi.h>
#include <stdio.h>

int main (int argc, char * argv[]) {
#if MPI_VERSION >= 4
	printf("4\n");
#elif MPI_VERSION >= 3
	printf("3\n");
#endif
	return 0;
}
