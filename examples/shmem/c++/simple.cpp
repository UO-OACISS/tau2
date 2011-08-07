#include <shmem.h>
#include <stdio.h>
/* Copyright: This example is taken from the Cray man shmem command on XT3 */
int main(int argc, char **argv)
{
        long source[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        static long target[10];
        start_pes(0);
        target[0] = 0;

        if (shmem_my_pe() == 0) {
          /* put 10 words into target on PE 1 */
          shmem_long_put(target, source, 10, 1);
        }

        shmem_barrier_all();  /* sync sender and receiver */
        printf("target[0] on PE %d is %d\n", shmem_my_pe(), target[0]);
#ifndef TAU_SGI_MPT_SHMEM
        shmem_finalize();
#endif /* TAU_SGI_MPT_SHMEM */
}

