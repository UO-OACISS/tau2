#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <mpi.h>
#include "common.h"

#define SIZE 10
int main(int argc, char **argv)
{
   int i, j;
   int fd;
   int buf[SIZE][SIZE];
   int proc, me;
   char filename[1024];

   MPI_Init (&argc, &argv);
   MPI_Comm_size (MPI_COMM_WORLD, &proc);
   MPI_Comm_rank (MPI_COMM_WORLD, &me);

   check_args(me, proc);
   exchange_data(me, proc);

   /* OPEN */

   /* Create a new file */
   snprintf(filename, sizeof(filename),  "open_out.%d.dat", me);
   fd = open(filename, O_WRONLY | O_CREAT, 0644);

   /* fill up our array with some dummy values */
   for (i=0; i < SIZE; i++) {
     for (j=0; j < SIZE; j++) {
       buf[i][j] = i+34*j;
     }
   }

   /* write the matrix in the file */
   for (i=0; i < SIZE; i++) {
     for (j=0; j < SIZE; j++) {
       write(fd, buf, sizeof(buf));
     }
   }
   close(fd);

   MPI_Barrier(MPI_COMM_WORLD);
   exchange_data(me, proc);

   /* FOPEN */

   /* Create a new file */
   snprintf(filename, sizeof(filename),  "fopen_out.%d.dat", me);
   FILE * fd2 = fopen(filename, "w");
   if (fd2 != NULL) {
     fputs("Dummy string into fopen_out.dat\n", fd2);
     fclose(fd2);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   exchange_data(me, proc);

   /* CREAT */

   /* Create a new file */
   snprintf(filename, sizeof(filename),  "creat_out.%d.dat", me);
   fd = creat(filename, 0655);

   /* write the matrix in the file */
   for (i=0; i < SIZE; i++) {
     for (j=0; j < SIZE; j++) {
       write(fd, buf, sizeof(buf));
     }
   }
   close(fd);
   
   MPI_Barrier(MPI_COMM_WORLD);
   exchange_data(me, proc);
   MPI_Finalize ();
}
