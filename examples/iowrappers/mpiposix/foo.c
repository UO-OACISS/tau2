#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define SIZE 100
int main(int argc, char **argv)
{
   int i, j;
   int fd;
   int rank;
   int buf[SIZE][SIZE];
   char filename[SIZE];


   MPI_Init(&argc, &argv);
   /* Create a new file */
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   sprintf(filename, "out.%d.dat", rank);
   
   fd = creat(filename, 0655); 

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
       /* How long does it take to write this? What bandwidth do I get? */
     }
   }
   close(fd);
   MPI_Finalize();
   
}
