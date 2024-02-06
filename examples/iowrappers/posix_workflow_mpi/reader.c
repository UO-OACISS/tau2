#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <mpi.h>
#include "common.h"


#define SIZE 10
int main(int argc, char **argv)
{
   int fd;
   int buf[SIZE][SIZE];
   struct stat stat_buf;
   int ret_in = 0;
   int proc, me;
   char filename[1024];

   MPI_Init (&argc, &argv);
   MPI_Comm_size (MPI_COMM_WORLD, &proc);
   MPI_Comm_rank (MPI_COMM_WORLD, &me);

   check_args(me, proc);
   exchange_data(me, proc);

   /* OPEN example */

   /* Wait for the file to exist */
   snprintf(filename, sizeof(filename),  "open_out.%d.dat", me);
   while (stat (filename, &stat_buf) != 0) {
       usleep(100000);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   exchange_data(me, proc);

   /* Open the output file */
   fd = open(filename, O_RDONLY); 

   while((ret_in = read (fd, &buf, SIZE)) > 0){
   }

   close(fd);

   MPI_Barrier(MPI_COMM_WORLD);
   exchange_data(me, proc);

   /* FOPEN example */

   /* Wait for the file to exist */
   snprintf(filename, sizeof(filename),  "fopen_out.%d.dat", me);
   while (stat (filename, &stat_buf) != 0) {
       usleep(100000);
   }

   /* Open the output file */
   FILE * fd2 = fopen(filename, "r"); 
   if (fd2==NULL) {fprintf (stderr, "File error: %s\n", filename); exit (1);}

   char buffer[1024];
   while((ret_in = fread (buf, 1, 1024, fd2)) > 0){
   }

   fclose(fd2);
   
   MPI_Barrier(MPI_COMM_WORLD);
   exchange_data(me, proc);
   MPI_Finalize ();
}
