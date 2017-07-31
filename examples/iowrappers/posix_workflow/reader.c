#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>


#define SIZE 10
int main(int argc, char **argv)
{
   int fd;
   int buf[SIZE][SIZE];
   struct stat stat_buf;
   int ret_in = 0;

   /* OPEN example */

   /* Wait for the file to exist */
   while (stat ("open_out.dat", &stat_buf) != 0) {
       usleep(100000);
   }

   /* Open the output file */
   fd = open("open_out.dat", O_RDONLY); 

   while((ret_in = read (fd, &buf, SIZE)) > 0){
   }

   close(fd);

   /* FOPEN example */

   /* Wait for the file to exist */
   while (stat ("fopen_out.dat", &stat_buf) != 0) {
       usleep(100000);
   }

   /* Open the output file */
   FILE * fd2 = fopen("fopen_out.dat", "r"); 
   if (fd2==NULL) {fprintf (stderr, "File error: fopen_out.dat\n"); exit (1);}

   char buffer[1024];
   while((ret_in = fread (buf, 1, 1024, fd2)) > 0){
   }

   fclose(fd2);
   
}
