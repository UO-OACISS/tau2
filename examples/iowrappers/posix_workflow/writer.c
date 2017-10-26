#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

#define SIZE 10
int main(int argc, char **argv)
{
   int i, j;
   int fd;
   int buf[SIZE][SIZE];

   /* OPEN */

   /* Create a new file */
   fd = open("open_out.dat", O_WRONLY | O_CREAT, 0644);

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

   /* FOPEN */

   /* Create a new file */
   FILE * fd2 = fopen("fopen_out.dat", "w");
   if (fd2 != NULL) {
     fputs("Dummy string into fopen_out.dat\n", fd2);
     fclose(fd2);
   }

   /* CREAT */

   /* Create a new file */
   fd = creat("creat_out.dat", 0655);

   /* write the matrix in the file */
   for (i=0; i < SIZE; i++) {
     for (j=0; j < SIZE; j++) {
       write(fd, buf, sizeof(buf));
     }
   }
   close(fd);
   
}
