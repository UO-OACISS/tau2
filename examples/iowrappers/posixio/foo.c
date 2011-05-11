#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 100
int main(int argc, char **argv)
{
   int i, j;
   int fd0;
   int fd;
   int buf[SIZE][SIZE];


   /* Create a new file */
   fd0 = creat("out.dat", 0655); 
   fd = dup(fd0);

   

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
   
}
