#include <stdio.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#define SIZE 100
int main(int argc, char **argv) {
   int i, j;
   int fd;
 
   int buf[SIZE][SIZE];


   /* Create a new file */
   if ((fd = open("out.dat", O_RDONLY )) == -1) {
     perror("Error opening out.dat");
   }

   for (i=0; i < SIZE; i++) {
     for (j=0; j < SIZE; j++) {
       int ret = read(fd, &buf[i][j], sizeof(int));
       /* How long does it take to read this? What bandwidth do I get? */
     }
   }

   close(fd);
}
