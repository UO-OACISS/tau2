#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 100
int main(int argc, char **argv) {
   int i, j;
   FILE *fp;
 
   int buf[SIZE][SIZE];


   /* Create a new file */
   fp = fopen("out.dat", "r+");

   /* fill up our array with some dummy values */
/*
   for (i=0; i < SIZE; i++) {
     for (j=0; j < SIZE; j++) {
       buf[i][j] = i+34*j;
     }
   }
*/

   /* write the matrix in the file */
   for (i=0; i < SIZE; i++) {
     for (j=0; j < SIZE; j++) {
       fread(&buf[i][j], sizeof(int), 1, fp);
       /* How long does it take to read this? What bandwidth do I get? */
     }
   }
   fclose(fp);

 
}
