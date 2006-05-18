/* This demonstrates how data cache misses can affect the performance of an 
application. We show how the time/counts for a simple matrix multiplication
algorithm dramatically reduce when we employ a strip mining optimization. */

#define SIZE 512
#define CACHE 64

double A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];

double multiply(void)
{
  int i, j, k, n, m;
  int vl, sz, strip;


/*begin*/for (n = 0; n < SIZE; n++)
           for (m = 0; m < SIZE; m++)
           { /* this belongs to the second loop */
             A[n][m] = B[n][m] = n + m ;
             C[n][m] = 0;
/*ending */}/*for the second loop, not the first!*/

  for (i = 0; i < SIZE; i ++)
  { 
    for (j = 0; j < SIZE; j++)
    {
      for (k = 0; k < SIZE; k++)
  	C[i][j] += A[i][k] * B[k][j];
    }
  }

  /* Now we employ the strip mining optimization */

  for(n = 0; n < SIZE; n++)
    for(m = 0; m < SIZE; m++)
      C[n][m] = 0; 
  
  for(i=0; i < SIZE; i++)
    for(k=0; k < SIZE; k++)
      for(sz = 0; sz < SIZE; sz+=CACHE)
      {
        /*vl = min(SIZE-sz, CACHE); */
  	vl = (SIZE - sz < CACHE ? SIZE - sz : CACHE); 
        for(strip = sz; strip < sz+vl; strip++)
          C[i][strip] += A[i][k]*B[k][strip];
      }

 
  return C[SIZE-10][SIZE-10];
}
       
       
int main(int argc, char **argv)
{
  multiply();
  return 0;
}
