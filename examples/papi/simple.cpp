#include <Profile/Profiler.h>

#define SIZE 300
#define CACHE 64

double multiply(void)
{
  int i, j, k, n, m;
  int vl, sz, strip;
  TAU_PROFILE("multiply", "void (void)", TAU_USER);
  TAU_PROFILE_TIMER(t1,"multiply-regular", "void (void)", TAU_USER);
  TAU_PROFILE_TIMER(strip_timer,"multiply-with-strip-mining-optimization", "void (void)", TAU_USER);

  double A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];

  for (n = 0; n < SIZE; n++)
    for (m = 0; m < SIZE; m++)
      A[n][m] = B[n][m] = n + m ;
  TAU_PROFILE_START(t1);
  for (i = 0; i < SIZE; i ++)
  { 
    for (j = 0; j < SIZE; j++)
    {
      C[i][j] = 0;
      for (k = 0; k < SIZE; k++)
  	C[i][j] += A[i][k] * B[k][j];
    }
  }
  TAU_PROFILE_STOP(t1);

  /* Now we employ the strip mining optimization */

  for(n = 0; n < SIZE; n++)
    for(m = 0; m < SIZE; m++)
      C[n][m] = 0; 
  
  TAU_PROFILE_START(strip_timer);
  for(i=0; i < SIZE; i++)
    for(k=0; k < SIZE; k++)
      for(sz = 0; sz < SIZE; sz+=CACHE)
      {
        //vl = min(SIZE-sz, CACHE);
  	vl = (SIZE - sz < CACHE ? SIZE - sz : CACHE); 
        for(strip = sz; strip < sz+vl; strip++)
          C[i][strip] += A[i][k]*B[k][strip];
      }
  TAU_PROFILE_STOP(strip_timer);

 
  return C[SIZE-10][SIZE-10];
  // So KCC doesn't optimize this loop away.
}
       
       
int main(int argc, char **argv)
{
  TAU_PROFILE("main()", "int (int, char **)", TAU_DEFAULT);
  TAU_PROFILE_SET_NODE(0);

  multiply();
  return 0;
}
