#include <Profile/Profiler.h>

#line 1 "matmult_initialize.c"
#include "matmult_initialize.h"

void initialize(double **matrix, int rows, int cols) {

#line 3

TAU_PROFILE_TIMER(tautimer, "void initialize(double **, int, int) C [{matmult_initialize.c} {3,1}-{16,1}]", " ", TAU_USER);
	TAU_PROFILE_START(tautimer);


#line 3
{
  int i,j;
#pragma omp parallel private(i,j) shared(matrix)
  {
    //set_num_threads();
    /*** Initialize matrices ***/
#pragma omp for nowait
    for (i=0; i<rows; i++) {
      for (j=0; j<cols; j++) {
        matrix[i][j]= i+j;
      }
    }
  }

#line 16

}
	
	TAU_PROFILE_STOP(tautimer);


#line 16
}

