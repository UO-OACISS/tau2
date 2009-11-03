////////////////////////////////////////////////////////////////////
// File:
// Date:
// Author:
//
// Description:
/////////////////////////////////////////////////////////////////////
#ifdef PROFILING_ON
#include <Profile/Profiler.h>
#endif

#include <cmath>
#include <iostream>

#ifdef USE_MPI
#include "mpi.h"
#endif


using namespace std;


double f(double x)
{
#ifdef PROFILING_ON
  TAU_PROFILE("f", "f(double x)", TAU_USER);
#endif
  for(int i=0;i<100;i++)
  {
      x*=1.0001;
  }
  
  return sin( pow(x,2.3)/sqrt(fabs(x)) );
}

double f1(double x)
{
  for(int i=0;i<100;i++)
  {
      x*=1.0001;
  }
  return sin( pow(x,2.3)/sqrt(fabs(x)) );
}






int checkResults() {
  const char **inFuncs;
  /* The first dimension is functions, and the second dimension is counters */
  double **counterExclusiveValues;
  double **counterInclusiveValues;
  int *numOfCalls;
  int *numOfSubRoutines;
  const char **counterNames;
  int numOfCouns;

  int numFunctions;
  const char **functionList;

  TAU_GET_FUNC_NAMES(functionList, numFunctions);

  double loop1ex=-1, loop2ex=-1;
  double loop1in=-1, loop2in=-1;

  TAU_GET_FUNC_VALS(functionList, numFunctions,
		    counterExclusiveValues,
		    counterInclusiveValues,
		    numOfCalls,
		    numOfSubRoutines,
		    counterNames,
		    numOfCouns);

  for (int i=0; i < numFunctions; i++) {
    if (strcmp("loop timer 1",functionList[i]) == 0) {
      loop1ex = counterExclusiveValues[i][0];
      loop1in = counterInclusiveValues[i][0];
    }

    if (strcmp("loop timer 2",functionList[i]) == 0) {
      loop2ex = counterExclusiveValues[i][0];
      loop2in = counterInclusiveValues[i][0];
    }
  }

  printf ("loop timer 1, exclusive = %G\n", loop1ex);
  printf ("loop timer 1, inclusive = %G\n", loop1in);
  printf ("loop timer 2, exclusive = %G\n", loop2ex);
  printf ("loop timer 2, inclusive = %G\n", loop2in);
  printf ("Difference = %G = %G%%\n", fabs(loop1in-loop2in), fabs(loop1in - loop2in) / loop2in*100);

  // int error = 0;

  // if (((loop1in - loop2in) / loop2in) > 0.6) {
  //   printf ("Difference is more than 60%%, please investigate!\n");
  //   error = 1;
  // }

  printf ("Exclusive loop 1 = %G%% of inclusive\n", loop1ex / loop1in*100);
  // if (loop1ex / loop1in > 0.15) {
  //   printf ("Exclusive timer 1 is more than 15%% of inclusive, please investigate!\n");
  //   error = 1;
  // }

  // if (error == 0) {
  //   printf ("\nCompensation seems to be working!\n");
  // }
  // return error;


}



int main(int argc,char **argv)
{ 
#ifdef PROFILING_ON
  TAU_PROFILE("main()", "int (int, char **)", TAU_DEFAULT);
  TAU_PROFILE_INIT(argc,argv);
#endif

  int thisProcNum = 0;


  double xx = 1.0;
  double yy = 1.0;

  for(int i=0;i< 1000000; i++ )
  {
    xx += f1(xx);
  }

  xx = 1.0;

#ifdef PROFILING_ON
  TAU_PROFILE_SET_NODE(thisProcNum);
  TAU_PROFILE_TIMER(t1,"loop timer 1","",TAU_USER);
  TAU_PROFILE_TIMER(t2,"loop timer 2","",TAU_USER);
  
  TAU_PROFILE_START(t1);
#endif /* PROFILING_ON */
  for(int i=0;i< 1000000; i++ )
  {
    xx += f(xx);
    xx = sqrt(1.0 + fabs(xx));
  }
#ifdef PROFILING_ON
  TAU_PROFILE_STOP(t1);

  TAU_PROFILE_START(t2);
#endif /* PROFILING_ON */

  for(int i=0;i< 1000000 ; i++ )
  {
    yy += f1(yy);
    yy = sqrt(1.0 + fabs(yy));
  }
#ifdef PROFILING_ON
  TAU_PROFILE_STOP(t2);
#endif /* PROFILING_ON */

  // cout<<" xx = "<<xx<<"  yy= "<<yy<<endl;
  
  
  return checkResults();
}

