#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <math.h>
#include <Profile/Profiler.h>
#include "global.h"
#include "mpi.h"

#define MAX_PROCS 8

using namespace std;

double f1(int worksize, int myid, int halfNproc) {
  TAU_PROFILE("f1()", "", TAU_DEFAULT);
  // for half of the nodes, sleep twice as long
  if (myid >= halfNproc) {
  	worksize += worksize;
  }
  int numprocs = halfNproc * 2;
  // for strong scaling, increase thework with the number of processors.
  int n = 10000000 * worksize * numprocs;
  // for weak scaling, keep the work constant.
  //int n = 100000000 * worksize;
  cout<<myid<<" working for:"<<worksize<<endl;
  /*
  // Calculate pi by integrating 4/(1 + x^2) from 0 to 1.
  */	 
  double mySum, h, sum, x;
  h   = 1.0 / (double) n;
  sum = 0.0;
  for (int i = myid + 1; i <= n; i += numprocs)
  {
      x = h * ((double)i - 0.5);
      sum += (4.0 / (1.0 + x*x));
  }
  mySum = h * sum;
  return mySum;

  return 0;
}

int main(int argc, char* argv[]) {
  TAU_PROFILE("main()", "", TAU_DEFAULT);

  //bool loadBalance = true;
  //bool useDecay = false;
  //int decay = 4;

  int myid, nproc, namelen, halfNproc;
  char processor_name[MPI_MAX_PROCESSOR_NAME];

  // initialize MPI stuff
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  halfNproc = nproc/2;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
  MPI_Get_processor_name(processor_name, &namelen);

  // check the arguments
  if (argc != 4) {
    if (myid == 0) {
  	  cout << "\n******************************************************************************" << endl;
  	  cout << " Usage: " << argv[0] << " load_blance (0 or 1), use_decay (0 or 1), decay_length (value)" << endl;
  	  cout << "******************************************************************************\n" << endl;
	}
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (nproc > MAX_PROCS) {
  	cout << "Too many processors.  Please recompile and increase MAX_PROCS" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
   
  bool loadBalance = (atoi(argv[1]) == 1) ? true : false;
  bool useDecay = (atoi(argv[2]) == 1) ? true : false;
  int decay = atoi(argv[3]);

  TAU_PROFILE_SET_NODE(myid);

  // initialize the runtime global view
  int viewID = 0;
  TAU_REGISTER_VIEW("f1()",&viewID);

  // initialize the first runtime global communicator
  int firstCommID = 0;
  int members[nproc];
  for (int i = 0 ; i < nproc ; i++) {
    members[i] = i;
  }
  TAU_REGISTER_COMMUNICATOR(members, nproc, &firstCommID);

  // initialize the work size uniformly
  int worksize = MAX_PROCS/2;
  double lastTime[nproc];
  double circularBuffer[decay][nproc];
  if (myid == 0) {
  	for (int i=0; i<nproc; i++) {
	  lastTime[i] = 0.0;
  	  for (int j=0; j<decay; j++) {
	    circularBuffer[j][i] = 0.0;
	  }
	}
  }

  for (int iter=0; iter<20; iter++) {
    char buf[32];
    sprintf(buf, "Iteration %d", iter);

    //TAU_PHASE_CREATE_DYNAMIC(phase, buf, "", TAU_USER);
    //TAU_PHASE_START(phase);

	if (myid == 0) {
    	cout << "Iteration " << iter << endl;
	}

	double mySum, sum;
    mySum = f1(worksize, myid, halfNproc);

  	MPI_Reduce(&mySum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	/*
    if (myid == 0)
    {
	  printf("\nFor integration intervals:\n");
          printf("  pi is approximately %.16f, Error is %.16f\n",
	         sum, fabs(sum - M_PI));
    }
	*/

	if (loadBalance) {
	  double *fTime1;
	  int timeSize1 = 0;
      TAU_GET_GLOBAL_DATA(viewID, firstCommID, TAU_ALL_TO_ONE, 0, &fTime1, &timeSize1);
	
	  // get all workloads
	  int data[nproc];
	  MPI_Gather(&worksize, 1, MPI_INT, data, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
	  if (myid == 0) {
	    double avg = 0;
	    if (useDecay) {
	      // DECAY
  	      for (int i=0; i<nproc; i++) {
	  	    circularBuffer[iter%decay][i] = fTime1[i] - lastTime[i];
	  	    lastTime[i] = fTime1[i];
		    //cout<<"circ["<<iter%decay<<"]["<<i<<"]:"<<circularBuffer[iter%decay][i]<<endl;
	      }
  
	      // calculate new averages, based on circular array values
	      double newAvg[nproc];
  	      for (int i=0; i<nproc; i++) {
		    newAvg[i]= 0;
  	  	    for (int j=0; j<decay; j++) {
		      //cout<<i<<":"<<j<<":"<<circularBuffer[j][i]<<endl;
		      newAvg[i] += circularBuffer[j][i];
	  	    }
	      }
  
	      double avg = 0;
  	      for (int i=0; i<nproc; i++) {
            avg += newAvg[i];
	  	    //cout <<i<<":"<<newAvg[i]<<", ";
  	      }
	      avg = avg/nproc;
	      //cout <<"avg:"<<avg<<endl;
  
  	      for (int i=0; i<nproc; i++) {
	  	    // if faster than the average, give more work
	  	    if (newAvg[i] < avg) {
			    data[i] = data[i] + 1;
	  	    // if slower than the average, give less work
		    } else if (newAvg[i] > avg) {
			    data[i] = data[i] - 1;
		    }
  	      }
  
	    } else {
  	      for (int i=0; i<nproc; i++) {
            avg += fTime1[i];
	  	    //cout <<i<<":"<<fTime1[i]<<", ";
  	      }
	      avg = avg/nproc;
	      //cout <<"avg:"<<avg<<endl;
    
  	      for (int i=0; i<nproc; i++) {
	  	    // if faster than the average, give more work
	  	    if (fTime1[i] < avg) {
			    data[i] = data[i] + 1;
	  	    // if slower than the average, give less work
		    } else if (fTime1[i] > avg) {
			    data[i] = data[i] - 1;
		    }
  	      }
	    }
	  }

	  // redistrubite the workloads
	  MPI_Scatter(data, 1, MPI_INT, &worksize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	}

    //TAU_PHASE_STOP(phase);
  }
  MPI_Finalize();

}
