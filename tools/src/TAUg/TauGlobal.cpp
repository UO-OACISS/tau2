/****************************************************************************
**			TAU Portable Profiling Package			                       **
**			http://www.cs.uoregon.edu/research/tau	                       **
*****************************************************************************
**    Copyright 1997-2008                                                  **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/

/***************************************************************************
**  File            : TauGlobal.cpp                                       **
**  Description     : TAU Global (TAUg) implementation file.              **
**  Author          : Kevin Huck                                          **
**  Contact         : khuck@cs.uoregon.edu                                **
**  Documentation   : See http://tau.uoregon.edu                          **
***************************************************************************/

#include "TauGlobal.h"
#include <stdio.h>
#include <math.h>
#include <Profile/Profiler.h>
#include "mpi.h"

#include <iostream>
using namespace std;

//Static stuff
TauGlobal* TauGlobal::aggregator = new TauGlobal();

void Error(char * msg) {
	cerr << msg << endl;
	MPI_Abort(MPI_COMM_WORLD, 0);
	exit(1);
}

void Error(char * msg, int mpierror) {
	cerr << msg << mpierror << endl;
	MPI_Abort(MPI_COMM_WORLD, mpierror);
	exit(mpierror);
}

TauGlobal::TauGlobal(void) {
    //TAU_PROFILE("TauGlobal()", "void (void)", TAU_GLOBAL_VIEW);
	// create a hashtable with prime size 
	//hashTable = new HashTable(11);

	// create a list of views
	views = NULL;
	lastView = NULL;
	numViews = 0;

	// create a list of communicators
	communicators = NULL;
	lastCommunicator = NULL;
	numCommunicators = 0;

	// memory management
	for (int i = 0 ; i < 10 ; i++) {
		for (int j = 0 ; j < 10 ; j++) {
			myheap[i][j] = NULL;
		}
	}
}

TauGlobal::~TauGlobal(void) {
    TAU_PROFILE("~TauGlobal()", "void (void)", TAU_GLOBAL_VIEW);
	// free the views
	while (views != NULL) {
		// get the first view
		global_view_t *firstView = views;
		// point the head to the next view
		views = firstView->next;
		// free the view
		//free(firstView->eventName);
		//free(firstView->metricName);
		free(firstView);
	}

	// free the communicators
	while (communicators != NULL) {
		// get the first communicator
		global_comm_t *firstCommunicator = communicators;
		// point the head to the next communicator
		communicators = firstCommunicator->next;
		// free the first communicator
		free(firstCommunicator->members);
		free(firstCommunicator);
	}

	// free the performance data
	for (int i = 0 ; i < 10 ; i++) {
		for (int j = 0 ; j < 10 ; j++) {
			if (myheap[i][j] != NULL)
				free (myheap[i][j]);
		}
	}

}

void TauGlobal::tau_register_view(const char* event_name, int* viewID) {
    TAU_PROFILE("tau_register_view()", "void (char*, char*, bool, int*)", TAU_GLOBAL_VIEW);
	// allocate the view structure
	global_view_t *newView;
	if ((newView = (global_view_t*)malloc(sizeof(global_view_t))) == NULL) 
		Error("Cannot allocate view structure");

	newView->eventName = event_name;
	newView->id = ++aggregator->numViews;
	newView->next = NULL;

	// add the view to the list
	if (aggregator->views == NULL) {
		aggregator->views = newView;
		aggregator->lastView = newView;
	} else {
		aggregator->lastView->next = newView;
		aggregator->lastView = newView;
	}

	*viewID = newView->id;
	return;
}

void TauGlobal::tau_register_communicator(int members[], int size, int* commID) {
    TAU_PROFILE("tau_register_communicator()", "void (int[], int, int*)", TAU_GLOBAL_VIEW);

	int ierr;
	ierr = MPI_Barrier(MPI_COMM_WORLD);
	if (ierr != MPI_SUCCESS ) Error("MPI Error: ", ierr);

	// allocate the communicator structure
	global_comm_t *newCommunicator;
	if ((newCommunicator = (global_comm_t*)malloc(sizeof(global_comm_t))) == NULL) 
		Error("Cannot allocate comm structure");

	if ((newCommunicator->members = (int*)malloc(sizeof(int)*size)) == NULL) 
		Error("Cannot allocate comm structure");

	for (int i = 0 ; i < size ; i++) {
		newCommunicator->members[i] = members[i];
	}

	MPI_Group world, newGroup;
	MPI_Comm newComm;
	int newRank;

	// get the group for the world communicator
	ierr = MPI_Comm_group(MPI_COMM_WORLD, &world);
	if (ierr != MPI_SUCCESS ) Error("MPI Error: ", ierr);

	// create a new group, a subset with the specified members
	ierr = MPI_Group_incl(world, size, members, &newGroup);
	if (ierr != MPI_SUCCESS ) Error("MPI Error: ", ierr);

	// create a new communicator from this group
	ierr = MPI_Comm_create(MPI_COMM_WORLD, newGroup, &newComm);
	if (ierr != MPI_SUCCESS ) Error("MPI Error: ", ierr);

	// get rank in this new group, if this process is in it
	ierr = MPI_Group_rank(newGroup, &newRank);
	if (ierr != MPI_SUCCESS ) Error("MPI Error: ", ierr);
	//cout << aggregator->getRank() << ": new rank: " << newRank << endl;

	newCommunicator->size = size;
	newCommunicator->handle = newComm;
	newCommunicator->id = ++aggregator->numCommunicators;
	newCommunicator->next = NULL;

	// add the comm to the list
	if (aggregator->communicators == NULL) {
		aggregator->communicators = newCommunicator;
		aggregator->lastCommunicator = newCommunicator;
	} else {
		aggregator->lastCommunicator->next = newCommunicator;
		aggregator->lastCommunicator = newCommunicator;
	}

	*commID = newCommunicator->id;
	return;
}

void TauGlobal::tau_get_global_data(int viewID, int commID, int type, int sink, double** data, int* outSize) {
    TAU_PROFILE("tau_get_global_data()", "void (int, int, double[])", TAU_GLOBAL_VIEW);

	int ierr = MPI_SUCCESS;

	int globalRank = 0;
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
	if (ierr != MPI_SUCCESS ) Error("MPI Error: ", ierr);

	// find the view
	//cout << "Looking for View: " << viewID << endl;
	global_view_t *tmpView = aggregator->views;

	while (tmpView != NULL) {
		//cout << "View: " << tmpView->id << endl;
		if (tmpView->id == viewID)
			break;
		tmpView = tmpView->next;
	}

	if (tmpView == NULL) {
		Error("Unable to find view!");
	}

	// find the communicator
	//cout << "Looking for Communicator: " << commID << endl;
	global_comm_t *tmpCommunicator = aggregator->communicators;

	while (tmpCommunicator != NULL) {
		//cout << "Communicator: " << tmpCommunicator->id << endl;
		if (tmpCommunicator->id == commID) {
			bool member = false;
			//cout << globalRank << ":size: " << tmpCommunicator->size << endl ;
			for (int i = 0 ; i < tmpCommunicator->size ; i++) {
				//cout << globalRank << ":member: " << tmpCommunicator->members[i] << endl ;
				if (tmpCommunicator->members[i] == globalRank) {
					member = true;
				}
				if (tmpCommunicator->members[i] == sink) {
					//cout << globalRank << ":newsink: " << sink << ":" <<i << endl ;
					// convert the sink from global rank to communicator rank
					sink = i;
				}
			}
			if (!member) {
				*outSize = 0;
				cout << globalRank << ":bailing!" << endl ;
				// didn't find this process in the communicator, so exit
				return;
			} else {
				break;
			}
		}
		tmpCommunicator = tmpCommunicator->next;
	}
	//cout << globalRank << ":continuing!" << endl ;

	if (tmpCommunicator == NULL) {
		Error("Unable to find communicator!");
	}
	
	// ask TAU for the data

	const char **inFuncs;
  	inFuncs = (const char **) malloc(sizeof(const char *));

	if (false) {
		const char **functionList;
		int numOfFunctions;
		TAU_GET_FUNC_NAMES(functionList, numOfFunctions);
  		inFuncs[0] = functionList[0];
	} else {
  		inFuncs[0] = tmpView->eventName;
	}
	//cout << "getting data for event: '" << inFuncs[0] << "'" << endl;
     	
	if (aggregator->myheap[viewID][commID] == NULL)
		aggregator->myheap[viewID][commID] = (tau_profile_data_t *)malloc(sizeof(tau_profile_data_t)*tmpCommunicator->size);
	tau_profile_data_t* buf = aggregator->myheap[viewID][commID];

  	double **counterExclusiveValues;
	double **counterInclusiveValues;
	int *numOfCalls;
	int *numOfSubRoutines;
	const char **counterNames;
	int numOfCounters;

	MPI_Comm comm = tmpCommunicator->handle;
	//ierr = MPI_Barrier(comm);
	//if (ierr != MPI_SUCCESS ) Error("MPI Error: ", ierr);

	TAU_GET_FUNC_VALS(inFuncs, 1, counterExclusiveValues, counterInclusiveValues,
  		numOfCalls, numOfSubRoutines, counterNames, numOfCounters);

	//cout << globalRank << ":" << counterExclusiveValues[0][0] << " " << counterInclusiveValues[0][0] << " " << numOfCalls[0] << " " << numOfSubRoutines[0] << endl;

	// send this data to all others in the communicator
	int recvcount = 1;
	int sendcount = recvcount;
	for (int i = 0 ; i < /*tmpCommunicator->size*/ 1 ; i++) {
		buf[i].counterExclusiveValue = counterExclusiveValues[0][0];
		buf[i].counterInclusiveValue = counterInclusiveValues[0][0];
		buf[i].numOfCalls = numOfCalls[0];
		buf[i].numOfSubRoutines = numOfSubRoutines[0];
	}

	//for (int i = 0 ; i < tmpCommunicator->size ; i++) {
		//cout << globalRank << ":before buf[" << i << "]: " << buf[i].counterExclusiveValue << endl ;
	//}

	// set up 3 blocks
	int blockCounts[3] = {2,2,1};
	MPI_Datatype types[3];
	MPI_Aint displacements[3];
	MPI_Datatype profileType;

	// initialize types and displs with addresses of items 
	MPI_Address(&buf[0].counterExclusiveValue, &displacements[0]);
	MPI_Address(&buf[0].numOfCalls, &displacements[1]);
	MPI_Address(&buf[1].counterExclusiveValue, &displacements[2]);
	types[0] = MPI_DOUBLE;
	types[1] = MPI_INT;
	types[2] = MPI_UB;

	// make the displacements relative to the first address
	displacements[1] -= displacements[0];
	displacements[2] -= displacements[0];
	displacements[0] = 0;

	// build the new type
	MPI_Type_struct(3, blockCounts, displacements, types, &profileType);
	MPI_Type_commit(&profileType);

	if (type == TAU_ALL_TO_ALL) {
		//cout << globalRank << ":doing alltoall" << endl ;
		//ierr = MPI_Alltoall (buf, sendcount, profileType, buf, recvcount, 
			//profileType, comm);
		ierr = MPI_Allgather (&(buf[0]), 1, profileType, buf, 1, profileType, comm);
		//cout << globalRank << ":done" << endl ;
		*outSize = tmpCommunicator->size;
		if (ierr != MPI_SUCCESS ) Error("MPI Error: ", ierr);
	} else if (type == TAU_ALL_TO_ONE) {
		//cout << globalRank << ":doing manytoone" << endl ;
		ierr = MPI_Gather (&(buf[0]), sendcount, profileType, buf, recvcount, 
			profileType, sink, comm);
		//cout << globalRank << ":done" << endl ;
		*outSize = tmpCommunicator->size;
	} else if (type == TAU_ONE_TO_ALL) {
		//cout << globalRank << ":doing onetomany" << endl ;
		ierr = MPI_Scatter (buf, sendcount, profileType, &(buf[0]), recvcount, 
			profileType, sink, comm);
		//cout << globalRank << ":done" << endl ;
		*outSize = 1;
	}

	//if (globalRank == 0) {
		//for (int i = 0 ; i < tmpCommunicator->size ; i++) {
			//cout << globalRank << ":after buf[" << i << "]: " << buf[i].counterExclusiveValue << endl ;
		//}
	//}

	// only return an array of exclusive times for now
	double* tmpBuf = (double *)malloc(sizeof(double)*tmpCommunicator->size);
	for (int i = 0 ; i < tmpCommunicator->size ; i++) {
		tmpBuf[i] = buf[i].counterExclusiveValue;
	}
	*data = tmpBuf;
	return;
}

