/****************************************************************************
** File 	: global.h
** Author 	: Sameer Shende					
** Contents 	: function prototypes for kl.c to calculate kth largest elt.
*****************************************************************************/
#ifndef _GLOBAL_H
#define _GLOBAL_H

#include "mpi.h"

/* external macro declarations */

#define TAU_REGISTER_VIEW(a,b) TauGlobal::tau_register_view(a,b);
#define TAU_REGISTER_COMMUNICATOR(a,b,c) TauGlobal::tau_register_communicator(a,b,c);
#define TAU_GET_GLOBAL_DATA(a,b,c,d,e,f) TauGlobal::tau_get_global_data(a,b,c,d,e,f);

#define TAU_GLOBAL_VIEW TAU_GET_PROFILE_GROUP("TAU_GLOBAL_VIEW")

/* class declarations */

typedef struct _global_view_t_ {
	const char *eventName;
	int id;
	struct _global_view_t_ *next;
} global_view_t;

typedef struct _global_comm_t_ {
	MPI_Comm handle;
	int* members;
	int size;
	int id;
	struct _global_comm_t_ *next;
} global_comm_t;

typedef struct _tau_profile_data_t_ {
	double counterExclusiveValue;
	double counterInclusiveValue;
	int numOfCalls;
	int numOfSubRoutines;
} tau_profile_data_t;

const int TAU_ALL_TO_ALL = 0;
const int TAU_ONE_TO_ALL = 1;
const int TAU_ALL_TO_ONE = 2;

class TauGlobal {
  public:
	void static tau_register_view(const char* event_name, int* viewID);
	void static tau_register_communicator(int members[], int size, int* commID);
	void static tau_get_global_data(int viewID, int commID, int type, int sink, double** data, int* outSize);

  private:
	TauGlobal(void);
	~TauGlobal(void);
	static TauGlobal* aggregator;
	global_view_t *views;
	global_view_t *lastView;
	int numViews;
	global_comm_t *communicators;
	global_comm_t *lastCommunicator;
	int numCommunicators;
	tau_profile_data_t* myheap[10][10];
};

#endif /* _GLOBAL_H */

/* EOF global.h */

/*

We have the Global Performance Space.
Do the following:
	Register metric name + event name -> Global Performance view ID
	Register process ranks -> Global Performance Communicator ID
	Constant Static pattern IDs -> NxN, Nx1, 1xN
When performance data desired:
	Request data (GPVID, GPCID, pattern ID);

*/
