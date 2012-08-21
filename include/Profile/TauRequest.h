
#ifndef __TAU_REQUEST_H__
#define __TAU_REQUEST_H__

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RQ_SEND    0x1
#define RQ_RECV    0x2
#define RQ_CANCEL  0x4

/* if MPI_Cancel is called on a request, 'or' RQ_CANCEL into status.
 * After a Wait* or Test* is called on that request, check for RQ_CANCEL.
 * If the bit is set, check with MPI_Test_cancelled before registering
 * the send/receive as 'happening'.
 */

typedef struct _request_data
{
	MPI_Request * request;
	int status;
	int size;
	int tag;
	int otherParty;
	int is_persistent;
	MPI_Comm comm;
} request_data;


request_data * 
TauAddRequestData(int status, int count, MPI_Datatype datatype, int other,
                  int tag, MPI_Comm comm, MPI_Request * request, int returnVal, 
                  int persistent);

request_data * TauGetRequestData(MPI_Request * request);

void TauDeleteRequestData(MPI_Request * request);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* __TAU_REQUEST_H__ */

