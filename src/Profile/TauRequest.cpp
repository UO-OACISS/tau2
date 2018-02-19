
#include <map>
#include <cstdlib>
#include <Profile/TauRequest.h>
#include <Profile/Profiler.h>

using namespace std;

typedef std::map<MPI_Request,request_data*> request_map;
static request_map & GetRequestMap()
{
	static request_map requests;
	return requests;
}


extern "C"
request_data * TauAddRequestData(int status, int count, MPI_Datatype datatype,
        int other, int tag, MPI_Comm comm, MPI_Request * request, int returnVal,
        int persistent)
{
    RtsLayer::LockDB();
	int typesize;
	request_map & requests = GetRequestMap();
	request_data * rq = (request_data*)(void*)0;

	if ((other != MPI_PROC_NULL) && 
            (returnVal == MPI_SUCCESS) &&
	    (requests.find(*request) == requests.end()))
       {
		rq = new request_data;
		PMPI_Type_size(datatype, &typesize);	
		rq->request = request;
		rq->status = status;
		rq->size = typesize * count;
		rq->otherParty = other;
		rq->comm = comm;
		rq->tag = tag;
		rq->is_persistent = persistent;
		requests[*request] = rq;
	}
    RtsLayer::UnLockDB();

	return rq;
}


extern "C"
request_data * TauGetRequestData(MPI_Request * request)
{
    RtsLayer::LockDB();
	request_map & requests = GetRequestMap();
	request_map::iterator it = requests.find(*request);
	if(it != requests.end()) {
        RtsLayer::UnLockDB();
		return it->second;
	} else {
        RtsLayer::UnLockDB();
		return NULL;
	}
}


extern "C"
void TauDeleteRequestData(MPI_Request * request)
{
    RtsLayer::LockDB();
	request_map & requests = GetRequestMap();
	request_map::iterator it = requests.find(*request);
	if(it != requests.end()) {
		delete it->second;
		requests.erase(it);
	}
    RtsLayer::UnLockDB();
}

