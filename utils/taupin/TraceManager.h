#pragma once
#include "SpecManager.h"
#include "pin.h" 
#include<mpi.h>
#include<list>
#include<fstream>

//#define TRACE
//some tracing macros
#ifdef TRACE
#define DBG_TRACE(exp) cout<<__FUNCTION__<<">:"<<exp<<endl;\
						 fflush(stdout) 
#define DBG_TR_EXIT(exp) cout<<__FUNCTION__<<">:"<<exp<<endl;\
						 fflush(stdout);\
						 exit(1)
#define DBG_TR_RET(exp) cout<<__FUNCTION__<<">:"<<exp<<endl;\
						fflush(stdout);\
						return 
#define DBG_TR_RETNULL(exp) cout<<__FUNCTION__<<">:"<<exp<<endl;\
							fflush(stdout);\
							return NULL
#define DBG_TR_RETZERO(exp)	cout<<__FUNCTION__<<">:"<<exp<<endl;\
							fflush(stdout);\
							return 0
#else 
#define DBG_TRACE(exp)
#define DBG_TR_EXIT(exp)
#define DBG_TR_RET(exp) 
#define DBG_TR_RETNULL(exp)
#define DBG_TR_RETZERO(exp)
#endif

//structure passed to 
//track the instrumentation 
struct RtnTrack{
	int stage; //not yet used 
	int flag; //flag
	string rtn; //rtn name 
	string img; //imagename
	void *tau; //some handel for TAU instrumentor
};

//Manages the tracing 
//knows TAU and PIN both
class TraceManager
{
private:	
	string img; //image name
	int rtncnt;
	int procid_0;
	SpecManager *Spm; //keep the specification manager
	string ImageTrim(string image);
	list<RtnTrack*> bef_list;
	list<RtnTrack*> aft_list;
	ofstream trace_file;
	bool mpi_setup;
public:
	
	TraceManager(SpecManager* spm );
	TraceManager(SpecManager* spm,string img);
	~TraceManager(void);
	//Apply instrumentation on an image by consulting spec manager
	void InstApply(IMG img);
	void InstApply();
	//before and after execute of the rtn block
	void BeforeExec(RtnTrack* rtntr);
	void AfterExec(RtnTrack* rtntr);
	//this is the function which will dump everything at the last
	void EndTrace();
	bool IsInstSafe(string rtn_name);
	bool IsMpiRtn(string rtn_name);
	bool IsNormal(RTN myrtn);
	void LogMessage(string msg);
};


extern "C" {
   //this was a hack to fix around rewriting thr existing code
   int  MPIAPI HMPI_Allgather( void * sendbuf, int sendcount,MPI_Datatype sendtype, 
					void * recvbuf,int recvcount,MPI_Datatype recvtype,MPI_Comm comm);
   int MPIAPI HMPI_Allgatherv(void * sendbuf,int sendcount,MPI_Datatype sendtype,
					 void * recvbuf,int * recvcounts,int * displs,
					  MPI_Datatype recvtype, MPI_Comm comm);
   int  MPIAPI HMPI_Allreduce(void * sendbuf,void * recvbuf,int count,
					MPI_Datatype datatype,MPI_Op op,MPI_Comm comm);
   int   MPIAPI HMPI_Alltoall(void * sendbuf,int sendcount,MPI_Datatype sendtype,
					void * recvbuf,int recvcnt,MPI_Datatype recvtype,MPI_Comm comm);
   int  MPIAPI HMPI_Alltoallv(void * sendbuf,int * sendcnts,int * sdispls,MPI_Datatype sendtype,
				void * recvbuf,int * recvcnts,int * rdispls,MPI_Datatype recvtype,MPI_Comm comm);
   int   MPIAPI HMPI_Barrier(MPI_Comm comm);
   int   MPIAPI HMPI_Bcast( void * buffer,int count,MPI_Datatype datatype,int root,MPI_Comm comm);
   int   MPIAPI HMPI_Gather(void * sendbuf,int sendcnt,MPI_Datatype sendtype,void * recvbuf,
				int recvcount,MPI_Datatype recvtype,int root,MPI_Comm comm);
   int   MPIAPI HMPI_Gatherv(void * sendbuf,int sendcnt,MPI_Datatype sendtype,void * recvbuf,
				int * recvcnts,int * displs,MPI_Datatype recvtype,int root,MPI_Comm comm);
   int  MPIAPI HMPI_Op_create(MPI_User_function * function,int commute,MPI_Op * op);
   int   MPIAPI HMPI_Op_free(MPI_Op * op);
   int   MPIAPI HMPI_Reduce_scatter(void * sendbuf,void * recvbuf,int * recvcnts,
						MPI_Datatype datatype,MPI_Op op, MPI_Comm comm);
   int   MPIAPI HMPI_Reduce(void * sendbuf,void * recvbuf,int count,MPI_Datatype datatype,
				MPI_Op op,int root,MPI_Comm comm);
   int   MPIAPI HMPI_Scan(void * sendbuf,void * recvbuf,int count,
				MPI_Datatype datatype,MPI_Op op,MPI_Comm comm);
   int  HMPI_Scatter(void * sendbuf,int sendcnt,MPI_Datatype sendtype,
				void * recvbuf,int recvcnt,MPI_Datatype recvtype,int root);
   int  HMPI_Scatterv(void * sendbuf,int * sendcnts,int * displs,MPI_Datatype sendtype,
				void * recvbuf,int recvcnt,MPI_Datatype recvtype,int root);
   int    MPIAPI HMPI_Attr_delete(MPI_Comm comm,int keyval);
   int   MPIAPI HMPI_Attr_get(MPI_Comm comm,int keyval,void * attr_value,int * flag);
   int    MPIAPI HMPI_Attr_put(MPI_Comm comm,int keyval,void * attr_value);
   int    MPIAPI HMPI_Comm_compare(MPI_Comm comm1,MPI_Comm comm2,int * result);
   int    MPIAPI HMPI_Comm_create(MPI_Comm comm,MPI_Group group,MPI_Comm * comm_out);
   int    MPIAPI HMPI_Comm_dup(MPI_Comm comm,MPI_Comm * comm_out);
   int    MPIAPI HMPI_Comm_free(MPI_Comm * comm);
   int    MPIAPI HMPI_Comm_group(MPI_Comm comm,MPI_Group * group);
   int    MPIAPI HMPI_Comm_rank(MPI_Comm comm,int * rank);
   int    MPIAPI HMPI_Comm_remote_group(MPI_Comm comm,MPI_Group * group);
   int    MPIAPI HMPI_Comm_remote_size(MPI_Comm comm,int * size);
   int    MPIAPI HMPI_Comm_size(MPI_Comm comm,int * size);
   int    MPIAPI HMPI_Comm_split(MPI_Comm comm,int color,int key,MPI_Comm * comm_out);
   int    MPIAPI HMPI_Comm_test_inter(MPI_Comm comm,int * flag);
   int    MPIAPI HMPI_Group_compare(MPI_Group group1,MPI_Group group2,int * result);
   int    MPIAPI HMPI_Group_difference(MPI_Group group1,MPI_Group group2,MPI_Group * group_out);
   int    MPIAPI HMPI_Group_excl(MPI_Group group,int n,int * ranks,MPI_Group * newgroup);
   int    MPIAPI HMPI_Group_free(MPI_Group * group);
   int    MPIAPI HMPI_Group_incl(MPI_Group group,int n,int * ranks,MPI_Group * group_out);
   int    MPIAPI HMPI_Group_intersection(MPI_Group group1,MPI_Group group2,MPI_Group * group_out);
   int    MPIAPI HMPI_Group_rank(MPI_Group group,int * rank);
   int    MPIAPI HMPI_Group_range_excl(MPI_Group group,int n,int ranges[][3],MPI_Group * newgroup);
   int    MPIAPI HMPI_Group_range_incl(MPI_Group group,int n,int ranges[][3],MPI_Group * newgroup);
   int    MPIAPI HMPI_Group_size(MPI_Group group,int * size);
   int    MPIAPI HMPI_Group_translate_ranks(MPI_Group group_a,int n,int * ranks_a,
												MPI_Group group_b,int * ranks_b);
   int    MPIAPI HMPI_Group_union(MPI_Group group1,MPI_Group group2,MPI_Group * group_out);
   int    MPIAPI HMPI_Intercomm_create(MPI_Comm local_comm,int local_leader,MPI_Comm peer_comm,
											int remote_leader,int tag,MPI_Comm * comm_out);
   int    MPIAPI HMPI_Intercomm_merge(MPI_Comm comm,int high,MPI_Comm * comm_out);
   int    MPIAPI HMPI_Keyval_create(MPI_Copy_function * copy_fn,MPI_Delete_function * delete_fn,
										int * keyval,void * extra_state);
   int    MPIAPI HMPI_Keyval_free(int * keyval);
   int   MPIAPI HMPI_Abort(MPI_Comm comm,int errorcode);
   int   MPIAPI HMPI_Error_class(int errorcode,int * errorclass);
   int   MPIAPI HMPI_Errhandler_create(MPI_Handler_function * function,MPI_Errhandler * errhandler);
   int   MPIAPI HMPI_Errhandler_free(MPI_Errhandler * errhandler);
   int   MPIAPI HMPI_Errhandler_get(MPI_Comm comm,MPI_Errhandler * errhandler);
   int   MPIAPI HMPI_Error_string(int errorcode,char * string,int * resultlen);
   int   MPIAPI HMPI_Errhandler_set(MPI_Comm comm,MPI_Errhandler errhandler);
   int   MPIAPI HMPI_Finalize();
   int   MPIAPI HMPI_Get_processor_name(char * name,int * resultlen);
   int   MPIAPI HMPI_Init(int * argc,char *** argv);
   int   MPIAPI HMPI_Init_thread (int * argc,char *** argv,int required,int *provided);
   double  MPIAPI HMPI_Wtime();
   int   MPIAPI HMPI_Address(void * location,MPI_Aint * address);
   int   MPIAPI HMPI_Bsend(void * buf,int count,MPI_Datatype datatype,
								int dest,int tag,MPI_Comm comm);
   int   MPIAPI HMPI_Bsend_init(void * buf,int count,MPI_Datatype datatype,int dest,
									int tag,MPI_Comm comm,MPI_Request * request);
   int   MPIAPI HMPI_Buffer_attach(void * buffer,int size);
   int   MPIAPI HMPI_Buffer_detach(void * buffer,int * size);
   int   MPIAPI HMPI_Cancel(MPI_Request * request);
   int   MPIAPI HMPI_Request_free(MPI_Request * request);
   int   MPIAPI HMPI_Recv_init(void * buf,int count,MPI_Datatype datatype,
									int source,int tag,MPI_Comm comm,MPI_Request * request);
   int   MPIAPI HMPI_Send_init(void * buf,int count,MPI_Datatype datatype,int dest,
									int tag,MPI_Comm comm,MPI_Request * request);
   int    MPIAPI HMPI_Get_elements(MPI_Status * status,MPI_Datatype datatype,int * elements);
   int   MPIAPI HMPI_Get_count(MPI_Status * status,MPI_Datatype datatype,int * count);
   int   MPIAPI HMPI_Ibsend(void * buf,int count,MPI_Datatype datatype,int dest,
								int tag,MPI_Comm comm,MPI_Request * request);
   int   MPIAPI HMPI_Iprobe(int source,int tag,MPI_Comm comm,int * flag,MPI_Status * status);
   int   MPIAPI HMPI_Irecv(void * buf,int count,MPI_Datatype datatype,int source,
								int tag,MPI_Comm comm,MPI_Request * request);
   int   MPIAPI HMPI_Irsend( void * buf,int count,MPI_Datatype datatype,int dest,
								int tag,MPI_Comm comm,MPI_Request * request);
   int   MPIAPI HMPI_Isend( void * buf,int count,MPI_Datatype datatype,int dest,
								int tag,MPI_Comm comm,MPI_Request * request);
   int   MPIAPI HMPI_Issend(void * buf,int count,MPI_Datatype datatype,int dest,
								int tag,MPI_Comm comm,MPI_Request * request);
   int    MPIAPI HMPI_Pack( void * inbuf,int incount,MPI_Datatype type,void * outbuf,
								int outcount,int * position,MPI_Comm comm);
   int    MPIAPI HMPI_Pack_size(int incount,MPI_Datatype datatype,MPI_Comm comm,int * size);
   int   MPIAPI HMPI_Probe( int source,int tag,MPI_Comm comm,MPI_Status * status);
   int   MPIAPI HMPI_Recv(void * buf,int count,MPI_Datatype datatype,int source,
								int tag,MPI_Comm comm,MPI_Status * status);
   int   MPIAPI HMPI_Rsend(void * buf,int count,MPI_Datatype datatype,int dest,int tag,MPI_Comm comm);
   int   MPIAPI HMPI_Rsend_init(void * buf,int count,MPI_Datatype datatype,int dest,
									int tag,MPI_Comm comm,MPI_Request * request);
   int   MPIAPI HMPI_Send( void * buf,int count,MPI_Datatype datatype,int dest,int tag,MPI_Comm comm);
   int   MPIAPI HMPI_Sendrecv( void * sendbuf,int sendcount,MPI_Datatype sendtype,int dest,
									int sendtag,void * recvbuf,int recvcount,MPI_Datatype recvtype,
									int source,int recvtag,MPI_Comm comm,MPI_Status * status);
   int   MPIAPI HMPI_Sendrecv_replace(void * buf,int count,MPI_Datatype datatype,int dest,int sendtag,
										int source,int recvtag,MPI_Comm comm,MPI_Status * status);
   int   MPIAPI HMPI_Ssend( void * buf,int count,MPI_Datatype datatype,int dest,int tag,MPI_Comm comm);
   int   MPIAPI HMPI_Ssend_init( void * buf,int count,MPI_Datatype datatype,int dest,
									int tag,MPI_Comm comm,MPI_Request * request);
   int   MPIAPI HMPI_Start(MPI_Request * request);
   int   MPIAPI HMPI_Startall(int count,MPI_Request * array_of_requests);
   int    MPIAPI HMPI_Test( MPI_Request * request,int * flag,MPI_Status * status);
   int   MPIAPI HMPI_Testall( int count,MPI_Request * array_of_requests,int * flag,MPI_Status * array_of_statuses);
   int   MPIAPI HMPI_Testany( int count,MPI_Request * array_of_requests,int * index,
								 int * flag,MPI_Status * status);
   int   MPIAPI HMPI_Test_cancelled(MPI_Status * status,int * flag);
   int   MPIAPI HMPI_Testsome(int incount,MPI_Request * array_of_requests,int * outcount,
									int * array_of_indices,MPI_Status * array_of_statuses);
   int   MPIAPI HMPI_Type_commit(MPI_Datatype * datatype);
   int   MPIAPI HMPI_Type_contiguous( int count,MPI_Datatype old_type,MPI_Datatype * newtype);
   int   MPIAPI HMPI_Type_extent(MPI_Datatype datatype,MPI_Aint * extent);
   int    MPIAPI HMPI_Type_free(MPI_Datatype * datatype);
   int   MPIAPI HMPI_Type_hindexed(int count,int * blocklens,MPI_Aint * indices,
										MPI_Datatype old_type,MPI_Datatype * newtype);
   int   MPIAPI HMPI_Type_hvector( int count,int blocklen,MPI_Aint stride,
										MPI_Datatype old_type,MPI_Datatype * newtype);
   int   MPIAPI HMPI_Type_indexed( int count,int * blocklens,int * indices,
										MPI_Datatype old_type,MPI_Datatype * newtype);
   int    MPIAPI HMPI_Type_lb(MPI_Datatype datatype,MPI_Aint * displacement);
   int    MPIAPI HMPI_Type_size(MPI_Datatype datatype,int * size);
   int   MPIAPI HMPI_Type_struct(int count,int * blocklens,MPI_Aint * indices,
									MPI_Datatype * old_types,MPI_Datatype * newtype);
   int    MPIAPI HMPI_Type_ub(MPI_Datatype datatype,MPI_Aint * displacement);
   int   MPIAPI HMPI_Type_vector(int count,int blocklen,int stride,
									MPI_Datatype old_type,MPI_Datatype * newtype);
   int   MPIAPI HMPI_Unpack(void * inbuf,int insize,int * position,void * outbuf,
								int outcount,MPI_Datatype type,MPI_Comm comm);
   int    MPIAPI HMPI_Wait(MPI_Request * request,MPI_Status * status);
   int   MPIAPI HMPI_Waitall(int count,MPI_Request * array_of_requests,MPI_Status * array_of_statuses);
   int   MPIAPI HMPI_Waitany(int count,MPI_Request * array_of_requests,int * index,MPI_Status * status);
   int   MPIAPI HMPI_Waitsome( int incount,MPI_Request * array_of_requests,int * outcount,
									int * array_of_indices,MPI_Status * array_of_statuses);
   int    MPIAPI HMPI_Cart_coords(MPI_Comm comm,int rank,int maxdims,int * coords);
   int    MPIAPI HMPI_Cart_create(MPI_Comm comm_old,int ndims,int * dims,
									 int * periods,int reorder,MPI_Comm * comm_cart);
   int  MPIAPI HMPI_Cart_get(MPI_Comm comm,int maxdims,int * dims,int * periods,int * coords);
   int  MPIAPI HMPI_Cart_map(MPI_Comm comm_old,int ndims,int * dims,int * periods,int * newrank);
   int   MPIAPI HMPI_Cart_rank(MPI_Comm comm,int * coords,int * rank);
   int   MPIAPI HMPI_Cart_shift(MPI_Comm comm,int direction,int displ,int * source,int * dest);
   int   MPIAPI HMPI_Cart_sub(MPI_Comm comm,int * remain_dims,MPI_Comm * comm_new);
   int   MPIAPI HMPI_Cartdim_get(MPI_Comm comm,int * ndims);
   int   MPIAPI HMPI_Dims_create(int nnodes,int ndims,int * dims);
   int    MPIAPI HMPI_Graph_create(MPI_Comm comm_old,int nnodes,int * index,int * edges,
										int reorder,MPI_Comm * comm_graph);
   int    MPIAPI HMPI_Graph_get(MPI_Comm comm,int maxindex,
									int maxedges,int * index,int * edges);
   int    MPIAPI HMPI_Graph_map(MPI_Comm comm_old,int nnodes,int * index,int * edges,int * newrank);
   int   MPIAPI HMPI_Graph_neighbors(MPI_Comm comm,int rank,int  maxneighbors,int * neighbors);
   int   MPIAPI HMPI_Graph_neighbors_count(MPI_Comm comm,int rank,int * nneighbors);
   int   MPIAPI HMPI_Graphdims_get(MPI_Comm comm,int * nnodes,int * nedges);
   int   MPIAPI HMPI_Topo_test(MPI_Comm comm,int * top_type);

   void DumpTrace();
   void StartProfile(char* rtn, void **tauptr); 
   void EndProfile(char *rtn,void * tau);
   void MpiSetUp();
   void TauTest();
   void SetupProfileFile();
};






