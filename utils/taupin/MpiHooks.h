#pragma once
#include <mpi.h>
#define MPI_LIB "msmpi.dll"

//start defining all the function prototypes for MPI that can be used in 
// They are required for routine replacement with signed and unsigned ways 

#define MPI_Allgather_FNAME "MPI_Allgather"
typedef int (*MPI_Allgather_FPTR)( void * sendbuf, int sendcount,MPI_Datatype sendtype, 
					void * recvbuf,int recvcount,MPI_Datatype recvtype,MPI_Comm comm);

#define MPI_Allgatherv_FNAME "MPI_Allgatherv"
typedef int(*MPI_Allgatherv_FPTR)(void * sendbuf,int sendcount,MPI_Datatype sendtype,
					 void * recvbuf,int * recvcounts,int * displs,
					  MPI_Datatype recvtype, MPI_Comm comm);

#define MPI_Allreduce_FNAME "MPI_Allreduce"
typedef int (*MPI_Allreduce_FPTR)(void * sendbuf,void * recvbuf,int count,
					MPI_Datatype datatype,MPI_Op op,MPI_Comm comm);

#define MPI_Alltoall_FNAME "MPI_Alltoall"
typedef int  (*MPI_Alltoall_FPTR)(void * sendbuf,int sendcount,MPI_Datatype sendtype,
					void * recvbuf,int recvcnt,MPI_Datatype recvtype,MPI_Comm comm);

#define MPI_Alltoallv_FNAME "MPI_Alltoallv"
typedef int (*MPI_Alltoallv_FPTR)(void * sendbuf,int * sendcnts,int * sdispls,MPI_Datatype sendtype,
				void * recvbuf,int * recvcnts,int * rdispls,MPI_Datatype recvtype,MPI_Comm comm);

#define MPI_Barrier_FNAME "MPI_Barrier"
typedef int  (*MPI_Barrier_FPTR)(MPI_Comm comm);

#define MPI_Bcast_FNAME "MPI_Bcast"
typedef int  (*MPI_Bcast_FPTR)( void * buffer,int count,MPI_Datatype datatype,int root,MPI_Comm comm);

#define MPI_Gather_FNAME "MPI_Gather"
typedef int  (*MPI_Gather_FPTR)(void * sendbuf,int sendcnt,MPI_Datatype sendtype,void * recvbuf,
				int recvcount,MPI_Datatype recvtype,int root,MPI_Comm comm);

#define MPI_Gatherv_FNAME "MPI_Gatherv"
typedef int  (*MPI_Gatherv_FPTR)(void * sendbuf,int sendcnt,MPI_Datatype sendtype,void * recvbuf,
				int * recvcnts,int * displs,MPI_Datatype recvtype,int root,MPI_Comm comm);

#define MPI_Op_create_FNAME "MPI_Op_create"
typedef int (*MPI_Op_create_FPTR)(MPI_User_function * function,int commute,MPI_Op * op);

#define MPI_Op_free_FNAME "MPI_Op_free"
typedef int  (*MPI_Op_free_FPTR)(MPI_Op * op);

#define MPI_Reduce_scatter_FNAME "MPI_Reduce_scatter"
typedef int  (*MPI_Reduce_scatter_FPTR)(void * sendbuf,void * recvbuf,int * recvcnts,
						MPI_Datatype datatype,MPI_Op op, MPI_Comm comm);

#define MPI_Reduce_FNAME "MPI_Reduce"
typedef int  (*MPI_Reduce_FPTR)(void * sendbuf,void * recvbuf,int count,MPI_Datatype datatype,
				MPI_Op op,int root,MPI_Comm comm);

#define MPI_Scan_FNAME "MPI_Scan"
typedef int  (*MPI_Scan_FPTR)(void * sendbuf,void * recvbuf,int count,
				MPI_Datatype datatype,MPI_Op op,MPI_Comm comm);

#define MPI_Scatter_FNAME "MPI_Scatter"
typedef int (* MPI_Scatter_FPTR)(void * sendbuf,int sendcnt,MPI_Datatype sendtype,
				void * recvbuf,int recvcnt,MPI_Datatype recvtype,int root);

#define MPI_Scatterv_FNAME "MPI_Scatterv"
typedef int  (*MPI_Scatterv_FPTR)(void * sendbuf,int * sendcnts,int * displs,MPI_Datatype sendtype,
				void * recvbuf,int recvcnt,MPI_Datatype recvtype,int root);


#define MPI_Attr_delete_FNAME "MPI_Attr_delete"
typedef int   (*MPI_Attr_delete_FPTR)(MPI_Comm comm,int keyval);

#define MPI_Attr_get_FNAME "MPI_Attr_get"
typedef int  (*MPI_Attr_get_FPTR)(MPI_Comm comm,int keyval,void * attr_value,int * flag);

#define MPI_Attr_put_FNAME "MPI_Attr_put"
typedef int   (*MPI_Attr_put_FPTR)(MPI_Comm comm,int keyval,void * attr_value);

#define MPI_Comm_compare_FNAME "MPI_Comm_compare"
typedef int   (*MPI_Comm_compare_FPTR)(MPI_Comm comm1,MPI_Comm comm2,int * result);

#define MPI_Comm_create_FNAME "MPI_Comm_create"
typedef int   (*MPI_Comm_create_FPTR)(MPI_Comm comm,MPI_Group group,MPI_Comm * comm_out);

#define MPI_Comm_dup_FNAME "MPI_Comm_dup"
typedef int   (*MPI_Comm_dup_FPTR)(MPI_Comm comm,MPI_Comm * comm_out);

#define MPI_Comm_free_FNAME "MPI_Comm_free"
typedef int   (*MPI_Comm_free_FPTR)(MPI_Comm * comm);
#define MPI_Comm_group_FNAME "MPI_Comm_group"
typedef int   (*MPI_Comm_group_FPTR)(MPI_Comm comm,MPI_Group * group);
#define MPI_Comm_rank_FNAME "MPI_Comm_rank"
typedef int   (*MPI_Comm_rank_FPTR)(MPI_Comm comm,int * rank);

#define MPI_Comm_remote_group_FNAME "MPI_Comm_remote_group"
typedef int   (*MPI_Comm_remote_group_FPTR)(MPI_Comm comm,MPI_Group * group);
#define MPI_Comm_remote_size_FNAME "MPI_Comm_remote_size"
typedef int   (*MPI_Comm_remote_size_FPTR)(MPI_Comm comm,int * size);
#define MPI_Comm_size_FNAME "MPI_Comm_size"
typedef int   (*MPI_Comm_size_FPTR)(MPI_Comm comm,int * size);
#define MPI_Comm_split_FNAME "MPI_Comm_split"
typedef int   (*MPI_Comm_split_FPTR)(MPI_Comm comm,int color,int key,MPI_Comm * comm_out);
#define MPI_Comm_test_inter_FNAME "MPI_Comm_test_inter"
typedef int   (*MPI_Comm_test_inter_FPTR)(MPI_Comm comm,int * flag);

#define MPI_Group_compare_FNAME "MPI_Group_compare"
typedef int   (*MPI_Group_compare_FPTR)(MPI_Group group1,MPI_Group group2,int * result);
#define MPI_Group_difference_FNAME "MPI_Group_difference"
typedef int   (*MPI_Group_difference_FPTR)(MPI_Group group1,MPI_Group group2,MPI_Group * group_out);
#define MPI_Group_excl_FNAME "MPI_Group_excl"
typedef int   (*MPI_Group_excl_FPTR)(MPI_Group group,int n,int * ranks,MPI_Group * newgroup);
#define MPI_Group_free_FNAME "MPI_Group_free"
typedef int   (*MPI_Group_free_FPTR)(MPI_Group * group);
#define MPI_Group_incl_FNAME "MPI_Group_incl"
typedef int   (*MPI_Group_incl_FPTR)(MPI_Group group,int n,int * ranks,MPI_Group * group_out);
#define MPI_Group_intersection_FNAME "MPI_Group_intersection"
typedef int   (*MPI_Group_intersection_FPTR)(MPI_Group group1,MPI_Group group2,MPI_Group * group_out);
#define MPI_Group_rank_FNAME "MPI_Group_rank"
typedef int   (*MPI_Group_rank_FPTR)(MPI_Group group,int * rank);
#define MPI_Group_range_excl_FNAME "MPI_Group_range_excl"
typedef int   (*MPI_Group_range_excl_FPTR)(MPI_Group group,int n,int ranges[][3],MPI_Group * newgroup);
#define MPI_Group_range_incl_FNAME "MPI_Group_range_incl"
typedef int   (*MPI_Group_range_incl_FPTR)(MPI_Group group,int n,int ranges[][3],MPI_Group * newgroup);
#define MPI_Group_size_FNAME "MPI_Group_size"
typedef int   (*MPI_Group_size_FPTR)(MPI_Group group,int * size);
#define MPI_Group_translate_ranks_FNAME "MPI_Group_translate_ranks"
typedef int   (*MPI_Group_translate_ranks_FPTR)(MPI_Group group_a,int n,int * ranks_a,
												MPI_Group group_b,int * ranks_b);
#define MPI_Group_union_FNAME "MPI_Group_union"
typedef int   (*MPI_Group_union_FPTR)(MPI_Group group1,MPI_Group group2,MPI_Group * group_out);
#define MPI_Intercomm_create_FNAME "MPI_Intercomm_create"
typedef int   (*MPI_Intercomm_create_FPTR)(MPI_Comm local_comm,int local_leader,MPI_Comm peer_comm,
											int remote_leader,int tag,MPI_Comm * comm_out);
#define MPI_Intercomm_merge_FNAME "MPI_Intercomm_merge"
typedef int   (*MPI_Intercomm_merge_FPTR)(MPI_Comm comm,int high,MPI_Comm * comm_out);
#define MPI_Keyval_create_FNAME "MPI_Keyval_create"
typedef int   (*MPI_Keyval_create_FPTR)(MPI_Copy_function * copy_fn,MPI_Delete_function * delete_fn,
										int * keyval,void * extra_state);
#define MPI_Keyval_free_FNAME "MPI_Keyval_free"
typedef int   (*MPI_Keyval_free_FPTR)(int * keyval);
#define MPI_Abort_FNAME "MPI_Abort"
typedef int  (*MPI_Abort_FPTR)(MPI_Comm comm,int errorcode);

#define MPI_Error_class_FNAME "MPI_Error_class"
typedef int  (*MPI_Error_class_FPTR)(int errorcode,int * errorclass);
#define MPI_Errhandler_create_FNAME "MPI_Errhandler_create"
typedef int  (*MPI_Errhandler_create_FPTR)(MPI_Handler_function * function,MPI_Errhandler * errhandler);
#define MPI_Errhandler_free_FNAME "MPI_Errhandler_free"
typedef int  (*MPI_Errhandler_free_FPTR)(MPI_Errhandler * errhandler);
#define MPI_Errhandler_get_FNAME "MPI_Errhandler_get"
typedef int  (*MPI_Errhandler_get_FPTR)(MPI_Comm comm,MPI_Errhandler * errhandler);
#define MPI_Error_string_FNAME "MPI_Error_string"
typedef int  (*MPI_Error_string_FPTR)(int errorcode,char * string,int * resultlen);
#define MPI_Errhandler_set_FNAME "MPI_Errhandler_set"
typedef int  (*MPI_Errhandler_set_FPTR)(MPI_Comm comm,MPI_Errhandler errhandler);
#define MPI_Finalize_FNAME "MPI_Finalize"
typedef int  (*MPI_Finalize_FPTR)();
#define MPI_Get_processor_name_FNAME "MPI_Get_processor_name"
typedef int  (*MPI_Get_processor_name_FPTR)(char * name,int * resultlen);
#define MPI_Init_FNAME "MPI_Init"
typedef int  (*MPI_Init_FPTR)(int * argc,char *** argv);
#define MPI_Init_thread_FNAME "MPI_Init_thread"
typedef int  (*MPI_Init_thread_FPTR) (int * argc,char *** argv,int required,int *provided);

#define MPI_Wtime_FNAME "MPI_Wtime" 
typedef double (*MPI_Wtime_FPTR)();

#define MPI_Address_FNAME "MPI_Address"
typedef int  (*MPI_Address_FPTR)(void * location,MPI_Aint * address);
#define MPI_Bsend_FNAME "MPI_Bsend"
typedef int  (*MPI_Bsend_FPTR)(void * buf,int count,MPI_Datatype datatype,
								int dest,int tag,MPI_Comm comm);
#define MPI_Bsend_init_FNAME "MPI_Bsend_init"
typedef int  (*MPI_Bsend_init_FPTR)(void * buf,int count,MPI_Datatype datatype,int dest,
									int tag,MPI_Comm comm,MPI_Request * request);
#define MPI_Buffer_attach_FNAME "MPI_Buffer_attach"
typedef int  (*MPI_Buffer_attach_FPTR)(void * buffer,int size);
#define MPI_Buffer_detach_FNAME "MPI_Buffer_detach"
typedef int  (*MPI_Buffer_detach_FPTR)(void * buffer,int * size);
#define MPI_Cancel_FNAME "MPI_Cancel"
typedef int  (*MPI_Cancel_FPTR)(MPI_Request * request);
#define MPI_Request_free_FNAME "MPI_Request_free"
typedef int  (*MPI_Request_free_FPTR)(MPI_Request * request);
#define MPI_Recv_init_FNAME "MPI_Recv_init"
typedef int  (*MPI_Recv_init_FPTR)(void * buf,int count,MPI_Datatype datatype,
									int source,int tag,MPI_Comm comm,MPI_Request * request);
#define MPI_Send_init_FNAME "MPI_Send_init"
typedef int  (*MPI_Send_init_FPTR)(void * buf,int count,MPI_Datatype datatype,int dest,
									int tag,MPI_Comm comm,MPI_Request * request);
#define MPI_Get_elements_FNAME "MPI_Get_elements"
typedef int   (*MPI_Get_elements_FPTR)(MPI_Status * status,MPI_Datatype datatype,int * elements);
#define MPI_Get_count_FNAME "MPI_Get_count"
typedef int  (*MPI_Get_count_FPTR)(MPI_Status * status,MPI_Datatype datatype,int * count);
#define MPI_Ibsend_FNAME "MPI_Ibsend"
typedef int  (*MPI_Ibsend_FPTR)(void * buf,int count,MPI_Datatype datatype,int dest,
								int tag,MPI_Comm comm,MPI_Request * request);
#define MPI_Iprobe_FNAME "MPI_Iprobe"
typedef int  (*MPI_Iprobe_FPTR)(int source,int tag,MPI_Comm comm,int * flag,MPI_Status * status);
#define MPI_Irecv_FNAME "MPI_Irecv"
typedef int  (*MPI_Irecv_FPTR)(void * buf,int count,MPI_Datatype datatype,int source,
								int tag,MPI_Comm comm,MPI_Request * request);
#define MPI_Irsend_FNAME "MPI_Irsend"
typedef int  (*MPI_Irsend_FPTR)( void * buf,int count,MPI_Datatype datatype,int dest,
								int tag,MPI_Comm comm,MPI_Request * request);
#define MPI_Isend_FNAME "MPI_Isend"
typedef int  (*MPI_Isend_FPTR)( void * buf,int count,MPI_Datatype datatype,int dest,
								int tag,MPI_Comm comm,MPI_Request * request);
#define MPI_Issend_FNAME "MPI_Issend"
typedef int  (*MPI_Issend_FPTR)(void * buf,int count,MPI_Datatype datatype,int dest,
								int tag,MPI_Comm comm,MPI_Request * request);
#define MPI_Pack_FNAME "MPI_Pack"
typedef int   (*MPI_Pack_FPTR)( void * inbuf,int incount,MPI_Datatype type,void * outbuf,
								int outcount,int * position,MPI_Comm comm);
#define MPI_Pack_size_FNAME "MPI_Pack_size"
typedef int   (*MPI_Pack_size_FPTR)(int incount,MPI_Datatype datatype,MPI_Comm comm,int * size);
#define MPI_Probe_FNAME "MPI_Probe"
typedef int  (*MPI_Probe_FPTR)( int source,int tag,MPI_Comm comm,MPI_Status * status);
#define MPI_Recv_FNAME "MPI_Recv"
typedef int  (*MPI_Recv_FPTR)(void * buf,int count,MPI_Datatype datatype,int source,
								int tag,MPI_Comm comm,MPI_Status * status);
#define MPI_Rsend_FNAME "MPI_Rsend"
typedef int  (*MPI_Rsend_FPTR)(void * buf,int count,MPI_Datatype datatype,int dest,int tag,MPI_Comm comm);
#define MPI_Rsend_init_FNAME "MPI_Rsend_init"
typedef int  (*MPI_Rsend_init_FPTR)(void * buf,int count,MPI_Datatype datatype,int dest,
									int tag,MPI_Comm comm,MPI_Request * request);
#define MPI_Send_FNAME "MPI_Send"
typedef int  (*MPI_Send_FPTR)( void * buf,int count,MPI_Datatype datatype,int dest,int tag,MPI_Comm comm);
#define MPI_Sendrecv_FNAME "MPI_Sendrecv"
typedef int  (*MPI_Sendrecv_FPTR)( void * sendbuf,int sendcount,MPI_Datatype sendtype,int dest,
									int sendtag,void * recvbuf,int recvcount,MPI_Datatype recvtype,
									int source,int recvtag,MPI_Comm comm,MPI_Status * status);
#define MPI_Sendrecv_replace_FNAME "MPI_Sendrecv_replace"
typedef int  (*MPI_Sendrecv_replace_FPTR)(void * buf,int count,MPI_Datatype datatype,int dest,int sendtag,
										int source,int recvtag,MPI_Comm comm,MPI_Status * status);

#define MPI_Ssend_FNAME "MPI_Ssend"
typedef int  (*MPI_Ssend_FPTR)( void * buf,int count,MPI_Datatype datatype,int dest,int tag,MPI_Comm comm);

#define MPI_Ssend_init_FNAME "MPI_Ssend_init"
typedef int  (*MPI_Ssend_init_FPTR)( void * buf,int count,MPI_Datatype datatype,int dest,
									int tag,MPI_Comm comm,MPI_Request * request);

#define MPI_Start_FNAME "MPI_Start"
typedef int  (*MPI_Start_FPTR)(MPI_Request * request);
#define MPI_Startall_FNAME "MPI_Startall"
typedef int  (*MPI_Startall_FPTR)(int count,MPI_Request * array_of_requests);
#define MPI_Test_FNAME "MPI_Test"
typedef int   (*MPI_Test_FPTR)( MPI_Request * request,int * flag,MPI_Status * status);
#define MPI_Testall_FNAME "MPI_Testall"
typedef int  (*MPI_Testall_FPTR)( int count,MPI_Request * array_of_requests,int * flag,MPI_Status * array_of_statuses);
#define MPI_Testany_FNAME "MPI_Testany"
typedef int  (*MPI_Testany_FPTR)( int count,MPI_Request * array_of_requests,int * index,
								 int * flag,MPI_Status * status);

#define MPI_Test_cancelled_FNAME "MPI_Test_cancelled"
typedef int  (*MPI_Test_cancelled_FPTR)(MPI_Status * status,int * flag);
#define MPI_Testsome_FNAME "MPI_Testsome" ///****
typedef int  (*MPI_Testsome_FPTR)(int incount,MPI_Request * array_of_requests,int * outcount,
									int * array_of_indices,MPI_Status * array_of_statuses);
#define MPI_Type_commit_FNAME "MPI_Type_commit"
typedef int  (*MPI_Type_commit_FPTR)(MPI_Datatype * datatype);
#define MPI_Type_contiguous_FNAME "MPI_Type_contiguous"
typedef int  (*MPI_Type_contiguous_FPTR)( int count,MPI_Datatype old_type,MPI_Datatype * newtype);
#define MPI_Type_extent_FNAME "MPI_Type_extent"
typedef int  (*MPI_Type_extent_FPTR)(MPI_Datatype datatype,MPI_Aint * extent);
#define MPI_Type_free_FNAME "MPI_Type_free"
typedef int   (*MPI_Type_free_FPTR)(MPI_Datatype * datatype);
#define MPI_Type_hindexed_FNAME "MPI_Type_hindexed"
typedef int  (*MPI_Type_hindexed_FPTR)(int count,int * blocklens,MPI_Aint * indices,
										MPI_Datatype old_type,MPI_Datatype * newtype);
#define MPI_Type_hvector_FNAME "MPI_Type_hvector"
typedef int  (*MPI_Type_hvector_FPTR)( int count,int blocklen,MPI_Aint stride,
										MPI_Datatype old_type,MPI_Datatype * newtype);
#define MPI_Type_indexed_FNAME "MPI_Type_indexed"
typedef int  (*MPI_Type_indexed_FPTR)( int count,int * blocklens,int * indices,
										MPI_Datatype old_type,MPI_Datatype * newtype);
#define MPI_Type_lb_FNAME "MPI_Type_lb"
typedef int   (*MPI_Type_lb_FPTR)(MPI_Datatype datatype,MPI_Aint * displacement);
#define MPI_Type_size_FNAME "MPI_Type_size"
typedef int   (*MPI_Type_size_FPTR)(MPI_Datatype datatype,int * size);
#define MPI_Type_struct_FNAME "MPI_Type_struct"
typedef int  (*MPI_Type_struct_FPTR)(int count,int * blocklens,MPI_Aint * indices,
									MPI_Datatype * old_types,MPI_Datatype * newtype);
#define MPI_Type_ub_FNAME "MPI_Type_ub"
typedef int   (*MPI_Type_ub_FPTR)(MPI_Datatype datatype,MPI_Aint * displacement);
#define MPI_Type_vector_FNAME "MPI_Type_vector"
typedef int  (*MPI_Type_vector_FPTR)(int count,int blocklen,int stride,
									MPI_Datatype old_type,MPI_Datatype * newtype);
#define MPI_Unpack_FNAME "MPI_Unpack"
typedef int  (*MPI_Unpack_FPTR)(void * inbuf,int insize,int * position,void * outbuf,
								int outcount,MPI_Datatype type,MPI_Comm comm);
#define MPI_Wait_FNAME "MPI_Wait"
typedef int   (*MPI_Wait_FPTR)(MPI_Request * request,MPI_Status * status);
#define MPI_Waitall_FNAME "MPI_Waitall"
typedef int  (*MPI_Waitall_FPTR)(int count,MPI_Request * array_of_requests,MPI_Status * array_of_statuses);
#define MPI_Waitany_FNAME "MPI_Waitany"
typedef int  (*MPI_Waitany_FPTR)(int count,MPI_Request * array_of_requests,int * index,MPI_Status * status);
#define MPI_Waitsome_FNAME "MPI_Waitsome"
typedef int  (*MPI_Waitsome_FPTR)( int incount,MPI_Request * array_of_requests,int * outcount,
									int * array_of_indices,MPI_Status * array_of_statuses);
#define MPI_Cart_coords_FNAME "MPI_Cart_coords"
typedef int   (*MPI_Cart_coords_FPTR)(MPI_Comm comm,int rank,int maxdims,int * coords);
#define MPI_Cart_create_FNAME "MPI_Cart_create"
typedef int   (*MPI_Cart_create_FPTR)(MPI_Comm comm_old,int ndims,int * dims,
									 int * periods,int reorder,MPI_Comm * comm_cart);
#define MPI_Cart_get_FNAME "MPI_Cart_get"
typedef int (*MPI_Cart_get_FPTR)(MPI_Comm comm,int maxdims,int * dims,int * periods,int * coords);
#define MPI_Cart_map_FNAME "MPI_Cart_map"
typedef int (*MPI_Cart_map_FPTR)(MPI_Comm comm_old,int ndims,int * dims,int * periods,int * newrank);
#define MPI_Cart_rank_FNAME "MPI_Cart_rank"
typedef int  (*MPI_Cart_rank_FPTR)(MPI_Comm comm,int * coords,int * rank);
#define MPI_Cart_shift_FNAME "MPI_Cart_shift"
typedef int  (*MPI_Cart_shift_FPTR)(MPI_Comm comm,int direction,int displ,int * source,int * dest);
#define MPI_Cart_sub_FNAME "MPI_Cart_sub"
typedef int  (*MPI_Cart_sub_FPTR)(MPI_Comm comm,int * remain_dims,MPI_Comm * comm_new);
#define MPI_Cartdim_get_FNAME "MPI_Cartdim_get"
typedef int  (*MPI_Cartdim_get_FPTR)(MPI_Comm comm,int * ndims);
#define MPI_Dims_create_FNAME "MPI_Dims_create"
typedef int  (*MPI_Dims_create_FPTR)(int nnodes,int ndims,int * dims);
#define MPI_Graph_create_FNAME "MPI_Graph_create"
typedef int   (*MPI_Graph_create_FPTR)(MPI_Comm comm_old,int nnodes,int * index,int * edges,
										int reorder,MPI_Comm * comm_graph);
#define MPI_Graph_get_FNAME "MPI_Graph_get"
typedef int   (*MPI_Graph_get_FPTR)(MPI_Comm comm,int maxindex,
									int maxedges,int * index,int * edges);
#define MPI_Graph_map_FNAME "MPI_Graph_map"
typedef int   (*MPI_Graph_map_FPTR)(MPI_Comm comm_old,int nnodes,int * index,int * edges,int * newrank);
#define MPI_Graph_neighbors_FNAME "MPI_Graph_neighbors"
typedef int  (*MPI_Graph_neighbors_FPTR)(MPI_Comm comm,int rank,int  maxneighbors,int * neighbors);
#define MPI_Graph_neighbors_count_FNAME "MPI_Graph_neighbors_count"
typedef int  (*MPI_Graph_neighbors_count_FPTR)(MPI_Comm comm,int rank,int * nneighbors);
#define MPI_Graphdims_get_FNAME "MPI_Graphdims_get"
typedef int  (*MPI_Graphdims_get_FPTR)(MPI_Comm comm,int * nnodes,int * nedges);
#define MPI_Topo_test_FNAME "MPI_Topo_test"
typedef int  (*MPI_Topo_test_FPTR)(MPI_Comm comm,int * top_type);

