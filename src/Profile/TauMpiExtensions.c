#include <mpi.h>
#include <TAU.h>
#include <unistd.h>
#include <stdlib.h>
#include <Profile/TauUtil.h>
/******************************************************/
/******************************************************/
#ifdef TAU_MPICONSTCHAR
#define TAU_CONST const
#else
#define TAU_CONST
#endif
/******************************************************/
/******************************************************/


/* We need to do different things on BGL! */
#ifdef __blrts__
#define TAU_BGL
#undef TAU_MPIOREQUEST
#endif

#define TAU_READ TAU_IO
#define TAU_WRITE TAU_IO




/******************************************************
***      MPI_Type_get_envelope wrapper function 
******************************************************/
int MPI_Type_get_envelope( MPI_Datatype datatype, int * num_integers, int * num_addresses, int * num_datatypes, int * combiner)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_get_envelope()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_get_envelope( datatype, num_integers, num_addresses, num_datatypes, combiner) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_get_envelope wrapper function 
******************************************************/
void MPI_TYPE_GET_ENVELOPE( MPI_Fint *  datatype, MPI_Fint *  num_integers, MPI_Fint *  num_addresses, MPI_Fint *  num_datatypes, MPI_Fint *  combiner, MPI_Fint * ierr)
{
  *ierr = MPI_Type_get_envelope( MPI_Type_f2c(*datatype), num_integers, num_addresses, num_datatypes, combiner) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_envelope wrapper function 
******************************************************/
void mpi_type_get_envelope( MPI_Fint *  datatype, MPI_Fint *  num_integers, MPI_Fint *  num_addresses, MPI_Fint *  num_datatypes, MPI_Fint *  combiner, MPI_Fint * ierr)
{
  MPI_TYPE_GET_ENVELOPE( datatype, num_integers, num_addresses, num_datatypes, combiner, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_envelope wrapper function 
******************************************************/
void mpi_type_get_envelope_( MPI_Fint *  datatype, MPI_Fint *  num_integers, MPI_Fint *  num_addresses, MPI_Fint *  num_datatypes, MPI_Fint *  combiner, MPI_Fint * ierr)
{
  MPI_TYPE_GET_ENVELOPE( datatype, num_integers, num_addresses, num_datatypes, combiner, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_envelope wrapper function 
******************************************************/
void mpi_type_get_envelope__( MPI_Fint *  datatype, MPI_Fint *  num_integers, MPI_Fint *  num_addresses, MPI_Fint *  num_datatypes, MPI_Fint *  combiner, MPI_Fint * ierr)
{
  MPI_TYPE_GET_ENVELOPE( datatype, num_integers, num_addresses, num_datatypes, combiner, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Type_get_contents wrapper function 
******************************************************/
int MPI_Type_get_contents( MPI_Datatype datatype, int max_integers, int max_addresses, int max_datatypes, int * array_of_integers, MPI_Aint * array_of_addresses, MPI_Datatype * array_of_datatypes)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_get_contents()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_get_contents( datatype, max_integers, max_addresses, max_datatypes, array_of_integers, array_of_addresses, array_of_datatypes) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_get_contents wrapper function 
******************************************************/
void MPI_TYPE_GET_CONTENTS( MPI_Fint *  datatype, MPI_Fint *  max_integers, MPI_Fint *  max_addresses, MPI_Fint *  max_datatypes, MPI_Fint *  array_of_integers, MPI_Aint * array_of_addresses, MPI_Aint * array_of_datatypes, MPI_Fint * ierr)
{
  TAU_DECL_ALLOC_LOCAL(MPI_Datatype, local_types, *max_datatypes);
  *ierr = MPI_Type_get_contents( MPI_Type_f2c(*datatype), *max_integers, *max_addresses, *max_datatypes, array_of_integers, array_of_addresses, local_types) ; 
  TAU_ASSIGN_VALUES(array_of_datatypes, local_types, *max_datatypes, MPI_Type_c2f);
  return ; 
}

/******************************************************
***      MPI_Type_get_contents wrapper function 
******************************************************/
void mpi_type_get_contents( MPI_Fint *  datatype, MPI_Fint *  max_integers, MPI_Fint *  max_addresses, MPI_Fint *  max_datatypes, MPI_Fint *  array_of_integers, MPI_Aint * array_of_addresses, MPI_Aint * array_of_datatypes, MPI_Fint * ierr)
{
  MPI_TYPE_GET_CONTENTS( datatype, max_integers, max_addresses, max_datatypes, array_of_integers, array_of_addresses, array_of_datatypes, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_contents wrapper function 
******************************************************/
void mpi_type_get_contents_( MPI_Fint *  datatype, MPI_Fint *  max_integers, MPI_Fint *  max_addresses, MPI_Fint *  max_datatypes, MPI_Fint *  array_of_integers, MPI_Aint * array_of_addresses, MPI_Aint * array_of_datatypes, MPI_Fint * ierr)
{
  MPI_TYPE_GET_CONTENTS( datatype, max_integers, max_addresses, max_datatypes, array_of_integers, array_of_addresses, array_of_datatypes, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_contents wrapper function 
******************************************************/
void mpi_type_get_contents__( MPI_Fint *  datatype, MPI_Fint *  max_integers, MPI_Fint *  max_addresses, MPI_Fint *  max_datatypes, MPI_Fint *  array_of_integers, MPI_Aint * array_of_addresses, MPI_Aint * array_of_datatypes, MPI_Fint * ierr)
{
  MPI_TYPE_GET_CONTENTS( datatype, max_integers, max_addresses, max_datatypes, array_of_integers, array_of_addresses, array_of_datatypes, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/
#ifdef TAU_MPIATTRFUNCTION


/******************************************************
***      MPI_Type_create_keyval wrapper function 
******************************************************/
int MPI_Type_create_keyval( MPI_Type_copy_attr_function * type_copy_attr_fn, MPI_Type_delete_attr_function * type_delete_attr_fn, int * type_keyval, void * extra_state)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_keyval()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_keyval( type_copy_attr_fn, type_delete_attr_fn, type_keyval, extra_state) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_create_keyval wrapper function 
******************************************************/
void MPI_TYPE_CREATE_KEYVAL( MPI_Type_copy_attr_function * type_copy_attr_fn, MPI_Type_delete_attr_function * type_delete_attr_fn, MPI_Fint *  type_keyval, MPI_Aint * extra_state, MPI_Fint * ierr)
{
  *ierr = MPI_Type_create_keyval( type_copy_attr_fn, type_delete_attr_fn, type_keyval, extra_state) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_keyval wrapper function 
******************************************************/
void mpi_type_create_keyval( MPI_Type_copy_attr_function * type_copy_attr_fn, MPI_Type_delete_attr_function * type_delete_attr_fn, MPI_Fint *  type_keyval, MPI_Aint * extra_state, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_KEYVAL( type_copy_attr_fn, type_delete_attr_fn, type_keyval, extra_state, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_keyval wrapper function 
******************************************************/
void mpi_type_create_keyval_( MPI_Type_copy_attr_function * type_copy_attr_fn, MPI_Type_delete_attr_function * type_delete_attr_fn, MPI_Fint *  type_keyval, MPI_Aint * extra_state, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_KEYVAL( type_copy_attr_fn, type_delete_attr_fn, type_keyval, extra_state, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_keyval wrapper function 
******************************************************/
void mpi_type_create_keyval__( MPI_Type_copy_attr_function * type_copy_attr_fn, MPI_Type_delete_attr_function * type_delete_attr_fn, MPI_Fint *  type_keyval, MPI_Aint * extra_state, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_KEYVAL( type_copy_attr_fn, type_delete_attr_fn, type_keyval, extra_state, ierr) ; 
  return ; 
}

#endif /* TAU_MPIATTRFUNCTION  */

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Type_delete_attr wrapper function 
******************************************************/
int MPI_Type_delete_attr( MPI_Datatype type, int type_keyval)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_delete_attr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_delete_attr( type, type_keyval) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_delete_attr wrapper function 
******************************************************/
void MPI_TYPE_DELETE_ATTR( MPI_Fint *  type, MPI_Fint *  type_keyval, MPI_Fint * ierr)
{
  *ierr = MPI_Type_delete_attr( MPI_Type_f2c(*type), *type_keyval) ; 
  return ; 
}

/******************************************************
***      MPI_Type_delete_attr wrapper function 
******************************************************/
void mpi_type_delete_attr( MPI_Fint *  type, MPI_Fint *  type_keyval, MPI_Fint * ierr)
{
  MPI_TYPE_DELETE_ATTR( type, type_keyval, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_delete_attr wrapper function 
******************************************************/
void mpi_type_delete_attr_( MPI_Fint *  type, MPI_Fint *  type_keyval, MPI_Fint * ierr)
{
  MPI_TYPE_DELETE_ATTR( type, type_keyval, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_delete_attr wrapper function 
******************************************************/
void mpi_type_delete_attr__( MPI_Fint *  type, MPI_Fint *  type_keyval, MPI_Fint * ierr)
{
  MPI_TYPE_DELETE_ATTR( type, type_keyval, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Type_free_keyval wrapper function 
******************************************************/
int MPI_Type_free_keyval( int * type_keyval)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_free_keyval()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_free_keyval( type_keyval) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_free_keyval wrapper function 
******************************************************/
void MPI_TYPE_FREE_KEYVAL( MPI_Fint *  type_keyval, MPI_Fint * ierr)
{
  *ierr = MPI_Type_free_keyval( type_keyval) ; 
  return ; 
}

/******************************************************
***      MPI_Type_free_keyval wrapper function 
******************************************************/
void mpi_type_free_keyval( MPI_Fint *  type_keyval, MPI_Fint * ierr)
{
  MPI_TYPE_FREE_KEYVAL( type_keyval, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_free_keyval wrapper function 
******************************************************/
void mpi_type_free_keyval_( MPI_Fint *  type_keyval, MPI_Fint * ierr)
{
  MPI_TYPE_FREE_KEYVAL( type_keyval, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_free_keyval wrapper function 
******************************************************/
void mpi_type_free_keyval__( MPI_Fint *  type_keyval, MPI_Fint * ierr)
{
  MPI_TYPE_FREE_KEYVAL( type_keyval, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


#ifdef TAU_MPIATTRFUNCTION
/******************************************************
***      MPI_Type_get_attr wrapper function 
******************************************************/
int MPI_Type_get_attr( MPI_Datatype type, int type_keyval, void * attribute_val, int * flag)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_get_attr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_get_attr( type, type_keyval, attribute_val, flag) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_get_attr wrapper function 
******************************************************/
void MPI_TYPE_GET_ATTR( MPI_Fint *  type, MPI_Fint *  type_keyval, MPI_Aint * attribute_val, MPI_Fint *  flag, MPI_Fint * ierr)
{
  *ierr = MPI_Type_get_attr( MPI_Type_f2c(*type), *type_keyval, attribute_val, flag) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_attr wrapper function 
******************************************************/
void mpi_type_get_attr( MPI_Fint *  type, MPI_Fint *  type_keyval, MPI_Aint * attribute_val, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_TYPE_GET_ATTR( type, type_keyval, attribute_val, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_attr wrapper function 
******************************************************/
void mpi_type_get_attr_( MPI_Fint *  type, MPI_Fint *  type_keyval, MPI_Aint * attribute_val, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_TYPE_GET_ATTR( type, type_keyval, attribute_val, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_attr wrapper function 
******************************************************/
void mpi_type_get_attr__( MPI_Fint *  type, MPI_Fint *  type_keyval, MPI_Aint * attribute_val, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_TYPE_GET_ATTR( type, type_keyval, attribute_val, flag, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Type_set_attr wrapper function 
******************************************************/
int MPI_Type_set_attr( MPI_Datatype type, int type_keyval, void * attribute_val)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_set_attr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_set_attr( type, type_keyval, attribute_val) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_set_attr wrapper function 
******************************************************/
void MPI_TYPE_SET_ATTR( MPI_Fint *  type, MPI_Fint *  type_keyval, MPI_Aint * attribute_val, MPI_Fint * ierr)
{
  *ierr = MPI_Type_set_attr( MPI_Type_f2c(*type), *type_keyval, attribute_val) ; 
  return ; 
}

/******************************************************
***      MPI_Type_set_attr wrapper function 
******************************************************/
void mpi_type_set_attr( MPI_Fint *  type, MPI_Fint *  type_keyval, MPI_Aint * attribute_val, MPI_Fint * ierr)
{
  MPI_TYPE_SET_ATTR( type, type_keyval, attribute_val, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_set_attr wrapper function 
******************************************************/
void mpi_type_set_attr_( MPI_Fint *  type, MPI_Fint *  type_keyval, MPI_Aint * attribute_val, MPI_Fint * ierr)
{
  MPI_TYPE_SET_ATTR( type, type_keyval, attribute_val, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_set_attr wrapper function 
******************************************************/
void mpi_type_set_attr__( MPI_Fint *  type, MPI_Fint *  type_keyval, MPI_Aint * attribute_val, MPI_Fint * ierr)
{
  MPI_TYPE_SET_ATTR( type, type_keyval, attribute_val, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/
#endif /* TAU_MPIATTRFUNCTION  */


#ifdef TAU_MPITYPEEX

/******************************************************
***      MPI_Type_dup wrapper function 
******************************************************/
int MPI_Type_dup( MPI_Datatype type, MPI_Datatype * newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_dup()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_dup( type, newtype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_dup wrapper function 
******************************************************/
void MPI_TYPE_DUP( MPI_Fint *  type, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_Datatype local_type;
  *ierr = MPI_Type_dup( MPI_Type_f2c(*type), &local_type) ; 
  *newtype = MPI_Type_c2f(local_type);
  return ; 
}

/******************************************************
***      MPI_Type_dup wrapper function 
******************************************************/
void mpi_type_dup( MPI_Fint *  type, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_DUP( type, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_dup wrapper function 
******************************************************/
void mpi_type_dup_( MPI_Fint *  type, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_DUP( type, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_dup wrapper function 
******************************************************/
void mpi_type_dup__( MPI_Fint *  type, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_DUP( type, newtype, ierr) ; 
  return ; 
}

#endif /* TAU_MPI_TYPEEX */

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Type_create_hindexed wrapper function 
******************************************************/
int MPI_Type_create_hindexed( int count, int * array_of_blocklengths, MPI_Aint * array_of_displacements, MPI_Datatype oldtype, MPI_Datatype * newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_hindexed()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_hindexed( count, array_of_blocklengths, array_of_displacements, oldtype, newtype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_create_hindexed wrapper function 
******************************************************/
void MPI_TYPE_CREATE_HINDEXED( MPI_Fint *  count, MPI_Fint *  array_of_blocklengths, MPI_Aint * array_of_displacements, MPI_Fint *  oldtype, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_Datatype local_type;
  *ierr = MPI_Type_create_hindexed( *count, array_of_blocklengths, array_of_displacements, MPI_Type_f2c(*oldtype), &local_type) ; 
  *newtype = MPI_Type_c2f(local_type);
  return ; 
}

/******************************************************
***      MPI_Type_create_hindexed wrapper function 
******************************************************/
void mpi_type_create_hindexed( MPI_Fint *  count, MPI_Fint *  array_of_blocklengths, MPI_Aint * array_of_displacements, MPI_Fint *  oldtype, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HINDEXED( count, array_of_blocklengths, array_of_displacements, oldtype, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_hindexed wrapper function 
******************************************************/
void mpi_type_create_hindexed_( MPI_Fint *  count, MPI_Fint *  array_of_blocklengths, MPI_Aint * array_of_displacements, MPI_Fint *  oldtype, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HINDEXED( count, array_of_blocklengths, array_of_displacements, oldtype, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_hindexed wrapper function 
******************************************************/
void mpi_type_create_hindexed__( MPI_Fint *  count, MPI_Fint *  array_of_blocklengths, MPI_Aint * array_of_displacements, MPI_Fint *  oldtype, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HINDEXED( count, array_of_blocklengths, array_of_displacements, oldtype, newtype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Type_create_hvector wrapper function 
******************************************************/
int MPI_Type_create_hvector( int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype * newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_hvector()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_hvector( count, blocklength, stride, oldtype, newtype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_create_hvector wrapper function 
******************************************************/
void MPI_TYPE_CREATE_HVECTOR( MPI_Fint *  count, MPI_Fint *  blocklength, MPI_Aint *  stride, MPI_Fint *  oldtype, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_Datatype local_type;
  *ierr = MPI_Type_create_hvector( *count, *blocklength, *stride, MPI_Type_f2c(*oldtype), &local_type) ; 
  *newtype = MPI_Type_c2f(local_type);
  return ; 
}

/******************************************************
***      MPI_Type_create_hvector wrapper function 
******************************************************/
void mpi_type_create_hvector( MPI_Fint *  count, MPI_Fint *  blocklength, MPI_Aint *  stride, MPI_Fint *  oldtype, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HVECTOR( count, blocklength, stride, oldtype, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_hvector wrapper function 
******************************************************/
void mpi_type_create_hvector_( MPI_Fint *  count, MPI_Fint *  blocklength, MPI_Aint *  stride, MPI_Fint *  oldtype, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HVECTOR( count, blocklength, stride, oldtype, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_hvector wrapper function 
******************************************************/
void mpi_type_create_hvector__( MPI_Fint *  count, MPI_Fint *  blocklength, MPI_Aint *  stride, MPI_Fint *  oldtype, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_HVECTOR( count, blocklength, stride, oldtype, newtype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Type_create_struct wrapper function 
******************************************************/
int MPI_Type_create_struct( int count, int * array_of_blocklengths, MPI_Aint * array_of_displacements, MPI_Datatype * array_of_types, MPI_Datatype * newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_struct()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_struct( count, array_of_blocklengths, array_of_displacements, array_of_types, newtype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_create_struct wrapper function 
******************************************************/
void MPI_TYPE_CREATE_STRUCT( MPI_Fint *  count, MPI_Fint *  array_of_blocklengths, MPI_Aint * array_of_displacements, MPI_Aint * array_of_types, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_Datatype local_type;
  TAU_DECL_ALLOC_LOCAL(MPI_Datatype, ary_local_types, *count);
  TAU_ASSIGN_VALUES(ary_local_types, array_of_types, *count, MPI_Type_f2c);
  *ierr = MPI_Type_create_struct( *count, array_of_blocklengths, array_of_displacements, ary_local_types, &local_type) ; 
  *newtype = MPI_Type_c2f(local_type);
  return ; 
}

/******************************************************
***      MPI_Type_create_struct wrapper function 
******************************************************/
void mpi_type_create_struct( MPI_Fint *  count, MPI_Fint *  array_of_blocklengths, MPI_Aint * array_of_displacements, MPI_Aint * array_of_types, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_STRUCT( count, array_of_blocklengths, array_of_displacements, array_of_types, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_struct wrapper function 
******************************************************/
void mpi_type_create_struct_( MPI_Fint *  count, MPI_Fint *  array_of_blocklengths, MPI_Aint * array_of_displacements, MPI_Aint * array_of_types, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_STRUCT( count, array_of_blocklengths, array_of_displacements, array_of_types, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_struct wrapper function 
******************************************************/
void mpi_type_create_struct__( MPI_Fint *  count, MPI_Fint *  array_of_blocklengths, MPI_Aint * array_of_displacements, MPI_Aint * array_of_types, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_STRUCT( count, array_of_blocklengths, array_of_displacements, array_of_types, newtype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Type_get_extent wrapper function 
******************************************************/
int MPI_Type_get_extent( MPI_Datatype datatype, MPI_Aint * lb, MPI_Aint * extent)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_get_extent()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_get_extent( datatype, lb, extent) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_get_extent wrapper function 
******************************************************/
void MPI_TYPE_GET_EXTENT( MPI_Fint *  datatype, MPI_Aint * lb, MPI_Aint * extent, MPI_Fint * ierr)
{
  *ierr = MPI_Type_get_extent( MPI_Type_f2c(*datatype), lb, extent) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_extent wrapper function 
******************************************************/
void mpi_type_get_extent( MPI_Fint *  datatype, MPI_Aint * lb, MPI_Aint * extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_EXTENT( datatype, lb, extent, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_extent wrapper function 
******************************************************/
void mpi_type_get_extent_( MPI_Fint *  datatype, MPI_Aint * lb, MPI_Aint * extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_EXTENT( datatype, lb, extent, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_extent wrapper function 
******************************************************/
void mpi_type_get_extent__( MPI_Fint *  datatype, MPI_Aint * lb, MPI_Aint * extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_EXTENT( datatype, lb, extent, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#ifdef TAU_MPITYPEEX
#ifndef TAU_BGL

#ifdef TAU_MPITYPEEX_F90
/******************************************************
***      MPI_Type_create_f90_real wrapper function 
******************************************************/
int MPI_Type_create_f90_real( int p, int r, MPI_Datatype * newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_f90_real()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_f90_real( p, r, newtype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_create_f90_real wrapper function 
******************************************************/
void MPI_TYPE_CREATE_F90_REAL( MPI_Fint *  p, MPI_Fint *  r, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_Datatype local_type;
  *ierr = MPI_Type_create_f90_real( *p, *r, &local_type) ; 
  *newtype = MPI_Type_c2f(local_type);
  return ; 
}

/******************************************************
***      MPI_Type_create_f90_real wrapper function 
******************************************************/
void mpi_type_create_f90_real( MPI_Fint *  p, MPI_Fint *  r, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_F90_REAL( p, r, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_f90_real wrapper function 
******************************************************/
void mpi_type_create_f90_real_( MPI_Fint *  p, MPI_Fint *  r, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_F90_REAL( p, r, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_f90_real wrapper function 
******************************************************/
void mpi_type_create_f90_real__( MPI_Fint *  p, MPI_Fint *  r, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_F90_REAL( p, r, newtype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Type_create_f90_complex wrapper function 
******************************************************/
int MPI_Type_create_f90_complex( int p, int r, MPI_Datatype * newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_f90_complex()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_f90_complex( p, r, newtype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_create_f90_complex wrapper function 
******************************************************/
void MPI_TYPE_CREATE_F90_COMPLEX( MPI_Fint *  p, MPI_Fint *  r, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_Datatype local_type;
  *ierr = MPI_Type_create_f90_complex( *p, *r, &local_type) ; 
  *newtype = MPI_Type_c2f(local_type);
  return ; 
}

/******************************************************
***      MPI_Type_create_f90_complex wrapper function 
******************************************************/
void mpi_type_create_f90_complex( MPI_Fint *  p, MPI_Fint *  r, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_F90_COMPLEX( p, r, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_f90_complex wrapper function 
******************************************************/
void mpi_type_create_f90_complex_( MPI_Fint *  p, MPI_Fint *  r, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_F90_COMPLEX( p, r, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_f90_complex wrapper function 
******************************************************/
void mpi_type_create_f90_complex__( MPI_Fint *  p, MPI_Fint *  r, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_F90_COMPLEX( p, r, newtype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Type_create_f90_integer wrapper function 
******************************************************/
int MPI_Type_create_f90_integer( int r, MPI_Datatype * newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_f90_integer()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_f90_integer( r, newtype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_create_f90_integer wrapper function 
******************************************************/
void MPI_TYPE_CREATE_F90_INTEGER( MPI_Fint *  r, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_Datatype local_type;
  *ierr = MPI_Type_create_f90_integer( *r, &local_type) ; 
  *newtype = MPI_Type_c2f(local_type);
  return ; 
}

/******************************************************
***      MPI_Type_create_f90_integer wrapper function 
******************************************************/
void mpi_type_create_f90_integer( MPI_Fint *  r, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_F90_INTEGER( r, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_f90_integer wrapper function 
******************************************************/
void mpi_type_create_f90_integer_( MPI_Fint *  r, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_F90_INTEGER( r, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_f90_integer wrapper function 
******************************************************/
void mpi_type_create_f90_integer__( MPI_Fint *  r, MPI_Fint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_F90_INTEGER( r, newtype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#endif /* TAU_MPITYPEEX_F90 */
#endif /* TAU_BGL */



/******************************************************
***      MPI_Type_match_size wrapper function 
******************************************************/
int MPI_Type_match_size( int typeclass, int size, MPI_Datatype * type)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_match_size()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_match_size( typeclass, size, type) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_match_size wrapper function 
******************************************************/
void MPI_TYPE_MATCH_SIZE( MPI_Fint *  typeclass, MPI_Fint *  size, MPI_Fint * type, MPI_Fint * ierr)
{
  MPI_Datatype local_type;
  *ierr = MPI_Type_match_size( *typeclass, *size, &local_type) ; 
  *type = MPI_Type_c2f(local_type);
  return ; 
}

/******************************************************
***      MPI_Type_match_size wrapper function 
******************************************************/
void mpi_type_match_size( MPI_Fint *  typeclass, MPI_Fint *  size, MPI_Fint * type, MPI_Fint * ierr)
{
  MPI_TYPE_MATCH_SIZE( typeclass, size, type, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_match_size wrapper function 
******************************************************/
void mpi_type_match_size_( MPI_Fint *  typeclass, MPI_Fint *  size, MPI_Fint * type, MPI_Fint * ierr)
{
  MPI_TYPE_MATCH_SIZE( typeclass, size, type, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_match_size wrapper function 
******************************************************/
void mpi_type_match_size__( MPI_Fint *  typeclass, MPI_Fint *  size, MPI_Fint * type, MPI_Fint * ierr)
{
  MPI_TYPE_MATCH_SIZE( typeclass, size, type, ierr) ; 
  return ; 
}

#endif /* TAU_MPI_TYPEEX */

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Alltoallw wrapper function 
******************************************************/
int MPI_Alltoallw( void * sendbuf, int * sendcounts, int * sdispls, MPI_Datatype * sendtypes, void * recvbuf, int * recvcounts, int * rdispls, MPI_Datatype * recvtypes, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Alltoallw()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Alltoallw( sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Alltoallw wrapper function 
******************************************************/
void MPI_ALLTOALLW( MPI_Aint * sendbuf, MPI_Fint *  sendcounts, MPI_Fint *  sdispls, MPI_Fint * sendtypes, MPI_Aint * recvbuf, MPI_Fint *  recvcounts, MPI_Fint *  rdispls, MPI_Fint * recvtypes, MPI_Fint *  comm, MPI_Fint * ierr)
{
  TAU_DECL_LOCAL(MPI_Datatype, local_send_types);
  TAU_DECL_ALLOC_LOCAL(MPI_Datatype, local_recv_types, *recvcounts);
  TAU_ALLOC_LOCAL(MPI_Datatype, local_send_types, *sendcounts);
  TAU_ASSIGN_VALUES(local_send_types, sendtypes, *sendcounts, MPI_Type_f2c);
  TAU_ASSIGN_VALUES(local_recv_types, recvtypes, *recvcounts, MPI_Type_f2c);
  *ierr = MPI_Alltoallw( sendbuf, sendcounts, sdispls, local_send_types, recvbuf, recvcounts, rdispls, local_recv_types, MPI_Comm_f2c(*comm)) ; 
  return ; 
}

/******************************************************
***      MPI_Alltoallw wrapper function 
******************************************************/
void mpi_alltoallw( MPI_Aint * sendbuf, MPI_Fint *  sendcounts, MPI_Fint *  sdispls, MPI_Fint * sendtypes, MPI_Aint * recvbuf, MPI_Fint *  recvcounts, MPI_Fint *  rdispls, MPI_Fint * recvtypes, MPI_Fint *  comm, MPI_Fint * ierr)
{
  MPI_ALLTOALLW( sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Alltoallw wrapper function 
******************************************************/
void mpi_alltoallw_( MPI_Aint * sendbuf, MPI_Fint *  sendcounts, MPI_Fint *  sdispls, MPI_Fint * sendtypes, MPI_Aint * recvbuf, MPI_Fint *  recvcounts, MPI_Fint *  rdispls, MPI_Fint * recvtypes, MPI_Fint *  comm, MPI_Fint * ierr)
{
  MPI_ALLTOALLW( sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Alltoallw wrapper function 
******************************************************/
void mpi_alltoallw__( MPI_Aint * sendbuf, MPI_Fint *  sendcounts, MPI_Fint *  sdispls, MPI_Fint * sendtypes, MPI_Aint * recvbuf, MPI_Fint *  recvcounts, MPI_Fint *  rdispls, MPI_Fint * recvtypes, MPI_Fint *  comm, MPI_Fint * ierr)
{
  MPI_ALLTOALLW( sendbuf, sendcounts, sdispls, sendtypes, recvbuf, recvcounts, rdispls, recvtypes, comm, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#ifdef TAU_MPITYPEEX

/******************************************************
***      MPI_Exscan wrapper function 
******************************************************/
int MPI_Exscan( void * sendbuf, void * recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Exscan()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Exscan( sendbuf, recvbuf, count, datatype, op, comm) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Exscan wrapper function 
******************************************************/
void MPI_EXSCAN( MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint *  op, MPI_Fint *  comm, MPI_Fint * ierr)
{
  *ierr = MPI_Exscan( sendbuf, recvbuf, *count, MPI_Type_f2c(*datatype), MPI_Op_f2c(*op), MPI_Comm_f2c(*comm)) ; 
  return ; 
}

/******************************************************
***      MPI_Exscan wrapper function 
******************************************************/
void mpi_exscan( MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint *  op, MPI_Fint *  comm, MPI_Fint * ierr)
{
  MPI_EXSCAN( sendbuf, recvbuf, count, datatype, op, comm, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Exscan wrapper function 
******************************************************/
void mpi_exscan_( MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint *  op, MPI_Fint *  comm, MPI_Fint * ierr)
{
  MPI_EXSCAN( sendbuf, recvbuf, count, datatype, op, comm, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Exscan wrapper function 
******************************************************/
void mpi_exscan__( MPI_Aint * sendbuf, MPI_Aint * recvbuf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint *  op, MPI_Fint *  comm, MPI_Fint * ierr)
{
  MPI_EXSCAN( sendbuf, recvbuf, count, datatype, op, comm, ierr) ; 
  return ; 
}

#endif /* TAU_MPI_TYPEEX */

/******************************************************/
/******************************************************/

#ifdef TAU_MPIERRHANDLER

/******************************************************
***      MPI_Comm_create_errhandler wrapper function 
******************************************************/
int MPI_Comm_create_errhandler( MPI_Comm_errhandler_fn * function, MPI_Errhandler * errhandler)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_create_errhandler()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_create_errhandler( function, errhandler) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Comm_create_errhandler wrapper function 
******************************************************/
void MPI_COMM_CREATE_ERRHANDLER( MPI_Comm_errhandler_fn * function, MPI_Errhandler * errhandler, MPI_Fint * ierr)
{
  MPI_Errhandler local_errhandler; 

  *ierr = MPI_Comm_create_errhandler( function, &local_errhandler) ; 
  *errhandler = MPI_Errhandler_c2f(local_errhandler);
  return ; 
}

/******************************************************
***      MPI_Comm_create_errhandler wrapper function 
******************************************************/
void mpi_comm_create_errhandler( MPI_Comm_errhandler_fn * function, MPI_Errhandler * errhandler, MPI_Fint * ierr)
{
  MPI_COMM_CREATE_ERRHANDLER( function, errhandler, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_create_errhandler wrapper function 
******************************************************/
void mpi_comm_create_errhandler_( MPI_Comm_errhandler_fn * function, MPI_Errhandler * errhandler, MPI_Fint * ierr)
{
  MPI_COMM_CREATE_ERRHANDLER( function, errhandler, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_create_errhandler wrapper function 
******************************************************/
void mpi_comm_create_errhandler__( MPI_Comm_errhandler_fn * function, MPI_Errhandler * errhandler, MPI_Fint * ierr)
{
  MPI_COMM_CREATE_ERRHANDLER( function, errhandler, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Comm_get_errhandler wrapper function 
******************************************************/
int MPI_Comm_get_errhandler( MPI_Comm comm, MPI_Errhandler * errhandler)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_get_errhandler()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_get_errhandler( comm, errhandler) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Comm_get_errhandler wrapper function 
******************************************************/
void MPI_COMM_GET_ERRHANDLER( MPI_Fint *  comm, MPI_Fint * errhandler, MPI_Fint * ierr)
{
  MPI_Comm local_comm; 
  MPI_Errhandler local_errhandler; 
  local_comm = MPI_Comm_f2c(*comm);
  *ierr = MPI_Comm_get_errhandler(local_comm, &local_errhandler) ; 
  *errhandler = MPI_Errhandler_c2f(local_errhandler);
  return ; 
}

/******************************************************
***      MPI_Comm_get_errhandler wrapper function 
******************************************************/
void mpi_comm_get_errhandler( MPI_Fint *  comm, MPI_Fint * errhandler, MPI_Fint * ierr)
{
  MPI_COMM_GET_ERRHANDLER( comm, errhandler, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_get_errhandler wrapper function 
******************************************************/
void mpi_comm_get_errhandler_( MPI_Fint *  comm, MPI_Fint * errhandler, MPI_Fint * ierr)
{
  MPI_COMM_GET_ERRHANDLER( comm, errhandler, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_get_errhandler wrapper function 
******************************************************/
void mpi_comm_get_errhandler__( MPI_Fint *  comm, MPI_Fint * errhandler, MPI_Fint * ierr)
{
  MPI_COMM_GET_ERRHANDLER( comm, errhandler, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Comm_set_errhandler wrapper function 
******************************************************/
int MPI_Comm_set_errhandler( MPI_Comm comm, MPI_Errhandler errhandler)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_set_errhandler()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_set_errhandler( comm, errhandler) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Comm_set_errhandler wrapper function 
******************************************************/
void MPI_COMM_SET_ERRHANDLER( MPI_Fint *  comm, MPI_Fint *  errhandler, MPI_Fint * ierr)
{
  *ierr = MPI_Comm_set_errhandler( MPI_Comm_f2c(*comm), MPI_Errhandler_f2c(*errhandler)) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_set_errhandler wrapper function 
******************************************************/
void mpi_comm_set_errhandler( MPI_Fint *  comm, MPI_Fint *  errhandler, MPI_Fint * ierr)
{
  MPI_COMM_SET_ERRHANDLER( comm, errhandler, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_set_errhandler wrapper function 
******************************************************/
void mpi_comm_set_errhandler_( MPI_Fint *  comm, MPI_Fint *  errhandler, MPI_Fint * ierr)
{
  MPI_COMM_SET_ERRHANDLER( comm, errhandler, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_set_errhandler wrapper function 
******************************************************/
void mpi_comm_set_errhandler__( MPI_Fint *  comm, MPI_Fint *  errhandler, MPI_Fint * ierr)
{
  MPI_COMM_SET_ERRHANDLER( comm, errhandler, ierr) ; 
  return ; 
}

#endif /* TAU_MPIERRHANDLER */

/******************************************************/
/******************************************************/

#ifdef TAU_MPIATTRFUNCTION

/******************************************************
***      MPI_Comm_create_keyval wrapper function 
******************************************************/
int MPI_Comm_create_keyval( MPI_Comm_copy_attr_function * comm_copy_attr_fn, MPI_Comm_delete_attr_function * comm_delete_attr_fn, int * comm_keyval, void * extra_state)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_create_keyval()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_create_keyval( comm_copy_attr_fn, comm_delete_attr_fn, comm_keyval, extra_state) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Comm_create_keyval wrapper function 
******************************************************/
void MPI_COMM_CREATE_KEYVAL( MPI_Comm_copy_attr_function * comm_copy_attr_fn, MPI_Comm_delete_attr_function * comm_delete_attr_fn, MPI_Fint *  comm_keyval, MPI_Aint * extra_state, MPI_Fint * ierr)
{
  *ierr = MPI_Comm_create_keyval( comm_copy_attr_fn, comm_delete_attr_fn, comm_keyval, extra_state) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_create_keyval wrapper function 
******************************************************/
void mpi_comm_create_keyval( MPI_Comm_copy_attr_function * comm_copy_attr_fn, MPI_Comm_delete_attr_function * comm_delete_attr_fn, MPI_Fint *  comm_keyval, MPI_Aint * extra_state, MPI_Fint * ierr)
{
  MPI_COMM_CREATE_KEYVAL( comm_copy_attr_fn, comm_delete_attr_fn, comm_keyval, extra_state, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_create_keyval wrapper function 
******************************************************/
void mpi_comm_create_keyval_( MPI_Comm_copy_attr_function * comm_copy_attr_fn, MPI_Comm_delete_attr_function * comm_delete_attr_fn, MPI_Fint *  comm_keyval, MPI_Aint * extra_state, MPI_Fint * ierr)
{
  MPI_COMM_CREATE_KEYVAL( comm_copy_attr_fn, comm_delete_attr_fn, comm_keyval, extra_state, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_create_keyval wrapper function 
******************************************************/
void mpi_comm_create_keyval__( MPI_Comm_copy_attr_function * comm_copy_attr_fn, MPI_Comm_delete_attr_function * comm_delete_attr_fn, MPI_Fint *  comm_keyval, MPI_Aint * extra_state, MPI_Fint * ierr)
{
  MPI_COMM_CREATE_KEYVAL( comm_copy_attr_fn, comm_delete_attr_fn, comm_keyval, extra_state, ierr) ; 
  return ; 
}

#endif /* TAU_MPIATTRFUNCTION  */

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Comm_delete_attr wrapper function 
******************************************************/
int MPI_Comm_delete_attr( MPI_Comm comm, int comm_keyval)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_delete_attr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_delete_attr( comm, comm_keyval) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Comm_delete_attr wrapper function 
******************************************************/
void MPI_COMM_DELETE_ATTR( MPI_Fint *  comm, MPI_Fint *  comm_keyval, MPI_Fint * ierr)
{
  *ierr = MPI_Comm_delete_attr( MPI_Comm_f2c(*comm), *comm_keyval) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_delete_attr wrapper function 
******************************************************/
void mpi_comm_delete_attr( MPI_Fint *  comm, MPI_Fint *  comm_keyval, MPI_Fint * ierr)
{
  MPI_COMM_DELETE_ATTR( comm, comm_keyval, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_delete_attr wrapper function 
******************************************************/
void mpi_comm_delete_attr_( MPI_Fint *  comm, MPI_Fint *  comm_keyval, MPI_Fint * ierr)
{
  MPI_COMM_DELETE_ATTR( comm, comm_keyval, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_delete_attr wrapper function 
******************************************************/
void mpi_comm_delete_attr__( MPI_Fint *  comm, MPI_Fint *  comm_keyval, MPI_Fint * ierr)
{
  MPI_COMM_DELETE_ATTR( comm, comm_keyval, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Comm_free_keyval wrapper function 
******************************************************/
int MPI_Comm_free_keyval( int * comm_keyval)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_free_keyval()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_free_keyval( comm_keyval) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Comm_free_keyval wrapper function 
******************************************************/
void MPI_COMM_FREE_KEYVAL( MPI_Fint *  comm_keyval, MPI_Fint * ierr)
{
  *ierr = MPI_Comm_free_keyval( comm_keyval) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_free_keyval wrapper function 
******************************************************/
void mpi_comm_free_keyval( MPI_Fint *  comm_keyval, MPI_Fint * ierr)
{
  MPI_COMM_FREE_KEYVAL( comm_keyval, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_free_keyval wrapper function 
******************************************************/
void mpi_comm_free_keyval_( MPI_Fint *  comm_keyval, MPI_Fint * ierr)
{
  MPI_COMM_FREE_KEYVAL( comm_keyval, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_free_keyval wrapper function 
******************************************************/
void mpi_comm_free_keyval__( MPI_Fint *  comm_keyval, MPI_Fint * ierr)
{
  MPI_COMM_FREE_KEYVAL( comm_keyval, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Comm_get_attr wrapper function 
******************************************************/
int MPI_Comm_get_attr( MPI_Comm comm, int comm_keyval, void * attribute_val, int * flag)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_get_attr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_get_attr( comm, comm_keyval, attribute_val, flag) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Comm_get_attr wrapper function 
******************************************************/
void MPI_COMM_GET_ATTR( MPI_Fint *  comm, MPI_Fint *  comm_keyval, MPI_Aint * attribute_val, MPI_Fint *  flag, MPI_Fint * ierr)
{
  *ierr = MPI_Comm_get_attr( MPI_Comm_f2c(*comm), *comm_keyval, attribute_val, flag) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_get_attr wrapper function 
******************************************************/
void mpi_comm_get_attr( MPI_Fint *  comm, MPI_Fint *  comm_keyval, MPI_Aint * attribute_val, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_COMM_GET_ATTR( comm, comm_keyval, attribute_val, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_get_attr wrapper function 
******************************************************/
void mpi_comm_get_attr_( MPI_Fint *  comm, MPI_Fint *  comm_keyval, MPI_Aint * attribute_val, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_COMM_GET_ATTR( comm, comm_keyval, attribute_val, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_get_attr wrapper function 
******************************************************/
void mpi_comm_get_attr__( MPI_Fint *  comm, MPI_Fint *  comm_keyval, MPI_Aint * attribute_val, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_COMM_GET_ATTR( comm, comm_keyval, attribute_val, flag, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Comm_set_attr wrapper function 
******************************************************/
int MPI_Comm_set_attr( MPI_Comm comm, int comm_keyval, void * attribute_val)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_set_attr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_set_attr( comm, comm_keyval, attribute_val) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Comm_set_attr wrapper function 
******************************************************/
void MPI_COMM_SET_ATTR( MPI_Fint *  comm, MPI_Fint *  comm_keyval, MPI_Aint * attribute_val, MPI_Fint * ierr)
{
  *ierr = MPI_Comm_set_attr( MPI_Comm_f2c(*comm), *comm_keyval, attribute_val) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_set_attr wrapper function 
******************************************************/
void mpi_comm_set_attr( MPI_Fint *  comm, MPI_Fint *  comm_keyval, MPI_Aint * attribute_val, MPI_Fint * ierr)
{
  MPI_COMM_SET_ATTR( comm, comm_keyval, attribute_val, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_set_attr wrapper function 
******************************************************/
void mpi_comm_set_attr_( MPI_Fint *  comm, MPI_Fint *  comm_keyval, MPI_Aint * attribute_val, MPI_Fint * ierr)
{
  MPI_COMM_SET_ATTR( comm, comm_keyval, attribute_val, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_set_attr wrapper function 
******************************************************/
void mpi_comm_set_attr__( MPI_Fint *  comm, MPI_Fint *  comm_keyval, MPI_Aint * attribute_val, MPI_Fint * ierr)
{
  MPI_COMM_SET_ATTR( comm, comm_keyval, attribute_val, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Get_version wrapper function 
******************************************************/
int MPI_Get_version( int * version, int * subversion)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Get_version()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Get_version( version, subversion) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Get_version wrapper function 
******************************************************/
void MPI_GET_VERSION( MPI_Fint *  version, MPI_Fint *  subversion, MPI_Fint * ierr)
{
  *ierr = MPI_Get_version( version, subversion) ; 
  return ; 
}

/******************************************************
***      MPI_Get_version wrapper function 
******************************************************/
void mpi_get_version( MPI_Fint *  version, MPI_Fint *  subversion, MPI_Fint * ierr)
{
  MPI_GET_VERSION( version, subversion, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Get_version wrapper function 
******************************************************/
void mpi_get_version_( MPI_Fint *  version, MPI_Fint *  subversion, MPI_Fint * ierr)
{
  MPI_GET_VERSION( version, subversion, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Get_version wrapper function 
******************************************************/
void mpi_get_version__( MPI_Fint *  version, MPI_Fint *  subversion, MPI_Fint * ierr)
{
  MPI_GET_VERSION( version, subversion, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_create wrapper function 
******************************************************/
int MPI_Win_create( void * base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win * win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_create()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_create( base, size, disp_unit, info, comm, win) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_create wrapper function 
******************************************************/
void MPI_WIN_CREATE( MPI_Aint * base, MPI_Aint *  size, MPI_Fint *  disp_unit, MPI_Fint *  info, MPI_Fint *  comm, MPI_Fint * win, MPI_Fint * ierr)
{
  MPI_Comm local_comm;
  MPI_Win local_win; 
  local_comm = MPI_Comm_f2c(*comm);
  *ierr = MPI_Win_create( base, *size, *disp_unit, MPI_Info_f2c(*info), local_comm, &local_win) ; 
  *win = MPI_Win_c2f(local_win);
  return ; 
}

/******************************************************
***      MPI_Win_create wrapper function 
******************************************************/
void mpi_win_create( MPI_Aint * base, MPI_Aint *  size, MPI_Fint *  disp_unit, MPI_Fint *  info, MPI_Fint *  comm, MPI_Fint * win, MPI_Fint * ierr)
{
  MPI_WIN_CREATE( base, size, disp_unit, info, comm, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_create wrapper function 
******************************************************/
void mpi_win_create_( MPI_Aint * base, MPI_Aint *  size, MPI_Fint *  disp_unit, MPI_Fint *  info, MPI_Fint *  comm, MPI_Fint * win, MPI_Fint * ierr)
{
  MPI_WIN_CREATE( base, size, disp_unit, info, comm, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_create wrapper function 
******************************************************/
void mpi_win_create__( MPI_Aint * base, MPI_Aint *  size, MPI_Fint *  disp_unit, MPI_Fint *  info, MPI_Fint *  comm, MPI_Fint * win, MPI_Fint * ierr)
{
  MPI_WIN_CREATE( base, size, disp_unit, info, comm, win, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_free wrapper function 
******************************************************/
int MPI_Win_free( MPI_Win * win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_free()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_free( win) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_free wrapper function 
******************************************************/
void MPI_WIN_FREE( MPI_Win * win, MPI_Fint * ierr)
{
  MPI_Win local_win = MPI_Win_f2c(*win);
  *ierr = MPI_Win_free( &local_win) ; 
  *win = MPI_Win_c2f(local_win);
  return ; 
}

/******************************************************
***      MPI_Win_free wrapper function 
******************************************************/
void mpi_win_free( MPI_Win * win, MPI_Fint * ierr)
{
  MPI_WIN_FREE( win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_free wrapper function 
******************************************************/
void mpi_win_free_( MPI_Win * win, MPI_Fint * ierr)
{
  MPI_WIN_FREE( win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_free wrapper function 
******************************************************/
void mpi_win_free__( MPI_Win * win, MPI_Fint * ierr)
{
  MPI_WIN_FREE( win, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_get_group wrapper function 
******************************************************/
int MPI_Win_get_group( MPI_Win win, MPI_Group * group)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_get_group()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_get_group( win, group) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_get_group wrapper function 
******************************************************/
void MPI_WIN_GET_GROUP( MPI_Fint *  win, MPI_Fint * group, MPI_Fint * ierr)
{
  MPI_Win local_win;
  MPI_Group local_group; 
  local_win = MPI_Win_f2c(*win);
  *ierr = MPI_Win_get_group( local_win, &local_group) ; 
  *group = MPI_Group_c2f(local_group); 
  return ; 
}

/******************************************************
***      MPI_Win_get_group wrapper function 
******************************************************/
void mpi_win_get_group( MPI_Fint *  win, MPI_Fint * group, MPI_Fint * ierr)
{
  MPI_WIN_GET_GROUP( win, group, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_get_group wrapper function 
******************************************************/
void mpi_win_get_group_( MPI_Fint *  win, MPI_Fint * group, MPI_Fint * ierr)
{
  MPI_WIN_GET_GROUP( win, group, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_get_group wrapper function 
******************************************************/
void mpi_win_get_group__( MPI_Fint *  win, MPI_Fint * group, MPI_Fint * ierr)
{
  MPI_WIN_GET_GROUP( win, group, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Put wrapper function 
******************************************************/
int MPI_Put( void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Put()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Put( origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Put wrapper function 
******************************************************/
void MPI_PUT( MPI_Aint * origin_addr, MPI_Fint *  origin_count, MPI_Fint *  origin_datatype, MPI_Fint *  target_rank, MPI_Aint *  target_disp, MPI_Fint *  target_count, MPI_Fint *  target_datatype, MPI_Fint *  win, MPI_Fint * ierr)
{
  *ierr = MPI_Put( origin_addr, *origin_count, MPI_Type_f2c(*origin_datatype), *target_rank, *target_disp, *target_count, MPI_Type_f2c(*target_datatype), MPI_Win_f2c(*win)) ; 
  return ; 
}

/******************************************************
***      MPI_Put wrapper function 
******************************************************/
void mpi_put( MPI_Aint * origin_addr, MPI_Fint *  origin_count, MPI_Fint *  origin_datatype, MPI_Fint *  target_rank, MPI_Aint *  target_disp, MPI_Fint *  target_count, MPI_Fint *  target_datatype, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_PUT( origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Put wrapper function 
******************************************************/
void mpi_put_( MPI_Aint * origin_addr, MPI_Fint *  origin_count, MPI_Fint *  origin_datatype, MPI_Fint *  target_rank, MPI_Aint *  target_disp, MPI_Fint *  target_count, MPI_Fint *  target_datatype, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_PUT( origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Put wrapper function 
******************************************************/
void mpi_put__( MPI_Aint * origin_addr, MPI_Fint *  origin_count, MPI_Fint *  origin_datatype, MPI_Fint *  target_rank, MPI_Aint *  target_disp, MPI_Fint *  target_count, MPI_Fint *  target_datatype, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_PUT( origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Get wrapper function 
******************************************************/
int MPI_Get( void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Get( origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Get wrapper function 
******************************************************/
void MPI_GET( MPI_Aint * origin_addr, MPI_Fint *  origin_count, MPI_Fint *  origin_datatype, MPI_Fint *  target_rank, MPI_Aint *  target_disp, MPI_Fint *  target_count, MPI_Fint *  target_datatype, MPI_Fint *  win, MPI_Fint * ierr)
{
  *ierr = MPI_Get( origin_addr, *origin_count, MPI_Type_f2c(*origin_datatype), *target_rank, *target_disp, *target_count, MPI_Type_f2c(*target_datatype), MPI_Win_f2c(*win)) ; 
  return ; 
}

/******************************************************
***      MPI_Get wrapper function 
******************************************************/
void mpi_get( MPI_Aint * origin_addr, MPI_Fint *  origin_count, MPI_Fint *  origin_datatype, MPI_Fint *  target_rank, MPI_Aint *  target_disp, MPI_Fint *  target_count, MPI_Fint *  target_datatype, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_GET( origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Get wrapper function 
******************************************************/
void mpi_get_( MPI_Aint * origin_addr, MPI_Fint *  origin_count, MPI_Fint *  origin_datatype, MPI_Fint *  target_rank, MPI_Aint *  target_disp, MPI_Fint *  target_count, MPI_Fint *  target_datatype, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_GET( origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Get wrapper function 
******************************************************/
void mpi_get__( MPI_Aint * origin_addr, MPI_Fint *  origin_count, MPI_Fint *  origin_datatype, MPI_Fint *  target_rank, MPI_Aint *  target_disp, MPI_Fint *  target_count, MPI_Fint *  target_datatype, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_GET( origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, win, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Accumulate wrapper function 
******************************************************/
int MPI_Accumulate( void * origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Op op, MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Accumulate()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Accumulate( origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Accumulate wrapper function 
******************************************************/
void MPI_ACCUMULATE( MPI_Aint * origin_addr, MPI_Fint *  origin_count, MPI_Fint *  origin_datatype, MPI_Fint *  target_rank, MPI_Aint *  target_disp, MPI_Fint *  target_count, MPI_Fint *  target_datatype, MPI_Fint *  op, MPI_Fint *  win, MPI_Fint * ierr)
{
  *ierr = MPI_Accumulate( origin_addr, *origin_count, MPI_Type_f2c(*origin_datatype), *target_rank, *target_disp, *target_count, MPI_Type_f2c(*target_datatype), MPI_Op_f2c(*op), MPI_Win_f2c(*win)) ; 
  return ; 
}

/******************************************************
***      MPI_Accumulate wrapper function 
******************************************************/
void mpi_accumulate( MPI_Aint * origin_addr, MPI_Fint *  origin_count, MPI_Fint *  origin_datatype, MPI_Fint *  target_rank, MPI_Aint *  target_disp, MPI_Fint *  target_count, MPI_Fint *  target_datatype, MPI_Fint *  op, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_ACCUMULATE( origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Accumulate wrapper function 
******************************************************/
void mpi_accumulate_( MPI_Aint * origin_addr, MPI_Fint *  origin_count, MPI_Fint *  origin_datatype, MPI_Fint *  target_rank, MPI_Aint *  target_disp, MPI_Fint *  target_count, MPI_Fint *  target_datatype, MPI_Fint *  op, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_ACCUMULATE( origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Accumulate wrapper function 
******************************************************/
void mpi_accumulate__( MPI_Aint * origin_addr, MPI_Fint *  origin_count, MPI_Fint *  origin_datatype, MPI_Fint *  target_rank, MPI_Aint *  target_disp, MPI_Fint *  target_count, MPI_Fint *  target_datatype, MPI_Fint *  op, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_ACCUMULATE( origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count, target_datatype, op, win, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_fence wrapper function 
******************************************************/
int MPI_Win_fence( int assert, MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_fence()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_fence( assert, win) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_fence wrapper function 
******************************************************/
void MPI_WIN_FENCE( MPI_Fint *  assert, MPI_Fint *  win, MPI_Fint * ierr)
{
  *ierr = MPI_Win_fence( *assert, MPI_Win_f2c(*win)) ; 
  return ; 
}

/******************************************************
***      MPI_Win_fence wrapper function 
******************************************************/
void mpi_win_fence( MPI_Fint *  assert, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_FENCE( assert, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_fence wrapper function 
******************************************************/
void mpi_win_fence_( MPI_Fint *  assert, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_FENCE( assert, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_fence wrapper function 
******************************************************/
void mpi_win_fence__( MPI_Fint *  assert, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_FENCE( assert, win, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#ifndef TAU_BGL
/******************************************************
***      MPI_Win_start wrapper function 
******************************************************/
int MPI_Win_start( MPI_Group group, int assert, MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_start()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_start( group, assert, win) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_start wrapper function 
******************************************************/
void MPI_WIN_START( MPI_Fint *  group, MPI_Fint *  assert, MPI_Fint *  win, MPI_Fint * ierr)
{
  *ierr = MPI_Win_start( MPI_Group_f2c(*group), *assert, MPI_Win_f2c(*win)) ; 
  return ; 
}

/******************************************************
***      MPI_Win_start wrapper function 
******************************************************/
void mpi_win_start( MPI_Fint *  group, MPI_Fint *  assert, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_START( group, assert, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_start wrapper function 
******************************************************/
void mpi_win_start_( MPI_Fint *  group, MPI_Fint *  assert, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_START( group, assert, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_start wrapper function 
******************************************************/
void mpi_win_start__( MPI_Fint *  group, MPI_Fint *  assert, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_START( group, assert, win, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#endif /* TAU_BGL */

/******************************************************
***      MPI_Win_complete wrapper function 
******************************************************/
int MPI_Win_complete( MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_complete()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_complete( win) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_complete wrapper function 
******************************************************/
void MPI_WIN_COMPLETE( MPI_Fint *  win, MPI_Fint * ierr)
{
  *ierr = MPI_Win_complete( MPI_Win_f2c(*win)) ; 
  return ; 
}

/******************************************************
***      MPI_Win_complete wrapper function 
******************************************************/
void mpi_win_complete( MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_COMPLETE( win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_complete wrapper function 
******************************************************/
void mpi_win_complete_( MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_COMPLETE( win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_complete wrapper function 
******************************************************/
void mpi_win_complete__( MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_COMPLETE( win, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_post wrapper function 
******************************************************/
int MPI_Win_post( MPI_Group group, int assert, MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_post()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_post( group, assert, win) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_post wrapper function 
******************************************************/
void MPI_WIN_POST( MPI_Fint *  group, MPI_Fint *  assert, MPI_Fint *  win, MPI_Fint * ierr)
{
  *ierr = MPI_Win_post( MPI_Group_f2c(*group), *assert, MPI_Win_f2c(*win)) ; 
  return ; 
}

/******************************************************
***      MPI_Win_post wrapper function 
******************************************************/
void mpi_win_post( MPI_Fint *  group, MPI_Fint *  assert, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_POST( group, assert, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_post wrapper function 
******************************************************/
void mpi_win_post_( MPI_Fint *  group, MPI_Fint *  assert, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_POST( group, assert, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_post wrapper function 
******************************************************/
void mpi_win_post__( MPI_Fint *  group, MPI_Fint *  assert, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_POST( group, assert, win, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_wait wrapper function 
******************************************************/
int MPI_Win_wait( MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_wait()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_wait( win) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_wait wrapper function 
******************************************************/
void MPI_WIN_WAIT( MPI_Fint *  win, MPI_Fint * ierr)
{
  *ierr = MPI_Win_wait( MPI_Win_f2c(*win)) ; 
  return ; 
}

/******************************************************
***      MPI_Win_wait wrapper function 
******************************************************/
void mpi_win_wait( MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_WAIT( win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_wait wrapper function 
******************************************************/
void mpi_win_wait_( MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_WAIT( win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_wait wrapper function 
******************************************************/
void mpi_win_wait__( MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_WAIT( win, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_test wrapper function 
******************************************************/
int MPI_Win_test( MPI_Win win, int * flag)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_test()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_test( win, flag) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_test wrapper function 
******************************************************/
void MPI_WIN_TEST( MPI_Fint *  win, MPI_Fint *  flag, MPI_Fint * ierr)
{
  *ierr = MPI_Win_test( MPI_Win_f2c(*win), flag) ; 
  return ; 
}

/******************************************************
***      MPI_Win_test wrapper function 
******************************************************/
void mpi_win_test( MPI_Fint *  win, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_WIN_TEST( win, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_test wrapper function 
******************************************************/
void mpi_win_test_( MPI_Fint *  win, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_WIN_TEST( win, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_test wrapper function 
******************************************************/
void mpi_win_test__( MPI_Fint *  win, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_WIN_TEST( win, flag, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_lock wrapper function 
******************************************************/
int MPI_Win_lock( int lock_type, int rank, int assert, MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_lock()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_lock( lock_type, rank, assert, win) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_lock wrapper function 
******************************************************/
void MPI_WIN_LOCK( MPI_Fint *  lock_type, MPI_Fint *  rank, MPI_Fint *  assert, MPI_Fint *  win, MPI_Fint * ierr)
{
  *ierr = MPI_Win_lock( *lock_type, *rank, *assert, MPI_Win_f2c(*win)) ; 
  return ; 
}

/******************************************************
***      MPI_Win_lock wrapper function 
******************************************************/
void mpi_win_lock( MPI_Fint *  lock_type, MPI_Fint *  rank, MPI_Fint *  assert, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_LOCK( lock_type, rank, assert, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_lock wrapper function 
******************************************************/
void mpi_win_lock_( MPI_Fint *  lock_type, MPI_Fint *  rank, MPI_Fint *  assert, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_LOCK( lock_type, rank, assert, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_lock wrapper function 
******************************************************/
void mpi_win_lock__( MPI_Fint *  lock_type, MPI_Fint *  rank, MPI_Fint *  assert, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_LOCK( lock_type, rank, assert, win, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_unlock wrapper function 
******************************************************/
int MPI_Win_unlock( int rank, MPI_Win win)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_unlock()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_unlock( rank, win) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_unlock wrapper function 
******************************************************/
void MPI_WIN_UNLOCK( MPI_Fint *  rank, MPI_Fint *  win, MPI_Fint * ierr)
{
  *ierr = MPI_Win_unlock( *rank, MPI_Win_f2c(*win)) ; 
  return ; 
}

/******************************************************
***      MPI_Win_unlock wrapper function 
******************************************************/
void mpi_win_unlock( MPI_Fint *  rank, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_UNLOCK( rank, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_unlock wrapper function 
******************************************************/
void mpi_win_unlock_( MPI_Fint *  rank, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_UNLOCK( rank, win, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_unlock wrapper function 
******************************************************/
void mpi_win_unlock__( MPI_Fint *  rank, MPI_Fint *  win, MPI_Fint * ierr)
{
  MPI_WIN_UNLOCK( rank, win, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_get_attr wrapper function 
******************************************************/
int MPI_Win_get_attr( MPI_Win win, int win_keyval, void * attribute_val, int * flag)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_get_attr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_get_attr( win, win_keyval, attribute_val, flag) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_get_attr wrapper function 
******************************************************/
void MPI_WIN_GET_ATTR( MPI_Fint *  win, MPI_Fint *  win_keyval, MPI_Aint * attribute_val, MPI_Fint *  flag, MPI_Fint * ierr)
{
  *ierr = MPI_Win_get_attr( MPI_Win_f2c(*win), *win_keyval, attribute_val, flag) ; 
  return ; 
}

/******************************************************
***      MPI_Win_get_attr wrapper function 
******************************************************/
void mpi_win_get_attr( MPI_Fint *  win, MPI_Fint *  win_keyval, MPI_Aint * attribute_val, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_WIN_GET_ATTR( win, win_keyval, attribute_val, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_get_attr wrapper function 
******************************************************/
void mpi_win_get_attr_( MPI_Fint *  win, MPI_Fint *  win_keyval, MPI_Aint * attribute_val, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_WIN_GET_ATTR( win, win_keyval, attribute_val, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_get_attr wrapper function 
******************************************************/
void mpi_win_get_attr__( MPI_Fint *  win, MPI_Fint *  win_keyval, MPI_Aint * attribute_val, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_WIN_GET_ATTR( win, win_keyval, attribute_val, flag, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_set_attr wrapper function 
******************************************************/
int MPI_Win_set_attr( MPI_Win win, int win_keyval, void * attribute_val)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_set_attr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_set_attr( win, win_keyval, attribute_val) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_set_attr wrapper function 
******************************************************/
void MPI_WIN_SET_ATTR( MPI_Fint *  win, MPI_Fint *  win_keyval, MPI_Aint * attribute_val, MPI_Fint * ierr)
{
  *ierr = MPI_Win_set_attr( MPI_Win_f2c(*win), *win_keyval, attribute_val) ; 
  return ; 
}

/******************************************************
***      MPI_Win_set_attr wrapper function 
******************************************************/
void mpi_win_set_attr( MPI_Fint *  win, MPI_Fint *  win_keyval, MPI_Aint * attribute_val, MPI_Fint * ierr)
{
  MPI_WIN_SET_ATTR( win, win_keyval, attribute_val, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_set_attr wrapper function 
******************************************************/
void mpi_win_set_attr_( MPI_Fint *  win, MPI_Fint *  win_keyval, MPI_Aint * attribute_val, MPI_Fint * ierr)
{
  MPI_WIN_SET_ATTR( win, win_keyval, attribute_val, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_set_attr wrapper function 
******************************************************/
void mpi_win_set_attr__( MPI_Fint *  win, MPI_Fint *  win_keyval, MPI_Aint * attribute_val, MPI_Fint * ierr)
{
  MPI_WIN_SET_ATTR( win, win_keyval, attribute_val, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


#ifdef TAU_MPIATTRFUNCTION

/******************************************************
***      MPI_Win_create_keyval wrapper function 
******************************************************/
int MPI_Win_create_keyval( MPI_Win_copy_attr_function * win_copy_attr_fn, MPI_Win_delete_attr_function * win_delete_attr_fn, int * win_keyval, void * extra_state)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_create_keyval()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_create_keyval( win_copy_attr_fn, win_delete_attr_fn, win_keyval, extra_state) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_create_keyval wrapper function 
******************************************************/
void MPI_WIN_CREATE_KEYVAL( MPI_Win_copy_attr_function * win_copy_attr_fn, MPI_Win_delete_attr_function * win_delete_attr_fn, MPI_Fint *  win_keyval, MPI_Aint * extra_state, MPI_Fint * ierr)
{
  *ierr = MPI_Win_create_keyval( win_copy_attr_fn, win_delete_attr_fn, win_keyval, extra_state) ; 
  return ; 
}

/******************************************************
***      MPI_Win_create_keyval wrapper function 
******************************************************/
void mpi_win_create_keyval( MPI_Win_copy_attr_function * win_copy_attr_fn, MPI_Win_delete_attr_function * win_delete_attr_fn, MPI_Fint *  win_keyval, MPI_Aint * extra_state, MPI_Fint * ierr)
{
  MPI_WIN_CREATE_KEYVAL( win_copy_attr_fn, win_delete_attr_fn, win_keyval, extra_state, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_create_keyval wrapper function 
******************************************************/
void mpi_win_create_keyval_( MPI_Win_copy_attr_function * win_copy_attr_fn, MPI_Win_delete_attr_function * win_delete_attr_fn, MPI_Fint *  win_keyval, MPI_Aint * extra_state, MPI_Fint * ierr)
{
  MPI_WIN_CREATE_KEYVAL( win_copy_attr_fn, win_delete_attr_fn, win_keyval, extra_state, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_create_keyval wrapper function 
******************************************************/
void mpi_win_create_keyval__( MPI_Win_copy_attr_function * win_copy_attr_fn, MPI_Win_delete_attr_function * win_delete_attr_fn, MPI_Fint *  win_keyval, MPI_Aint * extra_state, MPI_Fint * ierr)
{
  MPI_WIN_CREATE_KEYVAL( win_copy_attr_fn, win_delete_attr_fn, win_keyval, extra_state, ierr) ; 
  return ; 
}

#endif /* TAU_MPIATTRFUNCTION  */

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_delete_attr wrapper function 
******************************************************/
int MPI_Win_delete_attr( MPI_Win win, int win_keyval)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_delete_attr()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_delete_attr( win, win_keyval) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_delete_attr wrapper function 
******************************************************/
void MPI_WIN_DELETE_ATTR( MPI_Fint *  win, MPI_Fint *  win_keyval, MPI_Fint * ierr)
{
  MPI_Win local_win = MPI_Win_f2c(*win);
  *ierr = MPI_Win_delete_attr( local_win, *win_keyval) ; 
  *win = MPI_Win_c2f(local_win);
  return ; 
}

/******************************************************
***      MPI_Win_delete_attr wrapper function 
******************************************************/
void mpi_win_delete_attr( MPI_Fint *  win, MPI_Fint *  win_keyval, MPI_Fint * ierr)
{
  MPI_WIN_DELETE_ATTR( win, win_keyval, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_delete_attr wrapper function 
******************************************************/
void mpi_win_delete_attr_( MPI_Fint *  win, MPI_Fint *  win_keyval, MPI_Fint * ierr)
{
  MPI_WIN_DELETE_ATTR( win, win_keyval, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_delete_attr wrapper function 
******************************************************/
void mpi_win_delete_attr__( MPI_Fint *  win, MPI_Fint *  win_keyval, MPI_Fint * ierr)
{
  MPI_WIN_DELETE_ATTR( win, win_keyval, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_free_keyval wrapper function 
******************************************************/
int MPI_Win_free_keyval( int * win_keyval)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_free_keyval()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_free_keyval( win_keyval) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_free_keyval wrapper function 
******************************************************/
void MPI_WIN_FREE_KEYVAL( MPI_Fint *  win_keyval, MPI_Fint * ierr)
{
  *ierr = MPI_Win_free_keyval( win_keyval) ; 
  return ; 
}

/******************************************************
***      MPI_Win_free_keyval wrapper function 
******************************************************/
void mpi_win_free_keyval( MPI_Fint *  win_keyval, MPI_Fint * ierr)
{
  MPI_WIN_FREE_KEYVAL( win_keyval, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_free_keyval wrapper function 
******************************************************/
void mpi_win_free_keyval_( MPI_Fint *  win_keyval, MPI_Fint * ierr)
{
  MPI_WIN_FREE_KEYVAL( win_keyval, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_free_keyval wrapper function 
******************************************************/
void mpi_win_free_keyval__( MPI_Fint *  win_keyval, MPI_Fint * ierr)
{
  MPI_WIN_FREE_KEYVAL( win_keyval, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#ifdef TAU_MPIERRHANDLER

/******************************************************
***      MPI_Win_create_errhandler wrapper function 
******************************************************/
int MPI_Win_create_errhandler( MPI_Win_errhandler_fn * function, MPI_Errhandler * errhandler)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_create_errhandler()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_create_errhandler( function, errhandler) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_create_errhandler wrapper function 
******************************************************/
void MPI_WIN_CREATE_ERRHANDLER( MPI_Win_errhandler_fn * function, MPI_Errhandler * errhandler, MPI_Fint * ierr)
{
  MPI_Errhandler local_errhandler;
  *ierr = MPI_Win_create_errhandler( function, &local_errhandler) ; 
  *errhandler = MPI_Errhandler_c2f(local_errhandler);
  return ; 
}

/******************************************************
***      MPI_Win_create_errhandler wrapper function 
******************************************************/
void mpi_win_create_errhandler( MPI_Win_errhandler_fn * function, MPI_Errhandler * errhandler, MPI_Fint * ierr)
{
  MPI_WIN_CREATE_ERRHANDLER( function, errhandler, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_create_errhandler wrapper function 
******************************************************/
void mpi_win_create_errhandler_( MPI_Win_errhandler_fn * function, MPI_Errhandler * errhandler, MPI_Fint * ierr)
{
  MPI_WIN_CREATE_ERRHANDLER( function, errhandler, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_create_errhandler wrapper function 
******************************************************/
void mpi_win_create_errhandler__( MPI_Win_errhandler_fn * function, MPI_Errhandler * errhandler, MPI_Fint * ierr)
{
  MPI_WIN_CREATE_ERRHANDLER( function, errhandler, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_set_errhandler wrapper function 
******************************************************/
int MPI_Win_set_errhandler( MPI_Win win, MPI_Errhandler err)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_set_errhandler()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_set_errhandler( win, err) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_set_errhandler wrapper function 
******************************************************/
void MPI_WIN_SET_ERRHANDLER( MPI_Fint *  win, MPI_Fint *  err, MPI_Fint * ierr)
{
  *ierr = MPI_Win_set_errhandler( MPI_Win_f2c(*win), MPI_Errhandler_f2c(*err)) ; 
  return ; 
}

/******************************************************
***      MPI_Win_set_errhandler wrapper function 
******************************************************/
void mpi_win_set_errhandler( MPI_Fint *  win, MPI_Fint *  err, MPI_Fint * ierr)
{
  MPI_WIN_SET_ERRHANDLER( win, err, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_set_errhandler wrapper function 
******************************************************/
void mpi_win_set_errhandler_( MPI_Fint *  win, MPI_Fint *  err, MPI_Fint * ierr)
{
  MPI_WIN_SET_ERRHANDLER( win, err, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_set_errhandler wrapper function 
******************************************************/
void mpi_win_set_errhandler__( MPI_Fint *  win, MPI_Fint *  err, MPI_Fint * ierr)
{
  MPI_WIN_SET_ERRHANDLER( win, err, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_get_errhandler wrapper function 
******************************************************/
int MPI_Win_get_errhandler( MPI_Win win, MPI_Errhandler * errhandler)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_get_errhandler()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_get_errhandler( win, errhandler) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_get_errhandler wrapper function 
******************************************************/
void MPI_WIN_GET_ERRHANDLER( MPI_Fint *  win, MPI_Errhandler * errhandler, MPI_Fint * ierr)
{
  MPI_Win local_win;
  MPI_Errhandler local_errhandler; 
  local_win = MPI_Win_f2c(*win);
  *ierr = MPI_Win_get_errhandler( local_win, &local_errhandler) ; 
  *errhandler = MPI_Errhandler_c2f(local_errhandler);
  return ; 
}

/******************************************************
***      MPI_Win_get_errhandler wrapper function 
******************************************************/
void mpi_win_get_errhandler( MPI_Fint *  win, MPI_Errhandler * errhandler, MPI_Fint * ierr)
{
  MPI_WIN_GET_ERRHANDLER( win, errhandler, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_get_errhandler wrapper function 
******************************************************/
void mpi_win_get_errhandler_( MPI_Fint *  win, MPI_Errhandler * errhandler, MPI_Fint * ierr)
{
  MPI_WIN_GET_ERRHANDLER( win, errhandler, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_get_errhandler wrapper function 
******************************************************/
void mpi_win_get_errhandler__( MPI_Fint *  win, MPI_Errhandler * errhandler, MPI_Fint * ierr)
{
  MPI_WIN_GET_ERRHANDLER( win, errhandler, ierr) ; 
  return ; 
}

#endif /* TAU_MPIERRHANDLER */

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Alloc_mem wrapper function 
******************************************************/
int MPI_Alloc_mem( MPI_Aint size, MPI_Info info, void * baseptr)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Alloc_mem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Alloc_mem( size, info, baseptr) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Alloc_mem wrapper function 
******************************************************/
void MPI_ALLOC_MEM( MPI_Fint *  size, MPI_Fint *  info, MPI_Aint * baseptr, MPI_Fint * ierr)
{
  *ierr = MPI_Alloc_mem( *size, MPI_Info_f2c(*info), baseptr) ; 
  return ; 
}

/******************************************************
***      MPI_Alloc_mem wrapper function 
******************************************************/
void mpi_alloc_mem( MPI_Fint *  size, MPI_Fint *  info, MPI_Aint * baseptr, MPI_Fint * ierr)
{
  MPI_ALLOC_MEM( size, info, baseptr, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Alloc_mem wrapper function 
******************************************************/
void mpi_alloc_mem_( MPI_Fint *  size, MPI_Fint *  info, MPI_Aint * baseptr, MPI_Fint * ierr)
{
  MPI_ALLOC_MEM( size, info, baseptr, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Alloc_mem wrapper function 
******************************************************/
void mpi_alloc_mem__( MPI_Fint *  size, MPI_Fint *  info, MPI_Aint * baseptr, MPI_Fint * ierr)
{
  MPI_ALLOC_MEM( size, info, baseptr, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Free_mem wrapper function 
******************************************************/
int MPI_Free_mem( void * base)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Free_mem()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Free_mem( base) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Free_mem wrapper function 
******************************************************/
void MPI_FREE_MEM( MPI_Aint * base, MPI_Fint * ierr)
{
  *ierr = MPI_Free_mem( base) ; 
  return ; 
}

/******************************************************
***      MPI_Free_mem wrapper function 
******************************************************/
void mpi_free_mem( MPI_Aint * base, MPI_Fint * ierr)
{
  MPI_FREE_MEM( base, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Free_mem wrapper function 
******************************************************/
void mpi_free_mem_( MPI_Aint * base, MPI_Fint * ierr)
{
  MPI_FREE_MEM( base, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Free_mem wrapper function 
******************************************************/
void mpi_free_mem__( MPI_Aint * base, MPI_Fint * ierr)
{
  MPI_FREE_MEM( base, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_open wrapper function 
******************************************************/
int MPI_File_open( MPI_Comm comm, char * filename, int amode, MPI_Info info, MPI_File * fh)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_open()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_open( comm, filename, amode, info, fh) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_open wrapper function 
******************************************************/
void MPI_FILE_OPEN( MPI_Fint *  comm, char * filename, MPI_Fint *  amode, MPI_Fint *  info, MPI_Aint * fh, MPI_Fint * ierr)
{
  MPI_Comm local_comm;
  MPI_Info local_info; 
  MPI_File local_fh; 
  local_comm = MPI_Comm_f2c(*comm);
  local_info = MPI_Info_f2c(*info);
  
  *ierr = MPI_File_open( local_comm, filename, *amode, local_info, &local_fh) ; 
  *fh = MPI_File_c2f(local_fh);
  return ; 
}

/******************************************************
***      MPI_File_open wrapper function 
******************************************************/
void mpi_file_open( MPI_Fint *  comm, char * filename, MPI_Fint *  amode, MPI_Fint *  info, MPI_Aint * fh, MPI_Fint * ierr)
{
  MPI_FILE_OPEN( comm, filename, amode, info, fh, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_open wrapper function 
******************************************************/
void mpi_file_open_( MPI_Fint *  comm, char * filename, MPI_Fint *  amode, MPI_Fint *  info, MPI_Aint * fh, MPI_Fint * ierr)
{
  MPI_FILE_OPEN( comm, filename, amode, info, fh, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_open wrapper function 
******************************************************/
void mpi_file_open__( MPI_Fint *  comm, char * filename, MPI_Fint *  amode, MPI_Fint *  info, MPI_Aint * fh, MPI_Fint * ierr)
{
  MPI_FILE_OPEN( comm, filename, amode, info, fh, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_close wrapper function 
******************************************************/
int MPI_File_close( MPI_File * fh)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_close()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_close( fh) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_close wrapper function 
******************************************************/
void MPI_FILE_CLOSE( MPI_Aint * fh, MPI_Fint * ierr)
{
  MPI_File local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_close( &local_fh) ; 
  *fh = MPI_File_c2f(local_fh);
  return ; 
}

/******************************************************
***      MPI_File_close wrapper function 
******************************************************/
void mpi_file_close( MPI_Aint * fh, MPI_Fint * ierr)
{
  MPI_FILE_CLOSE( fh, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_close wrapper function 
******************************************************/
void mpi_file_close_( MPI_Aint * fh, MPI_Fint * ierr)
{
  MPI_FILE_CLOSE( fh, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_close wrapper function 
******************************************************/
void mpi_file_close__( MPI_Aint * fh, MPI_Fint * ierr)
{
  MPI_FILE_CLOSE( fh, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_delete wrapper function 
******************************************************/
int MPI_File_delete( char * filename, MPI_Info info)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_delete()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_delete( filename, info) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_delete wrapper function 
******************************************************/
void MPI_FILE_DELETE( char * filename, MPI_Fint *  info, MPI_Fint * ierr)
{
  MPI_Info local_info = MPI_Info_f2c(*info);
  *ierr = MPI_File_delete( filename, local_info) ; 
  return ; 
}

/******************************************************
***      MPI_File_delete wrapper function 
******************************************************/
void mpi_file_delete( char * filename, MPI_Fint *  info, MPI_Fint * ierr)
{
  MPI_FILE_DELETE( filename, info, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_delete wrapper function 
******************************************************/
void mpi_file_delete_( char * filename, MPI_Fint *  info, MPI_Fint * ierr)
{
  MPI_FILE_DELETE( filename, info, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_delete wrapper function 
******************************************************/
void mpi_file_delete__( char * filename, MPI_Fint *  info, MPI_Fint * ierr)
{
  MPI_FILE_DELETE( filename, info, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_set_size wrapper function 
******************************************************/
int MPI_File_set_size( MPI_File fh, MPI_Offset size)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_set_size()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_set_size( fh, size) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_set_size wrapper function 
******************************************************/
void MPI_FILE_SET_SIZE( MPI_Fint *  fh, MPI_Offset *  size, MPI_Fint * ierr)
{
  *ierr = MPI_File_set_size( MPI_File_f2c(*fh), *size) ; 
  return ; 
}

/******************************************************
***      MPI_File_set_size wrapper function 
******************************************************/
void mpi_file_set_size( MPI_Fint *  fh, MPI_Offset *  size, MPI_Fint * ierr)
{
  MPI_FILE_SET_SIZE( fh, size, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_set_size wrapper function 
******************************************************/
void mpi_file_set_size_( MPI_Fint *  fh, MPI_Offset *  size, MPI_Fint * ierr)
{
  MPI_FILE_SET_SIZE( fh, size, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_set_size wrapper function 
******************************************************/
void mpi_file_set_size__( MPI_Fint *  fh, MPI_Offset *  size, MPI_Fint * ierr)
{
  MPI_FILE_SET_SIZE( fh, size, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_get_size wrapper function 
******************************************************/
int MPI_File_get_size( MPI_File fh, MPI_Offset * size)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_get_size()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_get_size( fh, size) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_get_size wrapper function 
******************************************************/
void MPI_FILE_GET_SIZE( MPI_Fint *  fh, MPI_Offset * size, MPI_Fint * ierr)
{
  *ierr = MPI_File_get_size( MPI_File_f2c(*fh), size) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_size wrapper function 
******************************************************/
void mpi_file_get_size( MPI_Fint *  fh, MPI_Offset * size, MPI_Fint * ierr)
{
  MPI_FILE_GET_SIZE( fh, size, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_size wrapper function 
******************************************************/
void mpi_file_get_size_( MPI_Fint *  fh, MPI_Offset * size, MPI_Fint * ierr)
{
  MPI_FILE_GET_SIZE( fh, size, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_size wrapper function 
******************************************************/
void mpi_file_get_size__( MPI_Fint *  fh, MPI_Offset * size, MPI_Fint * ierr)
{
  MPI_FILE_GET_SIZE( fh, size, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_get_group wrapper function 
******************************************************/
int MPI_File_get_group( MPI_File fh, MPI_Group * group)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_get_group()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_get_group( fh, group) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_get_group wrapper function 
******************************************************/
void MPI_FILE_GET_GROUP( MPI_Fint *  fh, MPI_Aint * group, MPI_Fint * ierr)
{
  MPI_File local_file;
  MPI_Group local_group;
  local_file = MPI_File_f2c(*fh);
  *ierr = MPI_File_get_group( local_file, &local_group) ; 
  *group = MPI_Group_c2f(local_group);
  return ; 
}

/******************************************************
***      MPI_File_get_group wrapper function 
******************************************************/
void mpi_file_get_group( MPI_Fint *  fh, MPI_Aint * group, MPI_Fint * ierr)
{
  MPI_FILE_GET_GROUP( fh, group, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_group wrapper function 
******************************************************/
void mpi_file_get_group_( MPI_Fint *  fh, MPI_Aint * group, MPI_Fint * ierr)
{
  MPI_FILE_GET_GROUP( fh, group, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_group wrapper function 
******************************************************/
void mpi_file_get_group__( MPI_Fint *  fh, MPI_Aint * group, MPI_Fint * ierr)
{
  MPI_FILE_GET_GROUP( fh, group, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_get_amode wrapper function 
******************************************************/
int MPI_File_get_amode( MPI_File fh, int * amode)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_get_amode()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_get_amode( fh, amode) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_get_amode wrapper function 
******************************************************/
void MPI_FILE_GET_AMODE( MPI_Fint *  fh, MPI_Fint *  amode, MPI_Fint * ierr)
{
  *ierr = MPI_File_get_amode( MPI_File_f2c(*fh), amode) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_amode wrapper function 
******************************************************/
void mpi_file_get_amode( MPI_Fint *  fh, MPI_Fint *  amode, MPI_Fint * ierr)
{
  MPI_FILE_GET_AMODE( fh, amode, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_amode wrapper function 
******************************************************/
void mpi_file_get_amode_( MPI_Fint *  fh, MPI_Fint *  amode, MPI_Fint * ierr)
{
  MPI_FILE_GET_AMODE( fh, amode, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_amode wrapper function 
******************************************************/
void mpi_file_get_amode__( MPI_Fint *  fh, MPI_Fint *  amode, MPI_Fint * ierr)
{
  MPI_FILE_GET_AMODE( fh, amode, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_set_info wrapper function 
******************************************************/
int MPI_File_set_info( MPI_File fh, MPI_Info info)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_set_info()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_set_info( fh, info) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_set_info wrapper function 
******************************************************/
void MPI_FILE_SET_INFO( MPI_Fint *  fh, MPI_Fint *  info, MPI_Fint * ierr)
{
  *ierr = MPI_File_set_info( MPI_File_f2c(*fh), MPI_Info_f2c(*info)) ; 
  return ; 
}

/******************************************************
***      MPI_File_set_info wrapper function 
******************************************************/
void mpi_file_set_info( MPI_Fint *  fh, MPI_Fint *  info, MPI_Fint * ierr)
{
  MPI_FILE_SET_INFO( fh, info, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_set_info wrapper function 
******************************************************/
void mpi_file_set_info_( MPI_Fint *  fh, MPI_Fint *  info, MPI_Fint * ierr)
{
  MPI_FILE_SET_INFO( fh, info, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_set_info wrapper function 
******************************************************/
void mpi_file_set_info__( MPI_Fint *  fh, MPI_Fint *  info, MPI_Fint * ierr)
{
  MPI_FILE_SET_INFO( fh, info, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_get_info wrapper function 
******************************************************/
int MPI_File_get_info( MPI_File fh, MPI_Info * info_used)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_get_info()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_get_info( fh, info_used) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_get_info wrapper function 
******************************************************/
void MPI_FILE_GET_INFO( MPI_Fint *  fh, MPI_Aint * info_used, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Info local_info;

  local_fh = MPI_File_f2c(*fh);
  
  *ierr = MPI_File_get_info( local_fh, &local_info) ; 
  *info_used = MPI_Info_c2f(local_info);
  return ; 
}

/******************************************************
***      MPI_File_get_info wrapper function 
******************************************************/
void mpi_file_get_info( MPI_Fint *  fh, MPI_Aint * info_used, MPI_Fint * ierr)
{
  MPI_FILE_GET_INFO( fh, info_used, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_info wrapper function 
******************************************************/
void mpi_file_get_info_( MPI_Fint *  fh, MPI_Aint * info_used, MPI_Fint * ierr)
{
  MPI_FILE_GET_INFO( fh, info_used, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_info wrapper function 
******************************************************/
void mpi_file_get_info__( MPI_Fint *  fh, MPI_Aint * info_used, MPI_Fint * ierr)
{
  MPI_FILE_GET_INFO( fh, info_used, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_set_view wrapper function 
******************************************************/
int MPI_File_set_view( MPI_File fh, MPI_Offset disp, MPI_Datatype etype, MPI_Datatype filetype, char * datarep, MPI_Info info)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_set_view()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_set_view( fh, disp, etype, filetype, datarep, info) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_set_view wrapper function 
******************************************************/
void MPI_FILE_SET_VIEW( MPI_Fint *  fh, MPI_Offset *  disp, MPI_Fint *  etype, MPI_Fint *  filetype, char * datarep, MPI_Fint *  info, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Datatype local_etype;
  MPI_Datatype local_filetype; 
  MPI_Info local_info;

  local_fh = MPI_File_f2c(*fh);
  local_etype = MPI_Type_f2c(*etype);
  local_filetype = MPI_Type_f2c(*filetype);
  local_info = MPI_Info_f2c(*info);

  *ierr = MPI_File_set_view( local_fh, *disp, local_etype, local_filetype, datarep, local_info) ; 
  return ; 
}

/******************************************************
***      MPI_File_set_view wrapper function 
******************************************************/
void mpi_file_set_view( MPI_Fint *  fh, MPI_Offset *  disp, MPI_Fint *  etype, MPI_Fint *  filetype, char * datarep, MPI_Fint *  info, MPI_Fint * ierr)
{
  MPI_FILE_SET_VIEW( fh, disp, etype, filetype, datarep, info, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_set_view wrapper function 
******************************************************/
void mpi_file_set_view_( MPI_Fint *  fh, MPI_Offset *  disp, MPI_Fint *  etype, MPI_Fint *  filetype, char * datarep, MPI_Fint *  info, MPI_Fint * ierr)
{
  MPI_FILE_SET_VIEW( fh, disp, etype, filetype, datarep, info, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_set_view wrapper function 
******************************************************/
void mpi_file_set_view__( MPI_Fint *  fh, MPI_Offset *  disp, MPI_Fint *  etype, MPI_Fint *  filetype, char * datarep, MPI_Fint *  info, MPI_Fint * ierr)
{
  MPI_FILE_SET_VIEW( fh, disp, etype, filetype, datarep, info, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_get_view wrapper function 
******************************************************/
int MPI_File_get_view( MPI_File fh, MPI_Offset * disp, MPI_Datatype * etype, MPI_Datatype * filetype, char * datarep)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_get_view()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_get_view( fh, disp, etype, filetype, datarep) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_get_view wrapper function 
******************************************************/
void MPI_FILE_GET_VIEW( MPI_Fint *  fh, MPI_Offset * disp, MPI_Aint * etype, MPI_Aint * filetype, char * datarep, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Datatype local_etype;
  MPI_Datatype local_filetype; 

  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_get_view( local_fh, disp, &local_etype, &local_filetype, datarep) ; 
  *etype = MPI_Type_c2f(local_etype);
  *filetype = MPI_Type_c2f(local_filetype);
  return ; 
}

/******************************************************
***      MPI_File_get_view wrapper function 
******************************************************/
void mpi_file_get_view( MPI_Fint *  fh, MPI_Offset * disp, MPI_Aint * etype, MPI_Aint * filetype, char * datarep, MPI_Fint * ierr)
{
  MPI_FILE_GET_VIEW( fh, disp, etype, filetype, datarep, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_view wrapper function 
******************************************************/
void mpi_file_get_view_( MPI_Fint *  fh, MPI_Offset * disp, MPI_Aint * etype, MPI_Aint * filetype, char * datarep, MPI_Fint * ierr)
{
  MPI_FILE_GET_VIEW( fh, disp, etype, filetype, datarep, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_view wrapper function 
******************************************************/
void mpi_file_get_view__( MPI_Fint *  fh, MPI_Offset * disp, MPI_Aint * etype, MPI_Aint * filetype, char * datarep, MPI_Fint * ierr)
{
  MPI_FILE_GET_VIEW( fh, disp, etype, filetype, datarep, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_read_at wrapper function 
******************************************************/
int MPI_File_read_at( MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_at()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_at( fh, offset, buf, count, datatype, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_read_at wrapper function 
******************************************************/
void MPI_FILE_READ_AT( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Datatype local_type; 
  MPI_Status local_status; 

  local_fh = MPI_File_f2c(*fh);
  local_type = MPI_Type_f2c(*datatype);
  
  *ierr = MPI_File_read_at( local_fh, *offset, buf, *count, local_type, &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_File_read_at wrapper function 
******************************************************/
void mpi_file_read_at( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_AT( fh, offset, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_at wrapper function 
******************************************************/
void mpi_file_read_at_( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_AT( fh, offset, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_at wrapper function 
******************************************************/
void mpi_file_read_at__( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_AT( fh, offset, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_read_at_all wrapper function 
******************************************************/
int MPI_File_read_at_all( MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_at_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_at_all( fh, offset, buf, count, datatype, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_read_at_all wrapper function 
******************************************************/
void MPI_FILE_READ_AT_ALL( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Datatype local_type; 
  MPI_Status local_status; 

  local_fh = MPI_File_f2c(*fh);
  local_type = MPI_Type_f2c(*datatype);

  *ierr = MPI_File_read_at_all( local_fh, *offset, buf, *count, local_type, &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_File_read_at_all wrapper function 
******************************************************/
void mpi_file_read_at_all( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_ALL( fh, offset, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_at_all wrapper function 
******************************************************/
void mpi_file_read_at_all_( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_ALL( fh, offset, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_at_all wrapper function 
******************************************************/
void mpi_file_read_at_all__( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_ALL( fh, offset, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_write_at wrapper function 
******************************************************/
int MPI_File_write_at( MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_at()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_at( fh, offset, buf, count, datatype, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_write_at wrapper function 
******************************************************/
void MPI_FILE_WRITE_AT( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Datatype local_type; 
  MPI_Status local_status; 

  local_fh = MPI_File_f2c(*fh);
  local_type = MPI_Type_f2c(*datatype);
  *ierr = MPI_File_write_at( local_fh, *offset, buf, *count, local_type, &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_File_write_at wrapper function 
******************************************************/
void mpi_file_write_at( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT( fh, offset, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_at wrapper function 
******************************************************/
void mpi_file_write_at_( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT( fh, offset, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_at wrapper function 
******************************************************/
void mpi_file_write_at__( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT( fh, offset, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_write_at_all wrapper function 
******************************************************/
int MPI_File_write_at_all( MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_at_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_at_all( fh, offset, buf, count, datatype, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_write_at_all wrapper function 
******************************************************/
void MPI_FILE_WRITE_AT_ALL( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Datatype local_type; 
  MPI_Status local_status; 

  local_fh = MPI_File_f2c(*fh);
  local_type = MPI_Type_f2c(*datatype);
  *ierr = MPI_File_write_at_all( local_fh, *offset, buf, *count, local_type, &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_File_write_at_all wrapper function 
******************************************************/
void mpi_file_write_at_all( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_ALL( fh, offset, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_at_all wrapper function 
******************************************************/
void mpi_file_write_at_all_( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_ALL( fh, offset, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_at_all wrapper function 
******************************************************/
void mpi_file_write_at_all__( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_ALL( fh, offset, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#ifdef TAU_MPIOREQUEST

/******************************************************
***      MPI_File_iread_at wrapper function 
******************************************************/
int MPI_File_iread_at( MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPIO_Request * request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iread_at()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iread_at( fh, offset, buf, count, datatype, request) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_iread_at wrapper function 
******************************************************/
void MPI_FILE_IREAD_AT( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Datatype local_type; 
  MPIO_Request local_request; 

  local_fh = MPI_File_f2c(*fh);
  local_type = MPI_Type_f2c(*datatype);
  *ierr = MPI_File_iread_at( local_fh, *offset, buf, *count, local_type, &local_request) ; 
  *request = MPIO_Request_c2f(local_request);
  return ; 
}

/******************************************************
***      MPI_File_iread_at wrapper function 
******************************************************/
void mpi_file_iread_at( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_AT( fh, offset, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iread_at wrapper function 
******************************************************/
void mpi_file_iread_at_( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_AT( fh, offset, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iread_at wrapper function 
******************************************************/
void mpi_file_iread_at__( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_AT( fh, offset, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_iwrite_at wrapper function 
******************************************************/
int MPI_File_iwrite_at( MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPIO_Request * request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_at()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iwrite_at( fh, offset, buf, count, datatype, request) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_iwrite_at wrapper function 
******************************************************/
void MPI_FILE_IWRITE_AT( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Datatype local_type; 
  MPIO_Request local_request; 

  local_fh = MPI_File_f2c(*fh);
  local_type = MPI_Type_f2c(*datatype);
  *ierr = MPI_File_iwrite_at( local_fh, *offset, buf, *count, local_type, &local_request) ; 
  *request = MPIO_Request_c2f(local_request);
  return ; 
}

/******************************************************
***      MPI_File_iwrite_at wrapper function 
******************************************************/
void mpi_file_iwrite_at( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_AT( fh, offset, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iwrite_at wrapper function 
******************************************************/
void mpi_file_iwrite_at_( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_AT( fh, offset, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iwrite_at wrapper function 
******************************************************/
void mpi_file_iwrite_at__( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_AT( fh, offset, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#else /* ! defined TAU_MPIOREQUEST */

/******************************************************
***      MPI_File_iread_at wrapper function 
******************************************************/
int MPI_File_iread_at( MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPI_Request * request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iread_at()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iread_at( fh, offset, buf, count, datatype, request) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_iread_at wrapper function 
******************************************************/
void MPI_FILE_IREAD_AT( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Datatype local_type; 
  MPI_Request local_request; 

  local_fh = MPI_File_f2c(*fh);
  local_type = MPI_Type_f2c(*datatype);
  *ierr = MPI_File_iread_at( local_fh, *offset, buf, *count, local_type, &local_request) ; 
  *request = MPI_Request_c2f(local_request);
  return ; 
}

/******************************************************
***      MPI_File_iread_at wrapper function 
******************************************************/
void mpi_file_iread_at( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_AT( fh, offset, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iread_at wrapper function 
******************************************************/
void mpi_file_iread_at_( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_AT( fh, offset, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iread_at wrapper function 
******************************************************/
void mpi_file_iread_at__( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_AT( fh, offset, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_iwrite_at wrapper function 
******************************************************/
int MPI_File_iwrite_at( MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype, MPI_Request * request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_at()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iwrite_at( fh, offset, buf, count, datatype, request) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_iwrite_at wrapper function 
******************************************************/
void MPI_FILE_IWRITE_AT( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Datatype local_type; 
  MPI_Request local_request; 

  local_fh = MPI_File_f2c(*fh);
  local_type = MPI_Type_f2c(*datatype);
  *ierr = MPI_File_iwrite_at( local_fh, *offset, buf, *count, local_type, &local_request) ; 
  *request = MPI_Request_c2f(local_request);
  return ; 
}

/******************************************************
***      MPI_File_iwrite_at wrapper function 
******************************************************/
void mpi_file_iwrite_at( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_AT( fh, offset, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iwrite_at wrapper function 
******************************************************/
void mpi_file_iwrite_at_( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_AT( fh, offset, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iwrite_at wrapper function 
******************************************************/
void mpi_file_iwrite_at__( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_AT( fh, offset, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#endif /* TAU_MPIOREQUEST */

/******************************************************
***      MPI_File_get_atomicity wrapper function 
******************************************************/
int MPI_File_get_atomicity( MPI_File fh, int * flag)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_get_atomicity()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_get_atomicity( fh, flag) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_get_atomicity wrapper function 
******************************************************/
void MPI_FILE_GET_ATOMICITY( MPI_Fint *  fh, MPI_Fint *  flag, MPI_Fint * ierr)
{
  *ierr = MPI_File_get_atomicity( MPI_File_f2c(*fh), flag) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_atomicity wrapper function 
******************************************************/
void mpi_file_get_atomicity( MPI_Fint *  fh, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_FILE_GET_ATOMICITY( fh, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_atomicity wrapper function 
******************************************************/
void mpi_file_get_atomicity_( MPI_Fint *  fh, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_FILE_GET_ATOMICITY( fh, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_atomicity wrapper function 
******************************************************/
void mpi_file_get_atomicity__( MPI_Fint *  fh, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_FILE_GET_ATOMICITY( fh, flag, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_sync wrapper function 
******************************************************/
int MPI_File_sync( MPI_File fh)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_sync()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_sync( fh) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_sync wrapper function 
******************************************************/
void MPI_FILE_SYNC( MPI_Fint *  fh, MPI_Fint * ierr)
{
  *ierr = MPI_File_sync( MPI_File_f2c(*fh)) ; 
  return ; 
}

/******************************************************
***      MPI_File_sync wrapper function 
******************************************************/
void mpi_file_sync( MPI_Fint *  fh, MPI_Fint * ierr)
{
  MPI_FILE_SYNC( fh, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_sync wrapper function 
******************************************************/
void mpi_file_sync_( MPI_Fint *  fh, MPI_Fint * ierr)
{
  MPI_FILE_SYNC( fh, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_sync wrapper function 
******************************************************/
void mpi_file_sync__( MPI_Fint *  fh, MPI_Fint * ierr)
{
  MPI_FILE_SYNC( fh, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Type_create_subarray wrapper function 
******************************************************/
int MPI_Type_create_subarray( int ndims, int * array_of_sizes, int * array_of_subsizes, int * array_of_starts, int order, MPI_Datatype oldtype, MPI_Datatype * newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_subarray()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_subarray( ndims, array_of_sizes, array_of_subsizes, array_of_starts, order, oldtype, newtype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_create_subarray wrapper function 
******************************************************/
void MPI_TYPE_CREATE_SUBARRAY( MPI_Fint *  ndims, MPI_Fint *  array_of_sizes, MPI_Fint *  array_of_subsizes, MPI_Fint *  array_of_starts, MPI_Fint *  order, MPI_Fint *  oldtype, MPI_Aint * newtype, MPI_Fint * ierr)
{
  MPI_Datatype local_oldtype;
  MPI_Datatype local_newtype; 
  local_oldtype = MPI_Type_f2c(*oldtype);
  *ierr = MPI_Type_create_subarray( *ndims, array_of_sizes, array_of_subsizes, array_of_starts, *order, local_oldtype, &local_newtype) ; 
  *newtype = MPI_Type_c2f(local_newtype);
  return ; 
}

/******************************************************
***      MPI_Type_create_subarray wrapper function 
******************************************************/
void mpi_type_create_subarray( MPI_Fint *  ndims, MPI_Fint *  array_of_sizes, MPI_Fint *  array_of_subsizes, MPI_Fint *  array_of_starts, MPI_Fint *  order, MPI_Fint *  oldtype, MPI_Aint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_SUBARRAY( ndims, array_of_sizes, array_of_subsizes, array_of_starts, order, oldtype, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_subarray wrapper function 
******************************************************/
void mpi_type_create_subarray_( MPI_Fint *  ndims, MPI_Fint *  array_of_sizes, MPI_Fint *  array_of_subsizes, MPI_Fint *  array_of_starts, MPI_Fint *  order, MPI_Fint *  oldtype, MPI_Aint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_SUBARRAY( ndims, array_of_sizes, array_of_subsizes, array_of_starts, order, oldtype, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_subarray wrapper function 
******************************************************/
void mpi_type_create_subarray__( MPI_Fint *  ndims, MPI_Fint *  array_of_sizes, MPI_Fint *  array_of_subsizes, MPI_Fint *  array_of_starts, MPI_Fint *  order, MPI_Fint *  oldtype, MPI_Aint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_SUBARRAY( ndims, array_of_sizes, array_of_subsizes, array_of_starts, order, oldtype, newtype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Type_create_darray wrapper function 
******************************************************/
int MPI_Type_create_darray( int size, int rank, int ndims, int * array_of_gsizes, int * array_of_distribs, int * array_of_dargs, int * array_of_psizes, int order, MPI_Datatype oldtype, MPI_Datatype * newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_darray()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_darray( size, rank, ndims, array_of_gsizes, array_of_distribs, array_of_dargs, array_of_psizes, order, oldtype, newtype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_create_darray wrapper function 
******************************************************/
void MPI_TYPE_CREATE_DARRAY( MPI_Fint *  size, MPI_Fint *  rank, MPI_Fint *  ndims, MPI_Fint *  array_of_gsizes, MPI_Fint *  array_of_distribs, MPI_Fint *  array_of_dargs, MPI_Fint *  array_of_psizes, MPI_Fint *  order, MPI_Fint *  oldtype, MPI_Aint * newtype, MPI_Fint * ierr)
{
  MPI_Datatype local_newtype;
  *ierr = MPI_Type_create_darray( *size, *rank, *ndims, array_of_gsizes, array_of_distribs, array_of_dargs, array_of_psizes, *order, MPI_Type_f2c(*oldtype), &local_newtype) ; 
  *newtype = MPI_Type_c2f(local_newtype);
  return ; 
}

/******************************************************
***      MPI_Type_create_darray wrapper function 
******************************************************/
void mpi_type_create_darray( MPI_Fint *  size, MPI_Fint *  rank, MPI_Fint *  ndims, MPI_Fint *  array_of_gsizes, MPI_Fint *  array_of_distribs, MPI_Fint *  array_of_dargs, MPI_Fint *  array_of_psizes, MPI_Fint *  order, MPI_Fint *  oldtype, MPI_Aint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_DARRAY( size, rank, ndims, array_of_gsizes, array_of_distribs, array_of_dargs, array_of_psizes, order, oldtype, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_darray wrapper function 
******************************************************/
void mpi_type_create_darray_( MPI_Fint *  size, MPI_Fint *  rank, MPI_Fint *  ndims, MPI_Fint *  array_of_gsizes, MPI_Fint *  array_of_distribs, MPI_Fint *  array_of_dargs, MPI_Fint *  array_of_psizes, MPI_Fint *  order, MPI_Fint *  oldtype, MPI_Aint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_DARRAY( size, rank, ndims, array_of_gsizes, array_of_distribs, array_of_dargs, array_of_psizes, order, oldtype, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_darray wrapper function 
******************************************************/
void mpi_type_create_darray__( MPI_Fint *  size, MPI_Fint *  rank, MPI_Fint *  ndims, MPI_Fint *  array_of_gsizes, MPI_Fint *  array_of_distribs, MPI_Fint *  array_of_dargs, MPI_Fint *  array_of_psizes, MPI_Fint *  order, MPI_Fint *  oldtype, MPI_Aint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_DARRAY( size, rank, ndims, array_of_gsizes, array_of_distribs, array_of_dargs, array_of_psizes, order, oldtype, newtype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#ifdef TAU_MPIERRHANDLER

/******************************************************
***      MPI_File_create_errhandler wrapper function 
******************************************************/
int MPI_File_create_errhandler( MPI_File_errhandler_fn * function, MPI_Errhandler * errhandler)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_create_errhandler()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_create_errhandler( function, errhandler) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_create_errhandler wrapper function 
******************************************************/
void MPI_FILE_CREATE_ERRHANDLER( MPI_File_errhandler_fn * function, MPI_Fint * errhandler, MPI_Fint * ierr)
{
  MPI_Errhandler local_errhandler;
  *ierr = MPI_File_create_errhandler( function, &local_errhandler) ; 
  *errhandler = MPI_Errhandler_c2f(local_errhandler);
  return ; 
}

/******************************************************
***      MPI_File_create_errhandler wrapper function 
******************************************************/
void mpi_file_create_errhandler( MPI_File_errhandler_fn * function, MPI_Fint * errhandler, MPI_Fint * ierr)
{
  MPI_FILE_CREATE_ERRHANDLER( function, errhandler, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_create_errhandler wrapper function 
******************************************************/
void mpi_file_create_errhandler_( MPI_File_errhandler_fn * function, MPI_Fint * errhandler, MPI_Fint * ierr)
{
  MPI_FILE_CREATE_ERRHANDLER( function, errhandler, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_create_errhandler wrapper function 
******************************************************/
void mpi_file_create_errhandler__( MPI_File_errhandler_fn * function, MPI_Fint * errhandler, MPI_Fint * ierr)
{
  MPI_FILE_CREATE_ERRHANDLER( function, errhandler, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_set_errhandler wrapper function 
******************************************************/
int MPI_File_set_errhandler( MPI_File fh, MPI_Errhandler errhandler)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_set_errhandler()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_set_errhandler( fh, errhandler) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_set_errhandler wrapper function 
******************************************************/
void MPI_FILE_SET_ERRHANDLER( MPI_Fint *  fh, MPI_Fint *  errhandler, MPI_Fint * ierr)
{
  *ierr = MPI_File_set_errhandler( MPI_File_f2c(*fh), MPI_Errhandler_f2c(*errhandler)) ; 
  return ; 
}

/******************************************************
***      MPI_File_set_errhandler wrapper function 
******************************************************/
void mpi_file_set_errhandler( MPI_Fint *  fh, MPI_Fint *  errhandler, MPI_Fint * ierr)
{
  MPI_FILE_SET_ERRHANDLER( fh, errhandler, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_set_errhandler wrapper function 
******************************************************/
void mpi_file_set_errhandler_( MPI_Fint *  fh, MPI_Fint *  errhandler, MPI_Fint * ierr)
{
  MPI_FILE_SET_ERRHANDLER( fh, errhandler, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_set_errhandler wrapper function 
******************************************************/
void mpi_file_set_errhandler__( MPI_Fint *  fh, MPI_Fint *  errhandler, MPI_Fint * ierr)
{
  MPI_FILE_SET_ERRHANDLER( fh, errhandler, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_get_errhandler wrapper function 
******************************************************/
int MPI_File_get_errhandler( MPI_File fh, MPI_Errhandler * errhandler)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_get_errhandler()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_get_errhandler( fh, errhandler) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_get_errhandler wrapper function 
******************************************************/
void MPI_FILE_GET_ERRHANDLER( MPI_Fint *  fh, MPI_Aint * errhandler, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Errhandler local_errhandler; 

  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_get_errhandler( local_fh, &local_errhandler) ; 
  *errhandler = MPI_Errhandler_c2f(local_errhandler);
  return ; 
}

/******************************************************
***      MPI_File_get_errhandler wrapper function 
******************************************************/
void mpi_file_get_errhandler( MPI_Fint *  fh, MPI_Aint * errhandler, MPI_Fint * ierr)
{
  MPI_FILE_GET_ERRHANDLER( fh, errhandler, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_errhandler wrapper function 
******************************************************/
void mpi_file_get_errhandler_( MPI_Fint *  fh, MPI_Aint * errhandler, MPI_Fint * ierr)
{
  MPI_FILE_GET_ERRHANDLER( fh, errhandler, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_errhandler wrapper function 
******************************************************/
void mpi_file_get_errhandler__( MPI_Fint *  fh, MPI_Aint * errhandler, MPI_Fint * ierr)
{
  MPI_FILE_GET_ERRHANDLER( fh, errhandler, ierr) ; 
  return ; 
}

#endif /* TAU_MPIERRHANDLER */

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_get_byte_offset wrapper function 
******************************************************/
int MPI_File_get_byte_offset( MPI_File fh, MPI_Offset offset, MPI_Offset * disp)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_get_byte_offset()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_get_byte_offset( fh, offset, disp) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_get_byte_offset wrapper function 
******************************************************/
void MPI_FILE_GET_BYTE_OFFSET( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Offset * disp, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Offset local_offset;
  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_get_byte_offset( local_fh, *offset, &local_offset) ; 
  *disp = local_offset; 
  return ; 
}

/******************************************************
***      MPI_File_get_byte_offset wrapper function 
******************************************************/
void mpi_file_get_byte_offset( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Offset * disp, MPI_Fint * ierr)
{
  MPI_FILE_GET_BYTE_OFFSET( fh, offset, disp, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_byte_offset wrapper function 
******************************************************/
void mpi_file_get_byte_offset_( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Offset * disp, MPI_Fint * ierr)
{
  MPI_FILE_GET_BYTE_OFFSET( fh, offset, disp, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_byte_offset wrapper function 
******************************************************/
void mpi_file_get_byte_offset__( MPI_Fint *  fh, MPI_Offset *  offset, MPI_Offset * disp, MPI_Fint * ierr)
{
  MPI_FILE_GET_BYTE_OFFSET( fh, offset, disp, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_get_position wrapper function 
******************************************************/
int MPI_File_get_position( MPI_File fh, MPI_Offset * offset)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_get_position()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_get_position( fh, offset) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_get_position wrapper function 
******************************************************/
void MPI_FILE_GET_POSITION( MPI_Fint *  fh, MPI_Offset * offset, MPI_Fint * ierr)
{
  MPI_File local_fh; 
  MPI_Offset local_offset;
  local_fh = MPI_File_f2c(*fh);

  *ierr = MPI_File_get_position( local_fh, &local_offset) ; 
  *offset = local_offset; 
  return ; 
}

/******************************************************
***      MPI_File_get_position wrapper function 
******************************************************/
void mpi_file_get_position( MPI_Fint *  fh, MPI_Offset * offset, MPI_Fint * ierr)
{
  MPI_FILE_GET_POSITION( fh, offset, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_position wrapper function 
******************************************************/
void mpi_file_get_position_( MPI_Fint *  fh, MPI_Offset * offset, MPI_Fint * ierr)
{
  MPI_FILE_GET_POSITION( fh, offset, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_position wrapper function 
******************************************************/
void mpi_file_get_position__( MPI_Fint *  fh, MPI_Offset * offset, MPI_Fint * ierr)
{
  MPI_FILE_GET_POSITION( fh, offset, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_get_position_shared wrapper function 
******************************************************/
int MPI_File_get_position_shared( MPI_File fh, MPI_Offset * offset)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_get_position_shared()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_get_position_shared( fh, offset) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_get_position_shared wrapper function 
******************************************************/
void MPI_FILE_GET_POSITION_SHARED( MPI_Fint *  fh, MPI_Offset * offset, MPI_Fint * ierr)
{
  MPI_File local_fh; 
  local_fh = MPI_File_f2c(*fh);

  *ierr = MPI_File_get_position_shared( local_fh, offset) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_position_shared wrapper function 
******************************************************/
void mpi_file_get_position_shared( MPI_Fint *  fh, MPI_Offset * offset, MPI_Fint * ierr)
{
  MPI_FILE_GET_POSITION_SHARED( fh, offset, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_position_shared wrapper function 
******************************************************/
void mpi_file_get_position_shared_( MPI_Fint *  fh, MPI_Offset * offset, MPI_Fint * ierr)
{
  MPI_FILE_GET_POSITION_SHARED( fh, offset, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_position_shared wrapper function 
******************************************************/
void mpi_file_get_position_shared__( MPI_Fint *  fh, MPI_Offset * offset, MPI_Fint * ierr)
{
  MPI_FILE_GET_POSITION_SHARED( fh, offset, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_get_type_extent wrapper function 
******************************************************/
int MPI_File_get_type_extent( MPI_File fh, MPI_Datatype datatype, MPI_Aint * extent)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_get_type_extent()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_get_type_extent( fh, datatype, extent) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_get_type_extent wrapper function 
******************************************************/
void MPI_FILE_GET_TYPE_EXTENT( MPI_Fint *  fh, MPI_Fint *  datatype, MPI_Aint * extent, MPI_Fint * ierr)
{
  *ierr = MPI_File_get_type_extent( MPI_File_f2c(*fh), MPI_Type_f2c(*datatype), extent) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_type_extent wrapper function 
******************************************************/
void mpi_file_get_type_extent( MPI_Fint *  fh, MPI_Fint *  datatype, MPI_Aint * extent, MPI_Fint * ierr)
{
  MPI_FILE_GET_TYPE_EXTENT( fh, datatype, extent, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_type_extent wrapper function 
******************************************************/
void mpi_file_get_type_extent_( MPI_Fint *  fh, MPI_Fint *  datatype, MPI_Aint * extent, MPI_Fint * ierr)
{
  MPI_FILE_GET_TYPE_EXTENT( fh, datatype, extent, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_get_type_extent wrapper function 
******************************************************/
void mpi_file_get_type_extent__( MPI_Fint *  fh, MPI_Fint *  datatype, MPI_Aint * extent, MPI_Fint * ierr)
{
  MPI_FILE_GET_TYPE_EXTENT( fh, datatype, extent, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


#ifdef TAU_MPIOREQUEST

/******************************************************
***      MPI_File_iread wrapper function 
******************************************************/
int MPI_File_iread( MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPIO_Request * request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iread()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iread( fh, buf, count, datatype, request) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_iread wrapper function 
******************************************************/
void MPI_FILE_IREAD( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_File local_fh; 
  MPI_Datatype local_type; 
  MPIO_Request local_request;
  local_fh = MPI_File_f2c(*fh);
  local_type = MPI_Type_f2c(*datatype);
  *ierr = MPI_File_iread( local_fh, buf, *count, local_type, &local_request) ; 
  *fh = MPI_File_c2f(local_fh);
  *request = MPIO_Request_c2f(local_request);
  return ; 
}

/******************************************************
***      MPI_File_iread wrapper function 
******************************************************/
void mpi_file_iread( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iread wrapper function 
******************************************************/
void mpi_file_iread_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iread wrapper function 
******************************************************/
void mpi_file_iread__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_iread_shared wrapper function 
******************************************************/
int MPI_File_iread_shared( MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPIO_Request * request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iread_shared()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iread_shared( fh, buf, count, datatype, request) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_iread_shared wrapper function 
******************************************************/
void MPI_FILE_IREAD_SHARED( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Datatype local_type;
  MPIO_Request local_request;
  
  local_fh = MPI_File_f2c(*fh);
  local_type = MPI_Type_f2c(*datatype);

  *ierr = MPI_File_iread_shared( local_fh, buf, *count, local_type, &local_request) ; 
  *request = MPIO_Request_c2f(local_request);
  return ; 
}

/******************************************************
***      MPI_File_iread_shared wrapper function 
******************************************************/
void mpi_file_iread_shared( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_SHARED( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iread_shared wrapper function 
******************************************************/
void mpi_file_iread_shared_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_SHARED( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iread_shared wrapper function 
******************************************************/
void mpi_file_iread_shared__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_SHARED( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_iwrite wrapper function 
******************************************************/
int MPI_File_iwrite( MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPIO_Request * request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iwrite( fh, buf, count, datatype, request) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_iwrite wrapper function 
******************************************************/
void MPI_FILE_IWRITE( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPIO_Request local_request;
  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_iwrite( local_fh, buf, *count, MPI_Type_f2c(*datatype), &local_request) ; 
  *fh = MPI_File_c2f(local_fh);
  *request = MPIO_Request_c2f(local_request);
  return ; 
}

/******************************************************
***      MPI_File_iwrite wrapper function 
******************************************************/
void mpi_file_iwrite( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iwrite wrapper function 
******************************************************/
void mpi_file_iwrite_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iwrite wrapper function 
******************************************************/
void mpi_file_iwrite__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_iwrite_shared wrapper function 
******************************************************/
int MPI_File_iwrite_shared( MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPIO_Request * request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_shared()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iwrite_shared( fh, buf, count, datatype, request) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_iwrite_shared wrapper function 
******************************************************/
void MPI_FILE_IWRITE_SHARED( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPIO_Request local_request;
  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_iwrite_shared( local_fh, buf, *count, MPI_Type_f2c(*datatype), &local_request) ; 
  *fh = MPI_File_c2f(local_fh);
  *request = MPIO_Request_c2f(local_request);
  return ; 
}

/******************************************************
***      MPI_File_iwrite_shared wrapper function 
******************************************************/
void mpi_file_iwrite_shared( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_SHARED( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iwrite_shared wrapper function 
******************************************************/
void mpi_file_iwrite_shared_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_SHARED( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iwrite_shared wrapper function 
******************************************************/
void mpi_file_iwrite_shared__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_SHARED( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#else /* ! defined TAU_MPIOREQUEST */

/******************************************************
***      MPI_File_iread wrapper function 
******************************************************/
int MPI_File_iread( MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Request * request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iread()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iread( fh, buf, count, datatype, request) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_iread wrapper function 
******************************************************/
void MPI_FILE_IREAD( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_File local_fh; 
  MPI_Datatype local_type; 
  MPI_Request local_request;
  local_fh = MPI_File_f2c(*fh);
  local_type = MPI_Type_f2c(*datatype);
  *ierr = MPI_File_iread( local_fh, buf, *count, local_type, &local_request) ; 
  *fh = MPI_File_c2f(local_fh);
  *request = MPI_Request_c2f(local_request);
  return ; 
}

/******************************************************
***      MPI_File_iread wrapper function 
******************************************************/
void mpi_file_iread( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iread wrapper function 
******************************************************/
void mpi_file_iread_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iread wrapper function 
******************************************************/
void mpi_file_iread__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_iread_shared wrapper function 
******************************************************/
int MPI_File_iread_shared( MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Request * request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iread_shared()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iread_shared( fh, buf, count, datatype, request) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_iread_shared wrapper function 
******************************************************/
void MPI_FILE_IREAD_SHARED( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Datatype local_type;
  MPI_Request local_request;
  
  local_fh = MPI_File_f2c(*fh);
  local_type = MPI_Type_f2c(*datatype);

  *ierr = MPI_File_iread_shared( local_fh, buf, *count, local_type, &local_request) ; 
  *request = MPI_Request_c2f(local_request);
  return ; 
}

/******************************************************
***      MPI_File_iread_shared wrapper function 
******************************************************/
void mpi_file_iread_shared( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_SHARED( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iread_shared wrapper function 
******************************************************/
void mpi_file_iread_shared_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_SHARED( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iread_shared wrapper function 
******************************************************/
void mpi_file_iread_shared__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IREAD_SHARED( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_iwrite wrapper function 
******************************************************/
int MPI_File_iwrite( MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Request * request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iwrite( fh, buf, count, datatype, request) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_iwrite wrapper function 
******************************************************/
void MPI_FILE_IWRITE( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Request local_request;
  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_iwrite( local_fh, buf, *count, MPI_Type_f2c(*datatype), &local_request) ; 
  *fh = MPI_File_c2f(local_fh);
  *request = MPI_Request_c2f(local_request);
  return ; 
}

/******************************************************
***      MPI_File_iwrite wrapper function 
******************************************************/
void mpi_file_iwrite( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iwrite wrapper function 
******************************************************/
void mpi_file_iwrite_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iwrite wrapper function 
******************************************************/
void mpi_file_iwrite__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_iwrite_shared wrapper function 
******************************************************/
int MPI_File_iwrite_shared( MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Request * request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_iwrite_shared()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_iwrite_shared( fh, buf, count, datatype, request) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_iwrite_shared wrapper function 
******************************************************/
void MPI_FILE_IWRITE_SHARED( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Request local_request;

  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_iwrite_shared( local_fh, buf, *count, MPI_Type_f2c(*datatype), &local_request) ; 
  *fh = MPI_File_c2f(local_fh);
  *request = MPI_Request_c2f(local_request);
  return ; 
}

/******************************************************
***      MPI_File_iwrite_shared wrapper function 
******************************************************/
void mpi_file_iwrite_shared( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_SHARED( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iwrite_shared wrapper function 
******************************************************/
void mpi_file_iwrite_shared_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_SHARED( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_iwrite_shared wrapper function 
******************************************************/
void mpi_file_iwrite_shared__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_FILE_IWRITE_SHARED( fh, buf, count, datatype, request, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/
#endif /* TAU_MPIOREQUEST */

/******************************************************
***      MPI_File_preallocate wrapper function 
******************************************************/
int MPI_File_preallocate( MPI_File fh, MPI_Offset size)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_preallocate()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_preallocate( fh, size) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_preallocate wrapper function 
******************************************************/
void MPI_FILE_PREALLOCATE( MPI_Fint *  fh, MPI_Fint *  size, MPI_Fint * ierr)
{
  *ierr = MPI_File_preallocate( MPI_File_f2c(*fh), *size) ; 
  return ; 
}

/******************************************************
***      MPI_File_preallocate wrapper function 
******************************************************/
void mpi_file_preallocate( MPI_Fint *  fh, MPI_Fint *  size, MPI_Fint * ierr)
{
  MPI_FILE_PREALLOCATE( fh, size, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_preallocate wrapper function 
******************************************************/
void mpi_file_preallocate_( MPI_Fint *  fh, MPI_Fint *  size, MPI_Fint * ierr)
{
  MPI_FILE_PREALLOCATE( fh, size, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_preallocate wrapper function 
******************************************************/
void mpi_file_preallocate__( MPI_Fint *  fh, MPI_Fint *  size, MPI_Fint * ierr)
{
  MPI_FILE_PREALLOCATE( fh, size, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_read wrapper function 
******************************************************/
int MPI_File_read( MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read( fh, buf, count, datatype, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_read wrapper function 
******************************************************/
void MPI_FILE_READ( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Status local_status;
  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_read( local_fh, buf, *count, MPI_Type_f2c(*datatype), &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  *fh = MPI_File_c2f(local_fh);
  
  return ; 
}

/******************************************************
***      MPI_File_read wrapper function 
******************************************************/
void mpi_file_read( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read wrapper function 
******************************************************/
void mpi_file_read_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read wrapper function 
******************************************************/
void mpi_file_read__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_read_all wrapper function 
******************************************************/
int MPI_File_read_all( MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_all()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_all( fh, buf, count, datatype, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_read_all wrapper function 
******************************************************/
void MPI_FILE_READ_ALL( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Status local_status;
  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_read_all( local_fh, buf, *count, MPI_Type_f2c(*datatype), &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_File_read_all wrapper function 
******************************************************/
void mpi_file_read_all( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ALL( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_all wrapper function 
******************************************************/
void mpi_file_read_all_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ALL( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_all wrapper function 
******************************************************/
void mpi_file_read_all__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ALL( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_read_all_begin wrapper function 
******************************************************/
int MPI_File_read_all_begin( MPI_File fh, void * buf, int count, MPI_Datatype datatype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_all_begin()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_all_begin( fh, buf, count, datatype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_read_all_begin wrapper function 
******************************************************/
void MPI_FILE_READ_ALL_BEGIN( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  *ierr = MPI_File_read_all_begin( MPI_File_f2c(*fh), buf, *count, MPI_Type_f2c(*datatype)) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_all_begin wrapper function 
******************************************************/
void mpi_file_read_all_begin( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_ALL_BEGIN( fh, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_all_begin wrapper function 
******************************************************/
void mpi_file_read_all_begin_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_ALL_BEGIN( fh, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_all_begin wrapper function 
******************************************************/
void mpi_file_read_all_begin__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_ALL_BEGIN( fh, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_read_all_end wrapper function 
******************************************************/
int MPI_File_read_all_end( MPI_File fh, void * buf, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_all_end()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_all_end( fh, buf, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_read_all_end wrapper function 
******************************************************/
void MPI_FILE_READ_ALL_END( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Status local_status;
  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_read_all_end( local_fh, buf, &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_File_read_all_end wrapper function 
******************************************************/
void mpi_file_read_all_end( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ALL_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_all_end wrapper function 
******************************************************/
void mpi_file_read_all_end_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ALL_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_all_end wrapper function 
******************************************************/
void mpi_file_read_all_end__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ALL_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_read_at_all_begin wrapper function 
******************************************************/
int MPI_File_read_at_all_begin( MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_at_all_begin()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_at_all_begin( fh, offset, buf, count, datatype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_read_at_all_begin wrapper function 
******************************************************/
void MPI_FILE_READ_AT_ALL_BEGIN( MPI_Fint *  fh, MPI_Fint *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  *ierr = MPI_File_read_at_all_begin( MPI_File_f2c(*fh), *offset, buf, *count, MPI_Type_f2c(*datatype)) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_at_all_begin wrapper function 
******************************************************/
void mpi_file_read_at_all_begin( MPI_Fint *  fh, MPI_Fint *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_ALL_BEGIN( fh, offset, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_at_all_begin wrapper function 
******************************************************/
void mpi_file_read_at_all_begin_( MPI_Fint *  fh, MPI_Fint *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_ALL_BEGIN( fh, offset, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_at_all_begin wrapper function 
******************************************************/
void mpi_file_read_at_all_begin__( MPI_Fint *  fh, MPI_Fint *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_ALL_BEGIN( fh, offset, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_read_at_all_end wrapper function 
******************************************************/
int MPI_File_read_at_all_end( MPI_File fh, void * buf, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_at_all_end()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_at_all_end( fh, buf, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_read_at_all_end wrapper function 
******************************************************/
void MPI_FILE_READ_AT_ALL_END( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Status local_status;

  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_read_at_all_end( local_fh, buf, &local_status) ; 
  MPI_Status_c2f(&local_status, status);

  return ; 
}

/******************************************************
***      MPI_File_read_at_all_end wrapper function 
******************************************************/
void mpi_file_read_at_all_end( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_ALL_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_at_all_end wrapper function 
******************************************************/
void mpi_file_read_at_all_end_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_ALL_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_at_all_end wrapper function 
******************************************************/
void mpi_file_read_at_all_end__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_AT_ALL_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_read_ordered wrapper function 
******************************************************/
int MPI_File_read_ordered( MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_ordered()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_ordered( fh, buf, count, datatype, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_read_ordered wrapper function 
******************************************************/
void MPI_FILE_READ_ORDERED( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Status local_status;

  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_read_ordered( local_fh, buf, *count, MPI_Type_f2c(*datatype), &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_File_read_ordered wrapper function 
******************************************************/
void mpi_file_read_ordered( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ORDERED( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_ordered wrapper function 
******************************************************/
void mpi_file_read_ordered_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ORDERED( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_ordered wrapper function 
******************************************************/
void mpi_file_read_ordered__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ORDERED( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_read_ordered_begin wrapper function 
******************************************************/
int MPI_File_read_ordered_begin( MPI_File fh, void * buf, int count, MPI_Datatype datatype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_ordered_begin()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_ordered_begin( fh, buf, count, datatype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_read_ordered_begin wrapper function 
******************************************************/
void MPI_FILE_READ_ORDERED_BEGIN( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  *ierr = MPI_File_read_ordered_begin( MPI_File_f2c(*fh), buf, *count, MPI_Type_f2c(*datatype)) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_ordered_begin wrapper function 
******************************************************/
void mpi_file_read_ordered_begin( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_ORDERED_BEGIN( fh, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_ordered_begin wrapper function 
******************************************************/
void mpi_file_read_ordered_begin_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_ORDERED_BEGIN( fh, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_ordered_begin wrapper function 
******************************************************/
void mpi_file_read_ordered_begin__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_READ_ORDERED_BEGIN( fh, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_read_ordered_end wrapper function 
******************************************************/
int MPI_File_read_ordered_end( MPI_File fh, void * buf, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_ordered_end()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_ordered_end( fh, buf, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_read_ordered_end wrapper function 
******************************************************/
void MPI_FILE_READ_ORDERED_END( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Status local_status;

  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_read_ordered_end( local_fh, buf, &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_File_read_ordered_end wrapper function 
******************************************************/
void mpi_file_read_ordered_end( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ORDERED_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_ordered_end wrapper function 
******************************************************/
void mpi_file_read_ordered_end_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ORDERED_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_ordered_end wrapper function 
******************************************************/
void mpi_file_read_ordered_end__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_ORDERED_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_read_shared wrapper function 
******************************************************/
int MPI_File_read_shared( MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_read_shared()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_read_shared( fh, buf, count, datatype, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_read_shared wrapper function 
******************************************************/
void MPI_FILE_READ_SHARED( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Status local_status;

  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_read_shared( local_fh, buf, *count, MPI_Type_f2c(*datatype), &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_File_read_shared wrapper function 
******************************************************/
void mpi_file_read_shared( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_SHARED( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_shared wrapper function 
******************************************************/
void mpi_file_read_shared_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_SHARED( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_read_shared wrapper function 
******************************************************/
void mpi_file_read_shared__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_READ_SHARED( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_seek wrapper function 
******************************************************/
int MPI_File_seek( MPI_File fh, MPI_Offset offset, int whence)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_seek()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_seek( fh, offset, whence) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_seek wrapper function 
******************************************************/
void MPI_FILE_SEEK( MPI_Fint *  fh, MPI_Fint *  offset, MPI_Fint *  whence, MPI_Fint * ierr)
{
  *ierr = MPI_File_seek( MPI_File_f2c(*fh), *offset, *whence) ; 
  return ; 
}

/******************************************************
***      MPI_File_seek wrapper function 
******************************************************/
void mpi_file_seek( MPI_Fint *  fh, MPI_Fint *  offset, MPI_Fint *  whence, MPI_Fint * ierr)
{
  MPI_FILE_SEEK( fh, offset, whence, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_seek wrapper function 
******************************************************/
void mpi_file_seek_( MPI_Fint *  fh, MPI_Fint *  offset, MPI_Fint *  whence, MPI_Fint * ierr)
{
  MPI_FILE_SEEK( fh, offset, whence, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_seek wrapper function 
******************************************************/
void mpi_file_seek__( MPI_Fint *  fh, MPI_Fint *  offset, MPI_Fint *  whence, MPI_Fint * ierr)
{
  MPI_FILE_SEEK( fh, offset, whence, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_seek_shared wrapper function 
******************************************************/
int MPI_File_seek_shared( MPI_File fh, MPI_Offset offset, int whence)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_seek_shared()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_seek_shared( fh, offset, whence) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_seek_shared wrapper function 
******************************************************/
void MPI_FILE_SEEK_SHARED( MPI_Fint *  fh, MPI_Fint *  offset, MPI_Fint *  whence, MPI_Fint * ierr)
{
  *ierr = MPI_File_seek_shared( MPI_File_f2c(*fh), *offset, *whence) ; 
  return ; 
}

/******************************************************
***      MPI_File_seek_shared wrapper function 
******************************************************/
void mpi_file_seek_shared( MPI_Fint *  fh, MPI_Fint *  offset, MPI_Fint *  whence, MPI_Fint * ierr)
{
  MPI_FILE_SEEK_SHARED( fh, offset, whence, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_seek_shared wrapper function 
******************************************************/
void mpi_file_seek_shared_( MPI_Fint *  fh, MPI_Fint *  offset, MPI_Fint *  whence, MPI_Fint * ierr)
{
  MPI_FILE_SEEK_SHARED( fh, offset, whence, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_seek_shared wrapper function 
******************************************************/
void mpi_file_seek_shared__( MPI_Fint *  fh, MPI_Fint *  offset, MPI_Fint *  whence, MPI_Fint * ierr)
{
  MPI_FILE_SEEK_SHARED( fh, offset, whence, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_set_atomicity wrapper function 
******************************************************/
int MPI_File_set_atomicity( MPI_File fh, int flag)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_set_atomicity()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_set_atomicity( fh, flag) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_set_atomicity wrapper function 
******************************************************/
void MPI_FILE_SET_ATOMICITY( MPI_Fint *  fh, MPI_Fint *  flag, MPI_Fint * ierr)
{
  *ierr = MPI_File_set_atomicity( MPI_File_f2c(*fh), *flag) ; 
  return ; 
}

/******************************************************
***      MPI_File_set_atomicity wrapper function 
******************************************************/
void mpi_file_set_atomicity( MPI_Fint *  fh, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_FILE_SET_ATOMICITY( fh, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_set_atomicity wrapper function 
******************************************************/
void mpi_file_set_atomicity_( MPI_Fint *  fh, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_FILE_SET_ATOMICITY( fh, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_set_atomicity wrapper function 
******************************************************/
void mpi_file_set_atomicity__( MPI_Fint *  fh, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_FILE_SET_ATOMICITY( fh, flag, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_write wrapper function 
******************************************************/
int MPI_File_write( MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)
{
  int retvalue, typesize; 
  double currentWrite = 0.0;
  struct timeval t1, t2;

  TAU_PROFILE_TIMER(t, "MPI_File_write()", "", TAU_MESSAGE); 
  TAU_REGISTER_EVENT(wb, "WRITE Bandwidth (MB/s)");
  TAU_REGISTER_EVENT(byteswritten, "Bytes Written");

  TAU_PROFILE_START(t); 
  gettimeofday(&t1, 0);
  retvalue = PMPI_File_write( fh, buf, count, datatype, status) ; 
  gettimeofday(&t2, 0);

  /* calculate the time spent in operation */
  currentWrite = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  PMPI_Type_size(datatype, &typesize);
  if (currentWrite > 1e-12) {
    TAU_EVENT(wb, (double) count*typesize/currentWrite);
  }
  else {
    printf("TauWrapperWrite: currentWrite = %g\n", currentWrite);
  }
  TAU_EVENT(byteswritten, count*typesize);

  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_write wrapper function 
******************************************************/
void MPI_FILE_WRITE( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Status local_status;

  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_write( local_fh, buf, *count, MPI_Type_f2c(*datatype), &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_File_write wrapper function 
******************************************************/
void mpi_file_write( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write wrapper function 
******************************************************/
void mpi_file_write_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write wrapper function 
******************************************************/
void mpi_file_write__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_write_all wrapper function 
******************************************************/
int MPI_File_write_all( MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)
{
  int retvalue, typesize; 
  double currentWrite = 0.0;
  struct timeval t1, t2;

  TAU_PROFILE_TIMER(t, "MPI_File_write_all()", "", TAU_MESSAGE); 
  TAU_REGISTER_EVENT(wb, "WRITE Bandwidth (MB/s)");
  TAU_REGISTER_EVENT(byteswritten, "Bytes Written");

  TAU_PROFILE_START(t); 
  gettimeofday(&t1, 0);
  retvalue = PMPI_File_write_all( fh, buf, count, datatype, status) ; 
  gettimeofday(&t2, 0);

  /* calculate the time spent in operation */
  currentWrite = (double) (t2.tv_sec - t1.tv_sec) * 1.0e6 + (t2.tv_usec - t1.tv_usec);
  /* now we trigger the events */
  PMPI_Type_size(datatype, &typesize);
  if (currentWrite > 1e-12) {
    TAU_EVENT(wb, (double) count*typesize/currentWrite);
  }
  else {
    printf("TauWrapperWrite: currentWrite = %g\n", currentWrite);
  }
  TAU_EVENT(byteswritten, count*typesize);

  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_write_all wrapper function 
******************************************************/
void MPI_FILE_WRITE_ALL( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Status local_status;

  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_write_all( local_fh, buf, *count, MPI_Type_f2c(*datatype), &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_File_write_all wrapper function 
******************************************************/
void mpi_file_write_all( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ALL( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_all wrapper function 
******************************************************/
void mpi_file_write_all_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ALL( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_all wrapper function 
******************************************************/
void mpi_file_write_all__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ALL( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_write_all_begin wrapper function 
******************************************************/
int MPI_File_write_all_begin( MPI_File fh, void * buf, int count, MPI_Datatype datatype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_all_begin()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_all_begin( fh, buf, count, datatype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_write_all_begin wrapper function 
******************************************************/
void MPI_FILE_WRITE_ALL_BEGIN( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  *ierr = MPI_File_write_all_begin( MPI_File_f2c(*fh), buf, *count, MPI_Type_f2c(*datatype)) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_all_begin wrapper function 
******************************************************/
void mpi_file_write_all_begin( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ALL_BEGIN( fh, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_all_begin wrapper function 
******************************************************/
void mpi_file_write_all_begin_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ALL_BEGIN( fh, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_all_begin wrapper function 
******************************************************/
void mpi_file_write_all_begin__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ALL_BEGIN( fh, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_write_all_end wrapper function 
******************************************************/
int MPI_File_write_all_end( MPI_File fh, void * buf, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_all_end()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_all_end( fh, buf, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_write_all_end wrapper function 
******************************************************/
void MPI_FILE_WRITE_ALL_END( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Status local_status;

  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_write_all_end( local_fh, buf, &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_File_write_all_end wrapper function 
******************************************************/
void mpi_file_write_all_end( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ALL_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_all_end wrapper function 
******************************************************/
void mpi_file_write_all_end_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ALL_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_all_end wrapper function 
******************************************************/
void mpi_file_write_all_end__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ALL_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_write_at_all_begin wrapper function 
******************************************************/
int MPI_File_write_at_all_begin( MPI_File fh, MPI_Offset offset, void * buf, int count, MPI_Datatype datatype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_at_all_begin()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_at_all_begin( fh, offset, buf, count, datatype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_write_at_all_begin wrapper function 
******************************************************/
void MPI_FILE_WRITE_AT_ALL_BEGIN( MPI_Fint *  fh, MPI_Fint *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  *ierr = MPI_File_write_at_all_begin( MPI_File_f2c(*fh), *offset, buf, *count, MPI_Type_f2c(*datatype)) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_at_all_begin wrapper function 
******************************************************/
void mpi_file_write_at_all_begin( MPI_Fint *  fh, MPI_Fint *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_ALL_BEGIN( fh, offset, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_at_all_begin wrapper function 
******************************************************/
void mpi_file_write_at_all_begin_( MPI_Fint *  fh, MPI_Fint *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_ALL_BEGIN( fh, offset, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_at_all_begin wrapper function 
******************************************************/
void mpi_file_write_at_all_begin__( MPI_Fint *  fh, MPI_Fint *  offset, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_ALL_BEGIN( fh, offset, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_write_at_all_end wrapper function 
******************************************************/
int MPI_File_write_at_all_end( MPI_File fh, void * buf, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_at_all_end()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_at_all_end( fh, buf, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_write_at_all_end wrapper function 
******************************************************/
void MPI_FILE_WRITE_AT_ALL_END( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Status local_status;

  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_write_at_all_end( local_fh, buf, &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_File_write_at_all_end wrapper function 
******************************************************/
void mpi_file_write_at_all_end( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_ALL_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_at_all_end wrapper function 
******************************************************/
void mpi_file_write_at_all_end_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_ALL_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_at_all_end wrapper function 
******************************************************/
void mpi_file_write_at_all_end__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_AT_ALL_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_write_ordered wrapper function 
******************************************************/
int MPI_File_write_ordered( MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_ordered()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_ordered( fh, buf, count, datatype, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_write_ordered wrapper function 
******************************************************/
void MPI_FILE_WRITE_ORDERED( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Status local_status;

  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_write_ordered( local_fh, buf, *count, MPI_Type_f2c(*datatype), &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_File_write_ordered wrapper function 
******************************************************/
void mpi_file_write_ordered( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ORDERED( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_ordered wrapper function 
******************************************************/
void mpi_file_write_ordered_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ORDERED( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_ordered wrapper function 
******************************************************/
void mpi_file_write_ordered__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ORDERED( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_write_ordered_begin wrapper function 
******************************************************/
int MPI_File_write_ordered_begin( MPI_File fh, void * buf, int count, MPI_Datatype datatype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_ordered_begin()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_ordered_begin( fh, buf, count, datatype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_write_ordered_begin wrapper function 
******************************************************/
void MPI_FILE_WRITE_ORDERED_BEGIN( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  *ierr = MPI_File_write_ordered_begin( MPI_File_f2c(*fh), buf, *count, MPI_Type_f2c(*datatype)) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_ordered_begin wrapper function 
******************************************************/
void mpi_file_write_ordered_begin( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ORDERED_BEGIN( fh, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_ordered_begin wrapper function 
******************************************************/
void mpi_file_write_ordered_begin_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ORDERED_BEGIN( fh, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_ordered_begin wrapper function 
******************************************************/
void mpi_file_write_ordered_begin__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ORDERED_BEGIN( fh, buf, count, datatype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_write_ordered_end wrapper function 
******************************************************/
int MPI_File_write_ordered_end( MPI_File fh, void * buf, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_ordered_end()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_ordered_end( fh, buf, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_write_ordered_end wrapper function 
******************************************************/
void MPI_FILE_WRITE_ORDERED_END( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Status local_status;

  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_write_ordered_end( local_fh, buf, &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_File_write_ordered_end wrapper function 
******************************************************/
void mpi_file_write_ordered_end( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ORDERED_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_ordered_end wrapper function 
******************************************************/
void mpi_file_write_ordered_end_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ORDERED_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_ordered_end wrapper function 
******************************************************/
void mpi_file_write_ordered_end__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_ORDERED_END( fh, buf, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_File_write_shared wrapper function 
******************************************************/
int MPI_File_write_shared( MPI_File fh, void * buf, int count, MPI_Datatype datatype, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_write_shared()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_write_shared( fh, buf, count, datatype, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_write_shared wrapper function 
******************************************************/
void MPI_FILE_WRITE_SHARED( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_File local_fh;
  MPI_Status local_status;

  local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_write_shared( local_fh, buf, *count, MPI_Type_f2c(*datatype), &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_File_write_shared wrapper function 
******************************************************/
void mpi_file_write_shared( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_SHARED( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_shared wrapper function 
******************************************************/
void mpi_file_write_shared_( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_SHARED( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_write_shared wrapper function 
******************************************************/
void mpi_file_write_shared__( MPI_Fint *  fh, MPI_Aint * buf, MPI_Fint *  count, MPI_Fint *  datatype, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_FILE_WRITE_SHARED( fh, buf, count, datatype, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#ifdef TAU_MPIDATAREP

/******************************************************
***      MPI_Register_datarep wrapper function 
******************************************************/
int MPI_Register_datarep( char * datarep, MPI_Datarep_conversion_function * read_conversion_fn, MPI_Datarep_conversion_function * write_conversion_fn, MPI_Datarep_extent_function * dtype_file_extent_fn, void * extra_state)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Register_datarep()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Register_datarep( datarep, read_conversion_fn, write_conversion_fn, dtype_file_extent_fn, extra_state) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Register_datarep wrapper function 
******************************************************/
void MPI_REGISTER_DATAREP( char * datarep, MPI_Datarep_conversion_function * read_conversion_fn, MPI_Datarep_conversion_function * write_conversion_fn, MPI_Datarep_extent_function * dtype_file_extent_fn, MPI_Aint * extra_state, MPI_Fint * ierr)
{
  *ierr = MPI_Register_datarep( datarep, read_conversion_fn, write_conversion_fn, dtype_file_extent_fn, extra_state) ; 
  return ; 
}

/******************************************************
***      MPI_Register_datarep wrapper function 
******************************************************/
void mpi_register_datarep( char * datarep, MPI_Datarep_conversion_function * read_conversion_fn, MPI_Datarep_conversion_function * write_conversion_fn, MPI_Datarep_extent_function * dtype_file_extent_fn, MPI_Aint * extra_state, MPI_Fint * ierr)
{
  MPI_REGISTER_DATAREP( datarep, read_conversion_fn, write_conversion_fn, dtype_file_extent_fn, extra_state, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Register_datarep wrapper function 
******************************************************/
void mpi_register_datarep_( char * datarep, MPI_Datarep_conversion_function * read_conversion_fn, MPI_Datarep_conversion_function * write_conversion_fn, MPI_Datarep_extent_function * dtype_file_extent_fn, MPI_Aint * extra_state, MPI_Fint * ierr)
{
  MPI_REGISTER_DATAREP( datarep, read_conversion_fn, write_conversion_fn, dtype_file_extent_fn, extra_state, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Register_datarep wrapper function 
******************************************************/
void mpi_register_datarep__( char * datarep, MPI_Datarep_conversion_function * read_conversion_fn, MPI_Datarep_conversion_function * write_conversion_fn, MPI_Datarep_extent_function * dtype_file_extent_fn, MPI_Aint * extra_state, MPI_Fint * ierr)
{
  MPI_REGISTER_DATAREP( datarep, read_conversion_fn, write_conversion_fn, dtype_file_extent_fn, extra_state, ierr) ; 
  return ; 
}

#endif /* TAU_MPIDATAREP */

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Info_create wrapper function 
******************************************************/
int MPI_Info_create( MPI_Info * info)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Info_create()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Info_create( info) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Info_create wrapper function 
******************************************************/
void MPI_INFO_CREATE( MPI_Aint * info, MPI_Fint * ierr)
{
  MPI_Info local_info;
  *ierr = MPI_Info_create( &local_info) ; 
  *info = MPI_Info_c2f(local_info);
  return ; 
}

/******************************************************
***      MPI_Info_create wrapper function 
******************************************************/
void mpi_info_create( MPI_Aint * info, MPI_Fint * ierr)
{
  MPI_INFO_CREATE( info, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_create wrapper function 
******************************************************/
void mpi_info_create_( MPI_Aint * info, MPI_Fint * ierr)
{
  MPI_INFO_CREATE( info, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_create wrapper function 
******************************************************/
void mpi_info_create__( MPI_Aint * info, MPI_Fint * ierr)
{
  MPI_INFO_CREATE( info, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Info_set wrapper function 
******************************************************/
int MPI_Info_set( MPI_Info Info, TAU_CONST char * key, TAU_CONST char * value)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Info_set()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Info_set( Info, key, value) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Info_set wrapper function 
******************************************************/
void MPI_INFO_SET( MPI_Fint *  Info, TAU_CONST char * key, TAU_CONST char * value, MPI_Fint * ierr)
{
  MPI_Info local_info = MPI_Info_f2c(*Info);
  *ierr = MPI_Info_set( local_info, key, value) ; 
  return ; 
}

/******************************************************
***      MPI_Info_set wrapper function 
******************************************************/
void mpi_info_set( MPI_Fint *  Info, TAU_CONST char * key, TAU_CONST char * value, MPI_Fint * ierr)
{
  MPI_INFO_SET( Info, key, value, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_set wrapper function 
******************************************************/
void mpi_info_set_( MPI_Fint *  Info, TAU_CONST char * key, TAU_CONST char * value, MPI_Fint * ierr)
{
  MPI_INFO_SET( Info, key, value, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_set wrapper function 
******************************************************/
void mpi_info_set__( MPI_Fint *  Info, TAU_CONST char * key, TAU_CONST char * value, MPI_Fint * ierr)
{
  MPI_INFO_SET( Info, key, value, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Info_delete wrapper function 
******************************************************/
int MPI_Info_delete( MPI_Info info, TAU_CONST char * key)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Info_delete()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Info_delete( info, key) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Info_delete wrapper function 
******************************************************/
void MPI_INFO_DELETE( MPI_Fint *  info, TAU_CONST char * key, MPI_Fint * ierr)
{
  MPI_Info local_info = MPI_Info_f2c(*info);
  *ierr = MPI_Info_delete( local_info, key) ; 
  return ; 
}

/******************************************************
***      MPI_Info_delete wrapper function 
******************************************************/
void mpi_info_delete( MPI_Fint *  info, TAU_CONST char * key, MPI_Fint * ierr)
{
  MPI_INFO_DELETE( info, key, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_delete wrapper function 
******************************************************/
void mpi_info_delete_( MPI_Fint *  info, TAU_CONST char * key, MPI_Fint * ierr)
{
  MPI_INFO_DELETE( info, key, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_delete wrapper function 
******************************************************/
void mpi_info_delete__( MPI_Fint *  info, TAU_CONST char * key, MPI_Fint * ierr)
{
  MPI_INFO_DELETE( info, key, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Info_get wrapper function 
******************************************************/
int MPI_Info_get( MPI_Info info, TAU_CONST char * key, int valuelen, char * value, int * flag)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Info_get()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Info_get( info, key, valuelen, value, flag) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Info_get wrapper function 
******************************************************/
void MPI_INFO_GET( MPI_Fint *  info, TAU_CONST char * key, MPI_Fint *  valuelen, TAU_CONST char * value, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_Info local_info = MPI_Info_f2c(*info);
  *ierr = MPI_Info_get( local_info, key, *valuelen, value, flag) ; 
  return ; 
}

/******************************************************
***      MPI_Info_get wrapper function 
******************************************************/
void mpi_info_get( MPI_Fint *  info, TAU_CONST char * key, MPI_Fint *  valuelen, TAU_CONST char * value, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_INFO_GET( info, key, valuelen, value, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_get wrapper function 
******************************************************/
void mpi_info_get_( MPI_Fint *  info, TAU_CONST char * key, MPI_Fint *  valuelen, TAU_CONST char * value, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_INFO_GET( info, key, valuelen, value, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_get wrapper function 
******************************************************/
void mpi_info_get__( MPI_Fint *  info, TAU_CONST char * key, MPI_Fint *  valuelen, TAU_CONST char * value, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_INFO_GET( info, key, valuelen, value, flag, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Info_get_valuelen wrapper function 
******************************************************/
int MPI_Info_get_valuelen( MPI_Info info, TAU_CONST char * key, int * valuelen, int * flag)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Info_get_valuelen()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Info_get_valuelen( info, key, valuelen, flag) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Info_get_valuelen wrapper function 
******************************************************/
void MPI_INFO_GET_VALUELEN( MPI_Fint *  info, TAU_CONST char * key, MPI_Fint *  valuelen, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_Info local_info = MPI_Info_f2c(*info);
  *ierr = MPI_Info_get_valuelen( local_info, key, valuelen, flag) ; 
  return ; 
}

/******************************************************
***      MPI_Info_get_valuelen wrapper function 
******************************************************/
void mpi_info_get_valuelen( MPI_Fint *  info, TAU_CONST char * key, MPI_Fint *  valuelen, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_INFO_GET_VALUELEN( info, key, valuelen, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_get_valuelen wrapper function 
******************************************************/
void mpi_info_get_valuelen_( MPI_Fint *  info, TAU_CONST char * key, MPI_Fint *  valuelen, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_INFO_GET_VALUELEN( info, key, valuelen, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_get_valuelen wrapper function 
******************************************************/
void mpi_info_get_valuelen__( MPI_Fint *  info, TAU_CONST char * key, MPI_Fint *  valuelen, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_INFO_GET_VALUELEN( info, key, valuelen, flag, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Info_get_nkeys wrapper function 
******************************************************/
int MPI_Info_get_nkeys( MPI_Info info, int * nkeys)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Info_get_nkeys()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Info_get_nkeys( info, nkeys) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Info_get_nkeys wrapper function 
******************************************************/
void MPI_INFO_GET_NKEYS( MPI_Fint *  info, MPI_Fint *  nkeys, MPI_Fint * ierr)
{
  MPI_Info local_info = MPI_Info_f2c(*info);
  *ierr = MPI_Info_get_nkeys( local_info, nkeys) ; 
  return ; 
}

/******************************************************
***      MPI_Info_get_nkeys wrapper function 
******************************************************/
void mpi_info_get_nkeys( MPI_Fint *  info, MPI_Fint *  nkeys, MPI_Fint * ierr)
{
  MPI_INFO_GET_NKEYS( info, nkeys, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_get_nkeys wrapper function 
******************************************************/
void mpi_info_get_nkeys_( MPI_Fint *  info, MPI_Fint *  nkeys, MPI_Fint * ierr)
{
  MPI_INFO_GET_NKEYS( info, nkeys, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_get_nkeys wrapper function 
******************************************************/
void mpi_info_get_nkeys__( MPI_Fint *  info, MPI_Fint *  nkeys, MPI_Fint * ierr)
{
  MPI_INFO_GET_NKEYS( info, nkeys, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Info_get_nthkey wrapper function 
******************************************************/
int MPI_Info_get_nthkey( MPI_Info info, int n, char * key)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Info_get_nthkey()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Info_get_nthkey( info, n, key) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Info_get_nthkey wrapper function 
******************************************************/
void MPI_INFO_GET_NTHKEY( MPI_Fint *  info, MPI_Fint *  n, char * key, MPI_Fint * ierr)
{
  MPI_Info local_info = MPI_Info_f2c(*info);
  *ierr = MPI_Info_get_nthkey( local_info, *n, key) ; 
  return ; 
}

/******************************************************
***      MPI_Info_get_nthkey wrapper function 
******************************************************/
void mpi_info_get_nthkey( MPI_Fint *  info, MPI_Fint *  n, char * key, MPI_Fint * ierr)
{
  MPI_INFO_GET_NTHKEY( info, n, key, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_get_nthkey wrapper function 
******************************************************/
void mpi_info_get_nthkey_( MPI_Fint *  info, MPI_Fint *  n, char * key, MPI_Fint * ierr)
{
  MPI_INFO_GET_NTHKEY( info, n, key, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_get_nthkey wrapper function 
******************************************************/
void mpi_info_get_nthkey__( MPI_Fint *  info, MPI_Fint *  n, char * key, MPI_Fint * ierr)
{
  MPI_INFO_GET_NTHKEY( info, n, key, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Info_dup wrapper function 
******************************************************/
int MPI_Info_dup( MPI_Info info, MPI_Info * newinfo)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Info_dup()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Info_dup( info, newinfo) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Info_dup wrapper function 
******************************************************/
void MPI_INFO_DUP( MPI_Fint *  info, MPI_Fint * newinfo, MPI_Fint * ierr)
{
  MPI_Info local_newinfo;
  MPI_Info local_info;
  local_info = MPI_Info_f2c(*info);
  
  *ierr = MPI_Info_dup( local_info, &local_newinfo) ; 
  *newinfo = MPI_Info_c2f(local_newinfo);
  return ; 
}

/******************************************************
***      MPI_Info_dup wrapper function 
******************************************************/
void mpi_info_dup( MPI_Fint *  info, MPI_Fint * newinfo, MPI_Fint * ierr)
{
  MPI_INFO_DUP( info, newinfo, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_dup wrapper function 
******************************************************/
void mpi_info_dup_( MPI_Fint *  info, MPI_Fint * newinfo, MPI_Fint * ierr)
{
  MPI_INFO_DUP( info, newinfo, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_dup wrapper function 
******************************************************/
void mpi_info_dup__( MPI_Fint *  info, MPI_Fint * newinfo, MPI_Fint * ierr)
{
  MPI_INFO_DUP( info, newinfo, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Info_free wrapper function 
******************************************************/
int MPI_Info_free( MPI_Info * info)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Info_free()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Info_free( info) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Info_free wrapper function 
******************************************************/
void MPI_INFO_FREE( MPI_Aint * info, MPI_Fint * ierr)
{
  MPI_Info local_info = MPI_Info_f2c(*info);
  *ierr = MPI_Info_free( &local_info) ; 
  *info = MPI_Info_c2f(local_info);
  return ; 
}

/******************************************************
***      MPI_Info_free wrapper function 
******************************************************/
void mpi_info_free( MPI_Aint * info, MPI_Fint * ierr)
{
  MPI_INFO_FREE( info, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_free wrapper function 
******************************************************/
void mpi_info_free_( MPI_Aint * info, MPI_Fint * ierr)
{
  MPI_INFO_FREE( info, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Info_free wrapper function 
******************************************************/
void mpi_info_free__( MPI_Aint * info, MPI_Fint * ierr)
{
  MPI_INFO_FREE( info, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#ifdef TAU_MPIADDERROR

/******************************************************
***      MPI_Add_error_class wrapper function 
******************************************************/
int MPI_Add_error_class( int * errorclass)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Add_error_class()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Add_error_class( errorclass) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Add_error_class wrapper function 
******************************************************/
void MPI_ADD_ERROR_CLASS( MPI_Fint *  errorclass, MPI_Fint * ierr)
{
  *ierr = MPI_Add_error_class( errorclass) ; 
  return ; 
}

/******************************************************
***      MPI_Add_error_class wrapper function 
******************************************************/
void mpi_add_error_class( MPI_Fint *  errorclass, MPI_Fint * ierr)
{
  MPI_ADD_ERROR_CLASS( errorclass, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Add_error_class wrapper function 
******************************************************/
void mpi_add_error_class_( MPI_Fint *  errorclass, MPI_Fint * ierr)
{
  MPI_ADD_ERROR_CLASS( errorclass, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Add_error_class wrapper function 
******************************************************/
void mpi_add_error_class__( MPI_Fint *  errorclass, MPI_Fint * ierr)
{
  MPI_ADD_ERROR_CLASS( errorclass, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Add_error_code wrapper function 
******************************************************/
int MPI_Add_error_code( int errorclass, int * errorcode)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Add_error_code()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Add_error_code( errorclass, errorcode) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Add_error_code wrapper function 
******************************************************/
void MPI_ADD_ERROR_CODE( MPI_Fint *  errorclass, MPI_Fint *  errorcode, MPI_Fint * ierr)
{
  *ierr = MPI_Add_error_code( *errorclass, errorcode) ; 
  return ; 
}

/******************************************************
***      MPI_Add_error_code wrapper function 
******************************************************/
void mpi_add_error_code( MPI_Fint *  errorclass, MPI_Fint *  errorcode, MPI_Fint * ierr)
{
  MPI_ADD_ERROR_CODE( errorclass, errorcode, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Add_error_code wrapper function 
******************************************************/
void mpi_add_error_code_( MPI_Fint *  errorclass, MPI_Fint *  errorcode, MPI_Fint * ierr)
{
  MPI_ADD_ERROR_CODE( errorclass, errorcode, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Add_error_code wrapper function 
******************************************************/
void mpi_add_error_code__( MPI_Fint *  errorclass, MPI_Fint *  errorcode, MPI_Fint * ierr)
{
  MPI_ADD_ERROR_CODE( errorclass, errorcode, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Add_error_string wrapper function 
******************************************************/
int MPI_Add_error_string( int errorcode, char * string)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Add_error_string()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Add_error_string( errorcode, string) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Add_error_string wrapper function 
******************************************************/
void MPI_ADD_ERROR_STRING( MPI_Fint *  errorcode, char * string, MPI_Fint * ierr)
{
  *ierr = MPI_Add_error_string( *errorcode, string) ; 
  return ; 
}

/******************************************************
***      MPI_Add_error_string wrapper function 
******************************************************/
void mpi_add_error_string( MPI_Fint *  errorcode, char * string, MPI_Fint * ierr)
{
  MPI_ADD_ERROR_STRING( errorcode, string, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Add_error_string wrapper function 
******************************************************/
void mpi_add_error_string_( MPI_Fint *  errorcode, char * string, MPI_Fint * ierr)
{
  MPI_ADD_ERROR_STRING( errorcode, string, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Add_error_string wrapper function 
******************************************************/
void mpi_add_error_string__( MPI_Fint *  errorcode, char * string, MPI_Fint * ierr)
{
  MPI_ADD_ERROR_STRING( errorcode, string, ierr) ; 
  return ; 
}

#endif /* TAU_MPIADDERROR */

/******************************************************/
/******************************************************/


#ifdef TAU_MPIERRHANDLER

/******************************************************
***      MPI_Comm_call_errhandler wrapper function 
******************************************************/
int MPI_Comm_call_errhandler( MPI_Comm comm, int errorcode)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_call_errhandler()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_call_errhandler( comm, errorcode) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Comm_call_errhandler wrapper function 
******************************************************/
void MPI_COMM_CALL_ERRHANDLER( MPI_Fint *  comm, MPI_Fint *  errorcode, MPI_Fint * ierr)
{
  *ierr = MPI_Comm_call_errhandler( MPI_Comm_f2c(*comm), *errorcode) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_call_errhandler wrapper function 
******************************************************/
void mpi_comm_call_errhandler( MPI_Fint *  comm, MPI_Fint *  errorcode, MPI_Fint * ierr)
{
  MPI_COMM_CALL_ERRHANDLER( comm, errorcode, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_call_errhandler wrapper function 
******************************************************/
void mpi_comm_call_errhandler_( MPI_Fint *  comm, MPI_Fint *  errorcode, MPI_Fint * ierr)
{
  MPI_COMM_CALL_ERRHANDLER( comm, errorcode, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_call_errhandler wrapper function 
******************************************************/
void mpi_comm_call_errhandler__( MPI_Fint *  comm, MPI_Fint *  errorcode, MPI_Fint * ierr)
{
  MPI_COMM_CALL_ERRHANDLER( comm, errorcode, ierr) ; 
  return ; 
}

#endif /* TAU_MPIERRHANDLER */

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Comm_set_name wrapper function 
******************************************************/
int MPI_Comm_set_name( MPI_Comm comm, char * comm_name)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_set_name()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_set_name( comm, comm_name) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Comm_set_name wrapper function 
******************************************************/
void MPI_COMM_SET_NAME( MPI_Fint *  comm, char * comm_name, MPI_Fint * ierr)
{
  *ierr = MPI_Comm_set_name( MPI_Comm_f2c(*comm), comm_name) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_set_name wrapper function 
******************************************************/
void mpi_comm_set_name( MPI_Fint *  comm, char * comm_name, MPI_Fint * ierr)
{
  MPI_COMM_SET_NAME( comm, comm_name, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_set_name wrapper function 
******************************************************/
void mpi_comm_set_name_( MPI_Fint *  comm, char * comm_name, MPI_Fint * ierr)
{
  MPI_COMM_SET_NAME( comm, comm_name, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_set_name wrapper function 
******************************************************/
void mpi_comm_set_name__( MPI_Fint *  comm, char * comm_name, MPI_Fint * ierr)
{
  MPI_COMM_SET_NAME( comm, comm_name, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Comm_get_name wrapper function 
******************************************************/
int MPI_Comm_get_name( MPI_Comm comm, char * comm_name, int * resultlen)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Comm_get_name()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Comm_get_name( comm, comm_name, resultlen) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Comm_get_name wrapper function 
******************************************************/
void MPI_COMM_GET_NAME( MPI_Fint *  comm, char * comm_name, MPI_Fint *  resultlen, MPI_Fint * ierr)
{
  *ierr = MPI_Comm_get_name( MPI_Comm_f2c(*comm), comm_name, resultlen) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_get_name wrapper function 
******************************************************/
void mpi_comm_get_name( MPI_Fint *  comm, char * comm_name, MPI_Fint *  resultlen, MPI_Fint * ierr)
{
  MPI_COMM_GET_NAME( comm, comm_name, resultlen, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_get_name wrapper function 
******************************************************/
void mpi_comm_get_name_( MPI_Fint *  comm, char * comm_name, MPI_Fint *  resultlen, MPI_Fint * ierr)
{
  MPI_COMM_GET_NAME( comm, comm_name, resultlen, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Comm_get_name wrapper function 
******************************************************/
void mpi_comm_get_name__( MPI_Fint *  comm, char * comm_name, MPI_Fint *  resultlen, MPI_Fint * ierr)
{
  MPI_COMM_GET_NAME( comm, comm_name, resultlen, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#ifdef TAU_MPIERRHANDLER

/******************************************************
***      MPI_File_call_errhandler wrapper function 
******************************************************/
int MPI_File_call_errhandler( MPI_File fh, int errorcode)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_File_call_errhandler()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_File_call_errhandler( fh, errorcode) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_File_call_errhandler wrapper function 
******************************************************/
void MPI_FILE_CALL_ERRHANDLER( MPI_Fint *  fh, MPI_Fint *  errorcode, MPI_Fint * ierr)
{
  MPI_File local_fh = MPI_File_f2c(*fh);
  *ierr = MPI_File_call_errhandler( local_fh, *errorcode) ; 
  return ; 
}

/******************************************************
***      MPI_File_call_errhandler wrapper function 
******************************************************/
void mpi_file_call_errhandler( MPI_Fint *  fh, MPI_Fint *  errorcode, MPI_Fint * ierr)
{
  MPI_FILE_CALL_ERRHANDLER( fh, errorcode, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_call_errhandler wrapper function 
******************************************************/
void mpi_file_call_errhandler_( MPI_Fint *  fh, MPI_Fint *  errorcode, MPI_Fint * ierr)
{
  MPI_FILE_CALL_ERRHANDLER( fh, errorcode, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_File_call_errhandler wrapper function 
******************************************************/
void mpi_file_call_errhandler__( MPI_Fint *  fh, MPI_Fint *  errorcode, MPI_Fint * ierr)
{
  MPI_FILE_CALL_ERRHANDLER( fh, errorcode, ierr) ; 
  return ; 
}

#endif /* TAU_MPIERRHANDLER */

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Type_set_name wrapper function 
******************************************************/
int MPI_Type_set_name( MPI_Datatype type, char * type_name)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_set_name()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_set_name( type, type_name) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_set_name wrapper function 
******************************************************/
void MPI_TYPE_SET_NAME( MPI_Fint *  type, char * type_name, MPI_Fint * ierr)
{
  MPI_Datatype local_type = MPI_Type_f2c(*type);
  *ierr = MPI_Type_set_name( local_type, type_name) ; 
  return ; 
}

/******************************************************
***      MPI_Type_set_name wrapper function 
******************************************************/
void mpi_type_set_name( MPI_Fint *  type, char * type_name, MPI_Fint * ierr)
{
  MPI_TYPE_SET_NAME( type, type_name, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_set_name wrapper function 
******************************************************/
void mpi_type_set_name_( MPI_Fint *  type, char * type_name, MPI_Fint * ierr)
{
  MPI_TYPE_SET_NAME( type, type_name, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_set_name wrapper function 
******************************************************/
void mpi_type_set_name__( MPI_Fint *  type, char * type_name, MPI_Fint * ierr)
{
  MPI_TYPE_SET_NAME( type, type_name, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Type_get_name wrapper function 
******************************************************/
int MPI_Type_get_name( MPI_Datatype type, char * type_name, int * resultlen)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_get_name()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_get_name( type, type_name, resultlen) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_get_name wrapper function 
******************************************************/
void MPI_TYPE_GET_NAME( MPI_Fint *  type, char * type_name, MPI_Fint *  resultlen, MPI_Fint * ierr)
{
  MPI_Datatype local_type = MPI_Type_f2c(*type);
  *ierr = MPI_Type_get_name( local_type, type_name, resultlen) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_name wrapper function 
******************************************************/
void mpi_type_get_name( MPI_Fint *  type, char * type_name, MPI_Fint *  resultlen, MPI_Fint * ierr)
{
  MPI_TYPE_GET_NAME( type, type_name, resultlen, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_name wrapper function 
******************************************************/
void mpi_type_get_name_( MPI_Fint *  type, char * type_name, MPI_Fint *  resultlen, MPI_Fint * ierr)
{
  MPI_TYPE_GET_NAME( type, type_name, resultlen, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_name wrapper function 
******************************************************/
void mpi_type_get_name__( MPI_Fint *  type, char * type_name, MPI_Fint *  resultlen, MPI_Fint * ierr)
{
  MPI_TYPE_GET_NAME( type, type_name, resultlen, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#ifdef TAU_MPIERRHANDLER

/******************************************************
***      MPI_Win_call_errhandler wrapper function 
******************************************************/
int MPI_Win_call_errhandler( MPI_Win win, int errorcode)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_call_errhandler()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_call_errhandler( win, errorcode) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_call_errhandler wrapper function 
******************************************************/
void MPI_WIN_CALL_ERRHANDLER( MPI_Fint *  win, MPI_Fint *  errorcode, MPI_Fint * ierr)
{
  MPI_Win local_win = MPI_Win_f2c(*win);
  *ierr = MPI_Win_call_errhandler( local_win, *errorcode) ; 
  return ; 
}

/******************************************************
***      MPI_Win_call_errhandler wrapper function 
******************************************************/
void mpi_win_call_errhandler( MPI_Fint *  win, MPI_Fint *  errorcode, MPI_Fint * ierr)
{
  MPI_WIN_CALL_ERRHANDLER( win, errorcode, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_call_errhandler wrapper function 
******************************************************/
void mpi_win_call_errhandler_( MPI_Fint *  win, MPI_Fint *  errorcode, MPI_Fint * ierr)
{
  MPI_WIN_CALL_ERRHANDLER( win, errorcode, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_call_errhandler wrapper function 
******************************************************/
void mpi_win_call_errhandler__( MPI_Fint *  win, MPI_Fint *  errorcode, MPI_Fint * ierr)
{
  MPI_WIN_CALL_ERRHANDLER( win, errorcode, ierr) ; 
  return ; 
}

#endif /* TAU_MPIERRHANDLER */

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_set_name wrapper function 
******************************************************/
int MPI_Win_set_name( MPI_Win win, char * win_name)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_set_name()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_set_name( win, win_name) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_set_name wrapper function 
******************************************************/
void MPI_WIN_SET_NAME( MPI_Fint *  win, char * win_name, MPI_Fint * ierr)
{
  MPI_Win local_win = MPI_Win_f2c(*win);
  *ierr = MPI_Win_set_name( local_win, win_name) ; 
  return ; 
}

/******************************************************
***      MPI_Win_set_name wrapper function 
******************************************************/
void mpi_win_set_name( MPI_Fint *  win, char * win_name, MPI_Fint * ierr)
{
  MPI_WIN_SET_NAME( win, win_name, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_set_name wrapper function 
******************************************************/
void mpi_win_set_name_( MPI_Fint *  win, char * win_name, MPI_Fint * ierr)
{
  MPI_WIN_SET_NAME( win, win_name, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_set_name wrapper function 
******************************************************/
void mpi_win_set_name__( MPI_Fint *  win, char * win_name, MPI_Fint * ierr)
{
  MPI_WIN_SET_NAME( win, win_name, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Win_get_name wrapper function 
******************************************************/
int MPI_Win_get_name( MPI_Win win, char * win_name, int * resultlen)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Win_get_name()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Win_get_name( win, win_name, resultlen) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Win_get_name wrapper function 
******************************************************/
void MPI_WIN_GET_NAME( MPI_Fint *  win, char * win_name, MPI_Fint *  resultlen, MPI_Fint * ierr)
{
  MPI_Win local_win = MPI_Win_f2c(*win);
  *ierr = MPI_Win_get_name( local_win, win_name, resultlen) ; 
  return ; 
}

/******************************************************
***      MPI_Win_get_name wrapper function 
******************************************************/
void mpi_win_get_name( MPI_Fint *  win, char * win_name, MPI_Fint *  resultlen, MPI_Fint * ierr)
{
  MPI_WIN_GET_NAME( win, win_name, resultlen, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_get_name wrapper function 
******************************************************/
void mpi_win_get_name_( MPI_Fint *  win, char * win_name, MPI_Fint *  resultlen, MPI_Fint * ierr)
{
  MPI_WIN_GET_NAME( win, win_name, resultlen, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Win_get_name wrapper function 
******************************************************/
void mpi_win_get_name__( MPI_Fint *  win, char * win_name, MPI_Fint *  resultlen, MPI_Fint * ierr)
{
  MPI_WIN_GET_NAME( win, win_name, resultlen, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


#ifdef TAU_MPI_INIT_THREAD_WRAPPER
/******************************************************
***      MPI_Init_thread wrapper function 
******************************************************/
int MPI_Init_thread( int * argc, char *** argv, int required, int * provided)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Init_thread()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Init_thread( argc, argv, required, provided) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Init_thread wrapper function 
******************************************************/
void MPI_INIT_THREAD( MPI_Fint *  argc, char *** argv, MPI_Fint *  required, MPI_Fint *  provided, MPI_Fint * ierr)
{
  *ierr = MPI_Init_thread( argc, argv, *required, provided) ; 
  return ; 
}

/******************************************************
***      MPI_Init_thread wrapper function 
******************************************************/
void mpi_init_thread( MPI_Fint *  argc, char *** argv, MPI_Fint *  required, MPI_Fint *  provided, MPI_Fint * ierr)
{
  MPI_INIT_THREAD( argc, argv, required, provided, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Init_thread wrapper function 
******************************************************/
void mpi_init_thread_( MPI_Fint *  argc, char *** argv, MPI_Fint *  required, MPI_Fint *  provided, MPI_Fint * ierr)
{
  MPI_INIT_THREAD( argc, argv, required, provided, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Init_thread wrapper function 
******************************************************/
void mpi_init_thread__( MPI_Fint *  argc, char *** argv, MPI_Fint *  required, MPI_Fint *  provided, MPI_Fint * ierr)
{
  MPI_INIT_THREAD( argc, argv, required, provided, ierr) ; 
  return ; 
}

#endif /* TAU_MPI_INIT_THREAD_WRAPPER */

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Query_thread wrapper function 
******************************************************/
int MPI_Query_thread( int * provided)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Query_thread()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Query_thread( provided) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Query_thread wrapper function 
******************************************************/
void MPI_QUERY_THREAD( MPI_Fint *  provided, MPI_Fint * ierr)
{
  *ierr = MPI_Query_thread( provided) ; 
  return ; 
}

/******************************************************
***      MPI_Query_thread wrapper function 
******************************************************/
void mpi_query_thread( MPI_Fint *  provided, MPI_Fint * ierr)
{
  MPI_QUERY_THREAD( provided, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Query_thread wrapper function 
******************************************************/
void mpi_query_thread_( MPI_Fint *  provided, MPI_Fint * ierr)
{
  MPI_QUERY_THREAD( provided, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Query_thread wrapper function 
******************************************************/
void mpi_query_thread__( MPI_Fint *  provided, MPI_Fint * ierr)
{
  MPI_QUERY_THREAD( provided, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Is_thread_main wrapper function 
******************************************************/
int MPI_Is_thread_main( int * flag)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Is_thread_main()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Is_thread_main( flag) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Is_thread_main wrapper function 
******************************************************/
void MPI_IS_THREAD_MAIN( MPI_Fint *  flag, MPI_Fint * ierr)
{
  *ierr = MPI_Is_thread_main( flag) ; 
  return ; 
}

/******************************************************
***      MPI_Is_thread_main wrapper function 
******************************************************/
void mpi_is_thread_main( MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_IS_THREAD_MAIN( flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Is_thread_main wrapper function 
******************************************************/
void mpi_is_thread_main_( MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_IS_THREAD_MAIN( flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Is_thread_main wrapper function 
******************************************************/
void mpi_is_thread_main__( MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_IS_THREAD_MAIN( flag, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#ifdef TAU_MPIGREQUEST

/******************************************************
***      MPI_Grequest_start wrapper function 
******************************************************/
int MPI_Grequest_start( MPI_Grequest_query_function * grequest_query_fn, MPI_Grequest_free_function * grequest_free_fn, MPI_Grequest_cancel_function * grequest_cancel_fn, void * extra_state, MPI_Request * request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Grequest_start()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Grequest_start( grequest_query_fn, grequest_free_fn, grequest_cancel_fn, extra_state, request) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Grequest_start wrapper function 
******************************************************/
void MPI_GREQUEST_START( MPI_Grequest_query_function * grequest_query_fn, MPI_Grequest_free_function * grequest_free_fn, MPI_Grequest_cancel_function * grequest_cancel_fn, MPI_Aint * extra_state, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_Request local_request;
  *ierr = MPI_Grequest_start( grequest_query_fn, grequest_free_fn, grequest_cancel_fn, extra_state, &local_request) ; 
  *request = MPI_Request_c2f(local_request);
  return ; 
}

/******************************************************
***      MPI_Grequest_start wrapper function 
******************************************************/
void mpi_grequest_start( MPI_Grequest_query_function * grequest_query_fn, MPI_Grequest_free_function * grequest_free_fn, MPI_Grequest_cancel_function * grequest_cancel_fn, MPI_Aint * extra_state, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_GREQUEST_START( grequest_query_fn, grequest_free_fn, grequest_cancel_fn, extra_state, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Grequest_start wrapper function 
******************************************************/
void mpi_grequest_start_( MPI_Grequest_query_function * grequest_query_fn, MPI_Grequest_free_function * grequest_free_fn, MPI_Grequest_cancel_function * grequest_cancel_fn, MPI_Aint * extra_state, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_GREQUEST_START( grequest_query_fn, grequest_free_fn, grequest_cancel_fn, extra_state, request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Grequest_start wrapper function 
******************************************************/
void mpi_grequest_start__( MPI_Grequest_query_function * grequest_query_fn, MPI_Grequest_free_function * grequest_free_fn, MPI_Grequest_cancel_function * grequest_cancel_fn, MPI_Aint * extra_state, MPI_Aint * request, MPI_Fint * ierr)
{
  MPI_GREQUEST_START( grequest_query_fn, grequest_free_fn, grequest_cancel_fn, extra_state, request, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Grequest_complete wrapper function 
******************************************************/
int MPI_Grequest_complete( MPI_Request request)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Grequest_complete()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Grequest_complete( request) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Grequest_complete wrapper function 
******************************************************/
void MPI_GREQUEST_COMPLETE( MPI_Fint *  request, MPI_Fint * ierr)
{
  *ierr = MPI_Grequest_complete( MPI_Request_f2c(*request)) ; 
  return ; 
}

/******************************************************
***      MPI_Grequest_complete wrapper function 
******************************************************/
void mpi_grequest_complete( MPI_Fint *  request, MPI_Fint * ierr)
{
  MPI_GREQUEST_COMPLETE( request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Grequest_complete wrapper function 
******************************************************/
void mpi_grequest_complete_( MPI_Fint *  request, MPI_Fint * ierr)
{
  MPI_GREQUEST_COMPLETE( request, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Grequest_complete wrapper function 
******************************************************/
void mpi_grequest_complete__( MPI_Fint *  request, MPI_Fint * ierr)
{
  MPI_GREQUEST_COMPLETE( request, ierr) ; 
  return ; 
}

#endif /* TAU_MPIGREQUEST */

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Status_set_elements wrapper function 
******************************************************/
int MPI_Status_set_elements( MPI_Status * status, MPI_Datatype datatype, int count)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Status_set_elements()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Status_set_elements( status, datatype, count) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Status_set_elements wrapper function 
******************************************************/
void MPI_STATUS_SET_ELEMENTS( MPI_Fint * status, MPI_Fint *  datatype, MPI_Fint *  count, MPI_Fint * ierr)
{
  MPI_Status local_status; 
  MPI_Status_f2c(status, &local_status);
  
  *ierr = MPI_Status_set_elements( &local_status, MPI_Type_f2c(*datatype), *count) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_Status_set_elements wrapper function 
******************************************************/
void mpi_status_set_elements( MPI_Fint * status, MPI_Fint *  datatype, MPI_Fint *  count, MPI_Fint * ierr)
{
  MPI_STATUS_SET_ELEMENTS( status, datatype, count, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Status_set_elements wrapper function 
******************************************************/
void mpi_status_set_elements_( MPI_Fint * status, MPI_Fint *  datatype, MPI_Fint *  count, MPI_Fint * ierr)
{
  MPI_STATUS_SET_ELEMENTS( status, datatype, count, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Status_set_elements wrapper function 
******************************************************/
void mpi_status_set_elements__( MPI_Fint * status, MPI_Fint *  datatype, MPI_Fint *  count, MPI_Fint * ierr)
{
  MPI_STATUS_SET_ELEMENTS( status, datatype, count, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Status_set_cancelled wrapper function 
******************************************************/
int MPI_Status_set_cancelled( MPI_Status * status, int flag)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Status_set_cancelled()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Status_set_cancelled( status, flag) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Status_set_cancelled wrapper function 
******************************************************/
void MPI_STATUS_SET_CANCELLED( MPI_Fint * status, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_Status local_status; 
  MPI_Status_f2c(status, &local_status);
  *ierr = MPI_Status_set_cancelled( &local_status, *flag) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_Status_set_cancelled wrapper function 
******************************************************/
void mpi_status_set_cancelled( MPI_Fint * status, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_STATUS_SET_CANCELLED( status, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Status_set_cancelled wrapper function 
******************************************************/
void mpi_status_set_cancelled_( MPI_Fint * status, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_STATUS_SET_CANCELLED( status, flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Status_set_cancelled wrapper function 
******************************************************/
void mpi_status_set_cancelled__( MPI_Fint * status, MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_STATUS_SET_CANCELLED( status, flag, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Finalized wrapper function 
******************************************************/
int MPI_Finalized( int * flag)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Finalized()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Finalized( flag) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Finalized wrapper function 
******************************************************/
void MPI_FINALIZED( MPI_Fint *  flag, MPI_Fint * ierr)
{
  *ierr = MPI_Finalized( flag) ; 
  return ; 
}

/******************************************************
***      MPI_Finalized wrapper function 
******************************************************/
void mpi_finalized( MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_FINALIZED( flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Finalized wrapper function 
******************************************************/
void mpi_finalized_( MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_FINALIZED( flag, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Finalized wrapper function 
******************************************************/
void mpi_finalized__( MPI_Fint *  flag, MPI_Fint * ierr)
{
  MPI_FINALIZED( flag, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Type_create_indexed_block wrapper function 
******************************************************/
int MPI_Type_create_indexed_block( int count, int blocklength, int * array_of_displacements, MPI_Datatype oldtype, MPI_Datatype * newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_indexed_block()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_indexed_block( count, blocklength, array_of_displacements, oldtype, newtype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_create_indexed_block wrapper function 
******************************************************/
void MPI_TYPE_CREATE_INDEXED_BLOCK( MPI_Fint *  count, MPI_Fint *  blocklength, MPI_Fint *  array_of_displacements, MPI_Fint *  oldtype, MPI_Aint * newtype, MPI_Fint * ierr)
{
  MPI_Datatype local_type;
  *ierr = MPI_Type_create_indexed_block( *count, *blocklength, array_of_displacements, MPI_Type_f2c(*oldtype), &local_type) ; 
  *newtype = MPI_Type_c2f(local_type);
  return ; 
}

/******************************************************
***      MPI_Type_create_indexed_block wrapper function 
******************************************************/
void mpi_type_create_indexed_block( MPI_Fint *  count, MPI_Fint *  blocklength, MPI_Fint *  array_of_displacements, MPI_Fint *  oldtype, MPI_Aint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_INDEXED_BLOCK( count, blocklength, array_of_displacements, oldtype, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_indexed_block wrapper function 
******************************************************/
void mpi_type_create_indexed_block_( MPI_Fint *  count, MPI_Fint *  blocklength, MPI_Fint *  array_of_displacements, MPI_Fint *  oldtype, MPI_Aint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_INDEXED_BLOCK( count, blocklength, array_of_displacements, oldtype, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_indexed_block wrapper function 
******************************************************/
void mpi_type_create_indexed_block__( MPI_Fint *  count, MPI_Fint *  blocklength, MPI_Fint *  array_of_displacements, MPI_Fint *  oldtype, MPI_Aint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_INDEXED_BLOCK( count, blocklength, array_of_displacements, oldtype, newtype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Request_get_status wrapper function 
******************************************************/
int MPI_Request_get_status( MPI_Request request, int * flag, MPI_Status * status)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Request_get_status()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Request_get_status( request, flag, status) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Request_get_status wrapper function 
******************************************************/
void MPI_REQUEST_GET_STATUS( MPI_Fint *  request, MPI_Fint *  flag, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_Status local_status; 
  *ierr = MPI_Request_get_status( MPI_Request_f2c(*request), flag, &local_status) ; 
  MPI_Status_c2f(&local_status, status);
  return ; 
}

/******************************************************
***      MPI_Request_get_status wrapper function 
******************************************************/
void mpi_request_get_status( MPI_Fint *  request, MPI_Fint *  flag, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_REQUEST_GET_STATUS( request, flag, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Request_get_status wrapper function 
******************************************************/
void mpi_request_get_status_( MPI_Fint *  request, MPI_Fint *  flag, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_REQUEST_GET_STATUS( request, flag, status, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Request_get_status wrapper function 
******************************************************/
void mpi_request_get_status__( MPI_Fint *  request, MPI_Fint *  flag, MPI_Fint * status, MPI_Fint * ierr)
{
  MPI_REQUEST_GET_STATUS( request, flag, status, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Get_address wrapper function 
******************************************************/
int MPI_Get_address( void * location, MPI_Aint * address)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Get_address()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Get_address( location, address) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Get_address wrapper function 
******************************************************/
void MPI_GET_ADDRESS( MPI_Aint * location, MPI_Aint * address, MPI_Fint * ierr)
{
  *ierr = MPI_Get_address( location, address) ; 
  return ; 
}

/******************************************************
***      MPI_Get_address wrapper function 
******************************************************/
void mpi_get_address( MPI_Aint * location, MPI_Aint * address, MPI_Fint * ierr)
{
  MPI_GET_ADDRESS( location, address, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Get_address wrapper function 
******************************************************/
void mpi_get_address_( MPI_Aint * location, MPI_Aint * address, MPI_Fint * ierr)
{
  MPI_GET_ADDRESS( location, address, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Get_address wrapper function 
******************************************************/
void mpi_get_address__( MPI_Aint * location, MPI_Aint * address, MPI_Fint * ierr)
{
  MPI_GET_ADDRESS( location, address, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Type_create_resized wrapper function 
******************************************************/
int MPI_Type_create_resized( MPI_Datatype oldtype, MPI_Aint lb, MPI_Aint extent, MPI_Datatype * newtype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_create_resized()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_create_resized( oldtype, lb, extent, newtype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_create_resized wrapper function 
******************************************************/
void MPI_TYPE_CREATE_RESIZED( MPI_Fint *  oldtype, MPI_Aint *  lb, MPI_Aint *  extent, MPI_Aint * newtype, MPI_Fint * ierr)
{
  MPI_Datatype local_type; 
  *ierr = MPI_Type_create_resized( MPI_Type_f2c(*oldtype), *lb, *extent, &local_type) ; 
  *newtype = MPI_Type_c2f(local_type);
  return ; 
}

/******************************************************
***      MPI_Type_create_resized wrapper function 
******************************************************/
void mpi_type_create_resized( MPI_Fint *  oldtype, MPI_Aint *  lb, MPI_Aint *  extent, MPI_Aint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_RESIZED( oldtype, lb, extent, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_resized wrapper function 
******************************************************/
void mpi_type_create_resized_( MPI_Fint *  oldtype, MPI_Aint *  lb, MPI_Aint *  extent, MPI_Aint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_RESIZED( oldtype, lb, extent, newtype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_create_resized wrapper function 
******************************************************/
void mpi_type_create_resized__( MPI_Fint *  oldtype, MPI_Aint *  lb, MPI_Aint *  extent, MPI_Aint * newtype, MPI_Fint * ierr)
{
  MPI_TYPE_CREATE_RESIZED( oldtype, lb, extent, newtype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


#ifdef TAU_MPITYPEEX

/******************************************************
***      MPI_Type_get_true_extent wrapper function 
******************************************************/
int MPI_Type_get_true_extent( MPI_Datatype datatype, MPI_Aint * true_lb, MPI_Aint * true_extent)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Type_get_true_extent()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Type_get_true_extent( datatype, true_lb, true_extent) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Type_get_true_extent wrapper function 
******************************************************/
void MPI_TYPE_GET_TRUE_EXTENT( MPI_Fint *  datatype, MPI_Aint * true_lb, MPI_Aint * true_extent, MPI_Fint * ierr)
{
  *ierr = MPI_Type_get_true_extent( MPI_Type_f2c(*datatype), true_lb, true_extent) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_true_extent wrapper function 
******************************************************/
void mpi_type_get_true_extent( MPI_Fint *  datatype, MPI_Aint * true_lb, MPI_Aint * true_extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_TRUE_EXTENT( datatype, true_lb, true_extent, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_true_extent wrapper function 
******************************************************/
void mpi_type_get_true_extent_( MPI_Fint *  datatype, MPI_Aint * true_lb, MPI_Aint * true_extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_TRUE_EXTENT( datatype, true_lb, true_extent, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Type_get_true_extent wrapper function 
******************************************************/
void mpi_type_get_true_extent__( MPI_Fint *  datatype, MPI_Aint * true_lb, MPI_Aint * true_extent, MPI_Fint * ierr)
{
  MPI_TYPE_GET_TRUE_EXTENT( datatype, true_lb, true_extent, ierr) ; 
  return ; 
}

#endif /* TAU_MPI_TYPEEX */

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Pack_external wrapper function 
******************************************************/
int MPI_Pack_external( char * datarep, void * inbuf, int incount, MPI_Datatype datatype, void * outbuf, MPI_Aint outsize, MPI_Aint * position)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Pack_external()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Pack_external( datarep, inbuf, incount, datatype, outbuf, outsize, position) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Pack_external wrapper function 
******************************************************/
void MPI_PACK_EXTERNAL( char * datarep, MPI_Aint * inbuf, MPI_Fint *  incount, MPI_Fint *  datatype, MPI_Aint * outbuf, MPI_Fint *  outsize, MPI_Aint * position, MPI_Fint * ierr)
{
  *ierr = MPI_Pack_external( datarep, inbuf, *incount, MPI_Type_f2c(*datatype), outbuf, *outsize, position) ; 
  return ; 
}

/******************************************************
***      MPI_Pack_external wrapper function 
******************************************************/
void mpi_pack_external( char * datarep, MPI_Aint * inbuf, MPI_Fint *  incount, MPI_Fint *  datatype, MPI_Aint * outbuf, MPI_Fint *  outsize, MPI_Aint * position, MPI_Fint * ierr)
{
  MPI_PACK_EXTERNAL( datarep, inbuf, incount, datatype, outbuf, outsize, position, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Pack_external wrapper function 
******************************************************/
void mpi_pack_external_( char * datarep, MPI_Aint * inbuf, MPI_Fint *  incount, MPI_Fint *  datatype, MPI_Aint * outbuf, MPI_Fint *  outsize, MPI_Aint * position, MPI_Fint * ierr)
{
  MPI_PACK_EXTERNAL( datarep, inbuf, incount, datatype, outbuf, outsize, position, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Pack_external wrapper function 
******************************************************/
void mpi_pack_external__( char * datarep, MPI_Aint * inbuf, MPI_Fint *  incount, MPI_Fint *  datatype, MPI_Aint * outbuf, MPI_Fint *  outsize, MPI_Aint * position, MPI_Fint * ierr)
{
  MPI_PACK_EXTERNAL( datarep, inbuf, incount, datatype, outbuf, outsize, position, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/


/******************************************************
***      MPI_Unpack_external wrapper function 
******************************************************/
int MPI_Unpack_external( char * datarep, void * inbuf, MPI_Aint insize, MPI_Aint * position, void * outbuf, int outcount, MPI_Datatype datatype)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Unpack_external()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Unpack_external( datarep, inbuf, insize, position, outbuf, outcount, datatype) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Unpack_external wrapper function 
******************************************************/
void MPI_UNPACK_EXTERNAL( char * datarep, MPI_Aint * inbuf, MPI_Fint *  insize, MPI_Aint * position, MPI_Aint * outbuf, MPI_Fint *  outcount, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  *ierr = MPI_Unpack_external( datarep, inbuf, *insize, position, outbuf, *outcount, MPI_Type_f2c(*datatype)) ; 
  return ; 
}

/******************************************************
***      MPI_Unpack_external wrapper function 
******************************************************/
void mpi_unpack_external( char * datarep, MPI_Aint * inbuf, MPI_Fint *  insize, MPI_Aint * position, MPI_Aint * outbuf, MPI_Fint *  outcount, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_UNPACK_EXTERNAL( datarep, inbuf, insize, position, outbuf, outcount, datatype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Unpack_external wrapper function 
******************************************************/
void mpi_unpack_external_( char * datarep, MPI_Aint * inbuf, MPI_Fint *  insize, MPI_Aint * position, MPI_Aint * outbuf, MPI_Fint *  outcount, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_UNPACK_EXTERNAL( datarep, inbuf, insize, position, outbuf, outcount, datatype, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Unpack_external wrapper function 
******************************************************/
void mpi_unpack_external__( char * datarep, MPI_Aint * inbuf, MPI_Fint *  insize, MPI_Aint * position, MPI_Aint * outbuf, MPI_Fint *  outcount, MPI_Fint *  datatype, MPI_Fint * ierr)
{
  MPI_UNPACK_EXTERNAL( datarep, inbuf, insize, position, outbuf, outcount, datatype, ierr) ; 
  return ; 
}

/******************************************************/
/******************************************************/

#ifdef TAU_MPIADDERROR

/******************************************************
***      MPI_Pack_external_size wrapper function 
******************************************************/
int MPI_Pack_external_size( char * datarep, int incount, MPI_Datatype datatype, MPI_Aint * size)
{
  int retvalue; 
  TAU_PROFILE_TIMER(t, "MPI_Pack_external_size()", "", TAU_MESSAGE); 
  TAU_PROFILE_START(t); 
  retvalue = PMPI_Pack_external_size( datarep, incount, datatype, size) ; 
  TAU_PROFILE_STOP(t); 
  return retvalue; 
}

/******************************************************
***      MPI_Pack_external_size wrapper function 
******************************************************/
void MPI_PACK_EXTERNAL_SIZE( char * datarep, MPI_Fint *  incount, MPI_Fint *  datatype, MPI_Aint * size, MPI_Fint * ierr)
{
  *ierr = MPI_Pack_external_size( datarep, *incount, MPI_Type_f2c(*datatype), size) ; 
  return ; 
}

/******************************************************
***      MPI_Pack_external_size wrapper function 
******************************************************/
void mpi_pack_external_size( char * datarep, MPI_Fint *  incount, MPI_Fint *  datatype, MPI_Aint * size, MPI_Fint * ierr)
{
  MPI_PACK_EXTERNAL_SIZE( datarep, incount, datatype, size, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Pack_external_size wrapper function 
******************************************************/
void mpi_pack_external_size_( char * datarep, MPI_Fint *  incount, MPI_Fint *  datatype, MPI_Aint * size, MPI_Fint * ierr)
{
  MPI_PACK_EXTERNAL_SIZE( datarep, incount, datatype, size, ierr) ; 
  return ; 
}

/******************************************************
***      MPI_Pack_external_size wrapper function 
******************************************************/
void mpi_pack_external_size__( char * datarep, MPI_Fint *  incount, MPI_Fint *  datatype, MPI_Aint * size, MPI_Fint * ierr)
{
  MPI_PACK_EXTERNAL_SIZE( datarep, incount, datatype, size, ierr) ; 
  return ; 
}

#endif /* TAU_MPIADDERROR */

/******************************************************/
/******************************************************/
