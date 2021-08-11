#include "mpi.h"

/*************************************************************************/
/* Old method(s)... */
/*************************************************************************/

#ifdef TAU_MPICH3
#define TAU_MPICH3_CONST const
#else
#define TAU_MPICH3_CONST
#endif

#ifdef TAU_OPENMPI3
#define TAU_OPENMPI3_CONST  const
#else
#define TAU_OPENMPI3_CONST
#endif

#ifdef TAU_MPICONSTCHAR
#define TAU_CONST const
#define TAU_CONST2 const
#else
#define TAU_CONST
#define TAU_CONST2
#endif

// OpenMPI is stupid. MPI_Info_set() doesn't use const, but
// MPI_Info_delete() and MPI_Info_get_valuelen() do.
#ifdef TAU_OPENMPI3
#define TAU_CONST2 const
#endif

/*************************************************************************/
/* New method! */
/*************************************************************************/

/* test for existence of the MPI_VERSION macro - it wasn't always defined */
#if defined(MPI_VERSION)

#undef TAU_MPICH3_CONST
#undef TAU_OPENMPI3_CONST
#undef TAU_CONST
#undef TAU_CONST2

/*************************************************************************/
/* for the future... */
/*************************************************************************/
#if MPI_VERSION > 3
#warning "Found MPI Version > 3"
#define TAU_MPICH3_CONST const
#if defined(OPEN_MPI)
#define TAU_OPENMPI3_CONST const
#else
#define TAU_OPENMPI3_CONST
#endif
#define TAU_CONST const
#define TAU_CONST2 const
#define TAU_NONMPC_CONST const

/*************************************************************************/
/*************************************************************************/
#elif MPI_VERSION == 3
#warning "Found MPI Version == 3"
#define TAU_MPICH3_CONST const
#if defined(OPEN_MPI)
#define TAU_OPENMPI3_CONST const
#else
#define TAU_OPENMPI3_CONST
#endif
#define TAU_CONST const
#define TAU_CONST2 const
#define TAU_NONMPC_CONST const

/*************************************************************************/
/* Anything to set for MPI 2? */
/*************************************************************************/
#elif MPI_VERSION == 2
#warning "Found MPI Version == 2"
#define TAU_MPICH3_CONST
#define TAU_OPENMPI3_CONST
#define TAU_CONST
#define TAU_CONST2
#define TAU_NONMPC_CONST

/*************************************************************************/
/* Assume MPI 1 standard, because...MPI_VERSION wasn't always defined. */
/*************************************************************************/
# else /* MPI_VERSION == 1 */


#endif /* if MPI_VERSION == ... */

#endif /* defined(MPI_VERSION) */

/* Deal with symbols that changed names from version 1 to 2 */

#if defined(MPI_VERSION) && MPI_VERSION > 1

#define TAU_MPI_FILE_ERRHANDLER_FUNCTION MPI_File_errhandler_function
#define TAU_MPI_WIN_ERRHANDLER_FUNCTION MPI_Win_errhandler_function
#define TAU_MPI_COMM_ERRHANDLER_FUNCTION MPI_Comm_errhandler_function

#else /* defined(MPI_VERSION) && MPI_VERSION > 1 */

#define TAU_MPI_FILE_ERRHANDLER_FUNCTION MPI_File_errhandler_fn
#define TAU_MPI_WIN_ERRHANDLER_FUNCTION MPI_Win_errhandler_fn
#define TAU_MPI_COMM_ERRHANDLER_FUNCTION MPI_Comm_errhandler_fn

#endif /* defined(MPI_VERSION) && MPI_VERSION > 1 */

/* Deal with MPC */

#if defined(TAU_MPC)
#undef TAU_NONMPC_CONST
#define TAU_NONMPC_CONST
#endif

