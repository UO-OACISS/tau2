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
#define TAU_OPENMPI3_CONST const
#define TAU_CONST const
#define TAU_CONST2 const

/*************************************************************************/
/*************************************************************************/
#elif MPI_VERSION == 3
#warning "Found MPI Version == 3"
#define TAU_MPICH3_CONST const
#define TAU_OPENMPI3_CONST const
#define TAU_CONST const
#define TAU_CONST2 const

/*************************************************************************/
/* Anything to set for MPI 2? */
/*************************************************************************/
#elif MPI_VERSION == 2
#warning "Found MPI Version == 2"
#define TAU_MPICH3_CONST
#define TAU_OPENMPI3_CONST
#define TAU_CONST
#define TAU_CONST2

/*************************************************************************/
/* Assume MPI 1 standard, because...MPI_VERSION wasn't always defined. */
/*************************************************************************/
# else /* MPI_VERSION == 1 */

#endif /* if MPI_VERSION == ... */

#endif /* defined(MPI_VERSION) */

/* Deal with MPC */

#if defined(TAU_MPC)
#define TAU_NONMPC_CONST
#else
#define TAU_NONMPC_CONST const
#endif

