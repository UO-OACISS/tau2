// build with:
//    module load rca
//    cc -o get_node_loc get_node_loc.c

//#include <rca_lib.h>
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>
//#include <pmi.h>
//#include <sys/param.h>

int main( int argc, char* argv[] )
{
//   int node_id = -1, rc = -1, x, y, z;
//   mesh_coord_t mesh_coord;
//   rs_node_t rs_node;
   int numprocs, my_id,i;
   char hostname[1024];

   /* start MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);

//   PMI_Get_nid(my_id, &node_id);
//   gethostname(hostname, MAXHOSTNAMELEN);
FILE *nodefile = fopen("/proc/cray_xt/cname", "r");
fgets(hostname,1024,nodefile);
printf("%d:%s",my_id,hostname);
//   rc = rca_get_meshcoord( (uint16_t) node_id, &mesh_coord );
//   if ( rc )
//   {  printf( "error: rca_get_meshcoord failed with rc=%d .\n", rc );
//      x = y = z = -1;
//   }
//   else
//   {  x = (int) mesh_coord.mesh_x;
//      y = (int) mesh_coord.mesh_y;
//      z = (int) mesh_coord.mesh_z;
//   }

/* for prettiness print MPIRANK ordered list */
//   for (i=0;i<numprocs;i++)
//   {  if (i==my_id)
//      {  
//	printf( "mpirank %d with host %s node_id %d has coordinates: x y z = %d %d %d \n", my_id, hostname,node_id, x, y, z ); 
//        printf( "%d,%d,%d,%d\n", my_id, x, y, z );
//      }
//      MPI_Barrier(MPI_COMM_WORLD);
//   }
   MPI_Finalize();
   return 0;

}  // end int main

