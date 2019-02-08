#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"mpi.h"
#define maxarray 100000

void main(int argc, char* argv[])
{
  int Pid,rPid,bPid,newPid;
  int N_proc;
  int dim_size[2];
  int period[2];
  int coords[2];
  int source;
  int dest;
  int tag = 10;
  int color;
  int rank;
  int i;
  
  int Pdim0,Pdim1;
  int nbrleft,nbrright,nbrtop,nbrbottom;

  int buffer;

  MPI_Status  status;
  MPI_Comm    comm2d; /* Cartesian 2d communicator  */
  MPI_Comm redblue;   /* alternating communicators */


  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&Pid);
  MPI_Comm_size(MPI_COMM_WORLD,&N_proc);
 
  if (Pid == 0)
    {
      if (argc == 3)
	{
                  
	  Pdim0 = atol(argv[1]);
	  Pdim1 = atol(argv[2]);
 
	  printf(" Pdim0 = %d \n",Pdim0);
	  printf(" Pdim1 = %d \n",Pdim1);

	}
      else
	{
	  printf(" command line is wrong \n");
	  printf(" should be for example 'executable 2 2 '\n");
	  exit(0);
	}
    }

  MPI_Bcast(&Pdim0,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(&Pdim1,1,MPI_INT,0,MPI_COMM_WORLD);

  /*    printf("flag 1 \n"); */
  dim_size[0] =  Pdim0;
  dim_size[1] =  Pdim1;

  period[0] = 0;
  period[1] = 0;
  /*     printf(" N_proc %d dim_size0 = %d \n",N_proc,dim_size[0]); */
  MPI_Dims_create(N_proc,2,dim_size);

  MPI_Cart_create(MPI_COMM_WORLD,
		  2,
		  dim_size,
		  period,
		  1,
		  &comm2d);
  /*    printf("flag 2 \n"); */
  MPI_Comm_rank(comm2d,&Pid);
  /*     printf("flag 3 \n"); */
  MPI_Cart_shift(comm2d,0,1,&nbrleft,&nbrright);
  /*     printf("flag 4 \n"); */
  MPI_Cart_shift(comm2d,1,1,&nbrtop,&nbrbottom);

  /* fnd2ddecomp(comm2d,nx,ny,sx,ex,sy,ey); */

  MPI_Cart_get(comm2d,
	       2,
	       dim_size,
	       period,
	       coords);

  if (Pid % 2 == 0) /* use modulus to find even/odd pid */
    {
      color = 2;
      MPI_Comm_split(comm2d,color,rank, &redblue); /* even processor ids  */
      MPI_Comm_rank(redblue,&newPid);
      MPI_Comm_set_name(redblue, "EVEN");
    }
  else
    {
      color = 3;
      MPI_Comm_split(comm2d,color,rank, &redblue); /* odd processor ids  */
      MPI_Comm_rank(redblue,&newPid);
      MPI_Comm_set_name(redblue, "ODD");
    }

  /* initialize data to be broadcast */

  if (color == 2)
    buffer = 25;
  else
    buffer = 50;
    
  printf("Before Bcast: redblue=%p, comm2d=%p, world=%p\n", redblue, comm2d, MPI_COMM_WORLD);
  printf("Before Bcast: addr redblue=%lx, comm2d=%lx\n", &redblue, &comm2d );
  MPI_Bcast(&buffer,1,MPI_INT,0,redblue);

  for (i = 0;i< N_proc;i++)
    {
      if ((i == Pid) && (Pid % 2 == 0))
	{
	  printf("Pid = %d newPid = %d buffer is %d \n",Pid,newPid,buffer);
	}
       if ((i == Pid) && (Pid % 2 == 1))
	{
	  printf("Pid = %d newPid = %d buffer is %d \n",Pid,newPid,buffer);
	}     
    }

  MPI_Barrier(MPI_COMM_WORLD);
  //  MPI_Comm_free(&redblue);
  MPI_Comm_free(&comm2d);

  MPI_Finalize();

}
