/*---------------------------------------------------------------------------*\
  $Id: fox.c,v 1.1 2005/06/29 19:15:30 sameer Exp $
\*---------------------------------------------------------------------------*/

/*--------------------------*\
   All include files needed 
\*--------------------------*/
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "gpshmem.h"
#include <TAU.h>

/*------------*\
  Definitions
\*------------*/
/*--------------------------------------------------*\
  A global debuging print level.  Use with caution!
\*--------------------------------------------------*/
#define UTIL_PRINTIT 0
/*--------------------*\
  temp string length
\*--------------------*/
#define TMPSTRLEN 1024
/*--------------------------------------------------------------------------*\
  Constants for Matrix definitions.  These values will have an impact on
  the final difference matrix norm through numerical roundoff.  The final
  norm value will also be a function of the number of processes used.  
\*--------------------------------------------------------------------------*/
#define A_VAL ((double) 5/(double)8)    
#define B_VAL ((double) 1/(double)7)
#define C_VAL ((double) 5.0)
#define D_VAL ((double) -2/(double)13)
#define E_VAL ((double) 6/(double)5)
#define F_VAL ((double) 2.0)
/*------------------------------------------------------------------------*\
 functions used in the code
\*------------------------------------------------------------------------*/

#define PROC_FROM_GRID(i,j,n) ((i)*(n)+(j))
#define INC_WRAP(i,j,n) (((i)+(j)+(n))%(n))
#define CLEANSTRING(s,l){\
           int __i;\
           char *__s = (char *)(s);\
           for(__i=0;__i<(int)(l);__i++)__s[__i]=(char)0;\
        }
/*-----------------------------------*\
  Define the rank of the full matrix 
\*-----------------------------------*/
#define RANK 840
#define LOCAL_RANK_PRINT_LIMIT 40
/*------------------------------------------------------------------------*\
   2*3  = 6;  *4   = 24;  *5   = 120;  *6   = 720;  *7   = 5040
\*------------------------------------------------------------------------*/

/*-----------*\
  prototypes 
\*-----------*/
void output_seq(double *A, int ilo, int ihi, int jlo, int jhi, 
		int numRows, int numCols, int fmt, double threshold);
void output( double *A, int ilo, int ihi, int jlo, int jhi, 
	     int numRows, int numCols, int fmt, double threshold);
double diffnorm(double *a, double *b, int len, int me);
void mat_mul(double *a, double *b, double *c, int rowa, int cola, int colb);
void foxit(double *A, double *B, double *C,
	   int rank, int local_rank, int myrow, int mycol, 
	   int me, int nproc,int sqrtproc);
void gen_A(double *buffer, int ilo, int ihi, int jlo, int jhi, 
	   int rowdim_A, int coldim_A);
void gen_B(double *buffer, int ilo, int ihi, int jlo, int jhi, 
	   int rowdim_B, int coldim_B);
void gen_C(double *buffer, int ilo, int ihi, int jlo, int jhi, 
	   int rowdim_C, int coldim_C, int crossDim);
void printf_seq(char *string, int str_len);
/*--------------------------------------------------------------------------*\
   Main Fox Driver program. 
\*--------------------------------------------------------------------------*/
int main(int argc, char *argv[])

{
  int master = 0; /* master process ID */
  int nproc;      /* number of processes involved */
  int nn;         /* sqrt(nproc) */
  int me;         /* my process ID */
  int nproctest;  /* sqrt then squared test that nproc is perfect square */
  int local_rank; /* rank of matrix patch for each process */
  int myrow, mycol;    /* row and column ID */
  int my_low_r, my_high_r; /* row index range of patch of matrix */
  int my_low_c, my_high_c; /* column index range of patch of matrix */
  char tmps[TMPSTRLEN]; /* temp output string */
  double *A_patch; /* patch of A matrix */
  double *B_patch; /* patch of B matrix */
  double *C_patch; /* patch of C matrix (computed) */
  double *Z_patch; /* patch of C matrix (analytical) */
  double mytime, mytime0, mytime1; /* timing variables */
  double norm, normall; /* local and global norm */
  int i,j;  /* dummy loop variables */

/*----------------------------------------*\
  Initialize TAU 
\*----------------------------------------*/
  TAU_PROFILE_TIMER(t, "main()", "int (int, char **)", TAU_DEFAULT);
  TAU_PROFILE_START(t);
/*----------------------------------------*\
  Initialize SHMEM and process IDs, NPROC
\*----------------------------------------*/
  (void) gpshmem_init(&argc,&argv);
  nproc = gpnumpes();
  me = gpmype();
  master = (me == 0);
  CLEANSTRING(tmps,TMPSTRLEN);
  (void)snprintf(tmps, sizeof(tmps), " ME = %2d, NPROC = %2d ",me,nproc);
  (void)printf_seq(tmps,strlen(tmps));
/*------------------------------------------------------*\
  Set NN
  Test that nproc is a perfect square or 1 (fox requirement)
\*------------------------------------------------------*/
  nn = ((int)sqrt((double)nproc));
  nproctest = nn;
  nproctest *= nproctest;
  (void) gpshmem_barrier_all();
  if (nproc > 1) {
    if (nproc != nproctest) {
      if (master) {
	(void)printf(" nproc not a perfect square!\n");
	(void)printf(" nproctest = %d \n nproc     = %d\n\n",nproctest,nproc);
      }
      (void)gpshmem_error(" fatal error ");
      TAU_PROFILE_EXIT("fatal error");
      (void)exit((int)911);
    }
  }
/*-------------------------------------------------*\
  check rank remainder with respect to sqrt(nproc)
\*-------------------------------------------------*/
  if ((RANK % nn) != 0) {
    if (master) {
      (void)printf(" RANK not divisible by sqrt(nproc)!\n");
      (void)printf(" RANK = %d; sqrt(nproc) = %d; remainder = %d\n",
		   RANK,nn,(RANK % nn));
    }
    (void)gpshmem_error(" fatal error ");
    TAU_PROFILE_EXIT("fatal error");
    (void)exit((int)911);
  }

/*-------------------------------------*\
   compute process row and column IDs
\*-------------------------------------*/
  (void)gpshmem_barrier_all();
  myrow = me / nn;
  mycol = me % nn;
  CLEANSTRING(tmps,TMPSTRLEN);
  (void)snprintf(tmps, sizeof(tmps), " ME = %2d, NPROC = %2d, row=%3d, col=%3d",
		me,nproc,myrow,mycol);
  (void)printf_seq(tmps,strlen(tmps));

/*-------------------------------*\
  compute local rank and ranges
\*-------------------------------*/
  (void)gpshmem_barrier_all();
  local_rank = RANK/nn;
  my_low_r  = myrow*local_rank;
  my_high_r = my_low_r + local_rank - 1 ;
  my_low_c  = mycol*local_rank;
  my_high_c = my_low_c + local_rank - 1;
  if (master) {
    (void)printf("RANK = %d;  local_rank = %d\n",RANK,local_rank);
    (void)fflush(stdout);
  }
  CLEANSTRING(tmps,TMPSTRLEN);
  (void)snprintf(tmps, sizeof(tmps), "ME=%2d row[low=%4d high=%4d] col[low=%4d high=%4d]",
		me,my_low_r,my_high_r,my_low_c,my_high_c);
  (void)printf_seq(tmps,strlen(tmps));
  (void)gpshmem_barrier_all();
  if (master) {
    (void)printf(" testing proc mapping \n");
    for (i=0;i<nn;i++)
      for (j=0;j<nn;j++) {
	(void)printf(" (%d,%d) -> %d\n",i,j,PROC_FROM_GRID(i,j,nn));
      }
    (void)fflush(stdout);
  }
  (void)gpshmem_barrier_all();
  A_patch = gpshmalloc(sizeof(double)*local_rank*local_rank); 
  B_patch = gpshmalloc(sizeof(double)*local_rank*local_rank); 
  C_patch = (double *)malloc(sizeof(double)*local_rank*local_rank); 
  for(i=0;i<(local_rank*local_rank);i++) C_patch[i]=(double)0.0;
  Z_patch = (double *)malloc(sizeof(double)*local_rank*local_rank); 
  
  if (master) {
    printf("Generate A: .. ");
    fflush(stdout);
  }
  mytime0 = gpshmem_time();
  gen_A(A_patch,my_low_r,my_high_r,my_low_c,my_high_c,RANK,RANK); 
  mytime = gpshmem_time() - mytime0;
  if (master){
    (void)printf("Time to generate A: %.3f seconds\n",mytime);
    fflush(stdout);
  }
  if (local_rank < LOCAL_RANK_PRINT_LIMIT)
    output_seq(A_patch,0,(local_rank-1),0,(local_rank-1),local_rank,local_rank,2,(double)1.0e-7);
  if (master) {
    printf("Generate B: .. ");
    fflush(stdout);
  }
  mytime0 = gpshmem_time();
  gen_B(B_patch,my_low_r,my_high_r,my_low_c,my_high_c,RANK,RANK); 
  mytime = gpshmem_time() - mytime0;
  if (master) {
    (void)printf("Time to generate B: %.3f seconds\n",mytime);
    fflush(stdout);
  }
  if (local_rank < LOCAL_RANK_PRINT_LIMIT)
    output_seq(B_patch,0,(local_rank-1),0,(local_rank-1),local_rank,local_rank,2,(double)1.0e-7);
  if (master) {
    printf("Generate C: .. ");
    fflush(stdout);
  }
  mytime0 = gpshmem_time();
  gen_C(Z_patch,my_low_r,my_high_r,my_low_c,my_high_c,RANK,RANK,RANK); 
  mytime = gpshmem_time() - mytime0;
  if (master) {
    (void)printf("Time to generate C: %.3f seconds\n",mytime);
    fflush(stdout);
  }
  if (local_rank < LOCAL_RANK_PRINT_LIMIT)
    output_seq(Z_patch,0,(local_rank-1),0,(local_rank-1),local_rank,local_rank,2,(double)1.0e-7);
  if (master) {
    printf("Compute C via fox: ..");
    fflush(stdout);
  }
  mytime0 = gpshmem_time();
  foxit(A_patch,B_patch,C_patch,RANK,local_rank,myrow,mycol,me,nproc,nn);
  mytime = gpshmem_time() - mytime0;
  if (master) {
    (void)printf("Time to compute C: %.3f seconds\n",mytime);
    fflush(stdout);
  }
  if (local_rank < LOCAL_RANK_PRINT_LIMIT)
    output_seq(C_patch,0,(local_rank-1),0,(local_rank-1),local_rank,local_rank,1,(double)1.0e-7);

  norm = diffnorm(C_patch,Z_patch,(local_rank*local_rank),me);
  
  CLEANSTRING(tmps,TMPSTRLEN);
  (void)snprintf(tmps, sizeof(tmps), "norm = %.20e",norm);
  (void)printf_seq(tmps,strlen(tmps));
  (void) gpshmem_double_sum_to_all(&normall,&norm,1,0,0,nproc,
				     (double *)NULL,(double *)NULL);
  if (master)
    (void)printf("global norm = %.20e\n",normall);
  
  gpshfree(A_patch); 
  gpshfree(B_patch); 
  free(C_patch); 
  free(Z_patch); 
  gpshmem_barrier_all();
  gpshmem_finalize();
  TAU_PROFILE_STOP(t);
  return (int)0;
}
void foxit(double *A, double *B, double *C,
	   int rank, int local_rank, int myrow, int mycol, 
	   int me, int nproc,int sqrtproc)
{
/*--------------------------------------------------------------------------------*\
   Fox's Algorithm (or Broadcast Multiply and Roll):
   *  Assumptions level 1
      - Square matrices of order N
      - Nproc processes where Nproc = N**2
      - patches or Elements aij, bij, cij are mapped to processor i*n +j
        . This is defined as process (i,j)
      - There are N stages of the computation:
         . Each stage computes one dot product term on EACH processor
           aik*bkj
      - Stages are ?limited?
        .  At stage 0 on process (i,j)  broadcast diagonal A patch
           cij = aii*bij
        .  The next stage each process multiplies:
           The element immediately to the right of the
           diagonal of A (in it's process row) by
           The element of B directly beneath its own
           element of B (in it's process column)
        .  At stage 1 on process (i,j)
           cij += ai,i+1*bi+1,j
        .  At stage k on process (i,j)
           cij += ai,i+k*bi+k,j
\*--------------------------------------------------------------------------------*/
  int stage;
  int i,j,count;
  int I,J,K;
  int *rowids, *colids;
  double *Atmp, *Btmp;
  char *mystring;
  TAU_PROFILE_TIMER(t, "foxit()", "", TAU_DEFAULT);
  TAU_PROFILE_START(t);
  I = myrow;
  J = mycol;
  Atmp = (double *)malloc(sizeof(double)*local_rank*local_rank);
  Btmp = (double *)malloc(sizeof(double)*local_rank*local_rank);
  mystring = (char *)malloc(1024);
  for(stage=0;stage<sqrtproc;stage++) {
/*--------------------------------------------------------------------------------*\
    broadcast AII
\*--------------------------------------------------------------------------------*/
    
    gpshmem_barrier_all();  /* ensure all procs on on same stage */
    K = INC_WRAP(I,stage,sqrtproc);
    if (mycol == K) {
      for(i=0;i<(local_rank*local_rank);i++) 
	Atmp[i] = A[i];
/*--------------------------------------------------------------------------------*\
      (void)sprintf(mystring," stage:%d broadcast A(%d,%d) from node %d ",
		    stage,I,K,me);
\*--------------------------------------------------------------------------------*/
    } else {
/*--------------------------------------------------------------------------------*\
      (void)sprintf(mystring," stage:%d fetching  A(%d,%d) from      %d ",
		    stage,I,K,
		    PROC_FROM_GRID(I,K,sqrtproc));
\*--------------------------------------------------------------------------------*/
      gpshmem_get(Atmp,A,(local_rank*local_rank),PROC_FROM_GRID(I,K,sqrtproc));
    }
/*--------------------------------------------------------------------------------*\
    (void)printf_seq(mystring,strlen(mystring));
    count = 0;
    for(i=0;i<local_rank;i++) {
      for(j=0;j<local_rank;j++) {
	(void)sprintf(mystring,"stage:%3d, A(%3d,%3d)=%22f",stage,i,j,Atmp[i]);
	(void)printf_seq(mystring,strlen(mystring));
      }
    }
\*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*\
    BI,J ->  BI+1,J  ->  BI-1,J <- BI,J
\*--------------------------------------------------------------------------------*/
    if (myrow == K) {
      for(i=0;i<(local_rank*local_rank);i++) 
	Btmp[i] = B[i];
/*--------------------------------------------------------------------------------*\
      (void)sprintf(mystring,"stage:%d multiply with B(%d,%d) <copy> node:%d",stage,
		    K,J,me);
\*--------------------------------------------------------------------------------*/
    } else {
/*--------------------------------------------------------------------------------*\
      (void)sprintf(mystring,"stage:%d multiply with B(%d,%d) <get> node:%d",stage,
		    K,J,
		    PROC_FROM_GRID(K,J,sqrtproc));
\*--------------------------------------------------------------------------------*/
      gpshmem_get(Btmp,B,(local_rank*local_rank),PROC_FROM_GRID(K,J,sqrtproc));
    }      
/*--------------------------------------------------------------------------------*\
    (void)printf_seq(mystring,strlen(mystring));
    count = 0;
    for(i=0;i<local_rank;i++) {
      for(j=0;j<local_rank;j++) {
	(void)sprintf(mystring,"stage:%3d, B(%3d,%3d)=%22f",stage,i,j,Btmp[i]);
	(void)printf_seq(mystring,strlen(mystring));
      }
    }
\*--------------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------------*\
    CIJ=AIk*BkJ
\*--------------------------------------------------------------------------------*/
    (void)mat_mul(Atmp,Btmp,C,local_rank,local_rank,local_rank);
  }
  free(Atmp);
  free(Btmp);
  free(mystring);
  TAU_PROFILE_STOP(t);
}
/*-------------------------------------------------------------*\
   Routine to generate matrix A
   ilo, ihi, jlo, jhi are C based indices values of the matrix, 
   NOT loop limits.
   rowdim_A, coldim_A is the dimension of the full matrix A.
   routine fills the buffer passed "linearly" with the data
\*-------------------------------------------------------------*/
void gen_A(double *buffer, int ilo, int ihi, int jlo, int jhi, 
	   int rowdim_A, int coldim_A)
{
  /*-------------------------------------------*\
     functional form of A(i,j) = ai + bj + c
  \*-------------------------------------------*/
  int i, j, count;         /* buffer index and loop indices */
  int iend, jend;	   /* loop limits */
  double a, b, c;          /* equation constants */
  int til, tih, tjl, tjh;  /* argument validity test variables */
  TAU_PROFILE_TIMER(t, "gen_A()", "", TAU_DEFAULT);
  TAU_PROFILE_START(t);
  /*---------------------------------*\
    Test validity of argument ranges 
  \*---------------------------------*/
  til = (ilo < 0 || ilo > (rowdim_A-1)) ; /* is ilo valid */
  tih = (ihi < 0 || ihi > (rowdim_A-1)) ; /*    ihi       */
  tjl = (jlo < 0 || jlo > (coldim_A-1)) ; /*    jlo       */
  tjh = (jhi < 0 || jhi > (coldim_A-1)) ; /*    jhi       */
  if (til || tih || tjl || tjh ) {
    (void)printf("-------------------*********--------------------/`\n");
    (void)printf(" gen_A: fatal argument error %d%d%d%d\n",til,tih,tjl,tjh);
    (void)printf(" I range %d to %d \n",ilo,ihi);
    (void)printf(" J range %d to %d \n",jlo,jhi);
    (void)printf(" rows=%d columns=%d \n",rowdim_A,coldim_A);
    (void)exit((int) 911); /* in case of emergency call */
  }

  /*---------------------------------------------------*\
    initialize constants from definitions in the 
    generation include file.  Initialize count which
    indexes the input/output buffer.
  \*---------------------------------------------------*/
  a = A_VAL;
  b = B_VAL;
  c = C_VAL;
  count = 0;
  /*------------------------------------------------------------------*\
    Determine end loops so that (0,0,0,0) will give the first element
  \*------------------------------------------------------------------*/
  iend = ilo + (ihi-ilo) + 1;
  if (iend > rowdim_A) iend=rowdim_A; /* should not be executed */
  jend = jlo + (jhi-jlo) + 1;
  if (jend > coldim_A) jend=coldim_A; /* should not be executed */

  /*-----------------------------------------------------------*\
    Compute desired element, patch, or full matrix of Matrix A
  \*-----------------------------------------------------------*/

  for (i=ilo;i<iend;i++) {
    for (j=jlo;j<jend;j++) {
      buffer[count] = a*(double)i +b*(double)j + c;
      if (UTIL_PRINTIT)
	(void)printf("A count=%d, i=%d j=%d iend=%d jend=%d value=%f\n",
		     count,i,j,iend,jend,buffer[count]);
      count++;
    }
  }
  TAU_PROFILE_STOP(t);
}
/*-------------------------------------------------------------*\
   Routine to generate matrix B
   ilo, ihi, jlo, jhi are C based indices values of the matrix, 
   NOT loop limits.
   rowdim_A, coldim_A is the dimension of the matrix B.
\*-------------------------------------------------------------*/
void gen_B(double *buffer, 
	   int ilo, int ihi, 
	   int jlo, int jhi, 
	   int rowdim_B, int coldim_B)
{
  /* 
     functional form of B(i,j) = di + ej + f
  */
  int i, j, count;         /* buffer index and loop indices */   
  int iend, jend;	   /* loop limits */
  double d, e, f;	   /* equation constants */
  int til, tih, tjl, tjh;  /* argument validity test variables */
  TAU_PROFILE_TIMER(t, "gen_B()", "", TAU_DEFAULT);
  TAU_PROFILE_START(t);
/*---------------------------------*\
  Test validity of argument ranges 
\*---------------------------------*/
  til = (ilo < 0 || ilo > (rowdim_B-1)) ;  /* is ilo valid */
  tih = (ihi < 0 || ihi > (rowdim_B-1)) ;  /*    ihi       */
  tjl = (jlo < 0 || jlo > (coldim_B-1)) ;  /*    jlo       */
  tjh = (jhi < 0 || jhi > (coldim_B-1)) ;  /*    jhi       */
  if (til || tih || tjl || tjh ) {
    (void)printf("-------------------*********--------------------/`\n");
    (void)printf(" gen_B: fatal argument error %d%d%d%d\n",til,tih,tjl,tjh);
    (void)printf(" I range %d to %d \n",ilo,ihi);
    (void)printf(" J range %d to %d \n",jlo,jhi);
    (void)printf(" rows=%d columns=%d \n",rowdim_B,coldim_B);
    (void)exit((int) 911); /* in case of emergency call */
  }

/*---------------------------------------------------*\
  initialize constants from definitions in the 
  generation include file.  Initialize count which
  indexes the input/output buffer.
\*---------------------------------------------------*/
  d = D_VAL;
  e = E_VAL;
  f = F_VAL;
  count = 0;
/*------------------------------------------------------------------*\
  Determine end loops so that (0,0,0,0) will give the first element
\*------------------------------------------------------------------*/
  iend = ilo + (ihi-ilo) + 1;
  if (iend > rowdim_B) iend=rowdim_B;
  jend = jlo + (jhi-jlo) + 1;
  if (jend > coldim_B) jend=coldim_B;

/*-----------------------------------------------------------*\
  Compute desired element, patch, or full matrix of Matrix B
\*-----------------------------------------------------------*/

  for (i=ilo;i<iend;i++) {
    for (j=jlo;j<jend;j++) {
      buffer[count] = d*(double)i + e*(double)j + f;
      if (UTIL_PRINTIT)
	(void)printf("B count=%d, i=%d j=%d iend=%d jend=%d value=%f\n",
		     count,i,j,iend,jend,buffer[count]);
      count++;
    }
  }
  TAU_PROFILE_STOP(t);
}

/*-------------------------------------------------------------*\
   Routine to generate matrix C
   ilo, ihi, jlo, jhi are C based indices values of the matrix, 
   NOT loop limits.
   rowdim_A, coldim_A is the dimension of the matrix C.
   crossDim is the inner dimension (Columns_of_A or Rows_of_B)
            of the matrix multiply
\*-------------------------------------------------------------*/
void gen_C(double *buffer, int ilo, int ihi, int jlo, int jhi, 
	   int rowdim_C, int coldim_C, int crossDim)
{
  /* 
     functional form of A(i,j) = (ai + bj +c)
     functional form of B(i,j) = (di + ej +f)
     C(i,j) = sum(k,k=1,crossDim)(ai + bk +c)*(dk + ej +f)
  */
  int i, j, k, count;         /* buffer index and loop indices */
  int iend, jend;             /* loop limits */
  double sum;                 /* temporary sum */
  double a, b, c;             /* equation constants for A*/
  double d, e, f;             /* more equation constants for B*/
  int til, tih, tjl, tjh;     /* argument validity test variables */
  TAU_PROFILE_TIMER(t, "gen_C()", "", TAU_DEFAULT);
  TAU_PROFILE_START(t);
/*---------------------------------*\
  Test validity of argument ranges 
\*---------------------------------*/
  til = (ilo < 0 || ilo > (rowdim_C-1)) ; /* is ilo valid */
  tih = (ihi < 0 || ihi > (rowdim_C-1)) ; /*    ihi       */
  tjl = (jlo < 0 || jlo > (coldim_C-1)) ; /*    jlo       */
  tjh = (jhi < 0 || jhi > (coldim_C-1)) ; /*    jhi       */
  if (til || tih || tjl || tjh ) {
    (void)printf("-------------------*********--------------------/`\n");
    (void)printf(" gen_C: fatal argument error %d%d%d%d\n",til,tih,tjl,tjh);
    (void)printf(" I range %d to %d \n",ilo,ihi);
    (void)printf(" J range %d to %d \n",jlo,jhi);
    (void)printf(" rows=%d columns=%d \n",rowdim_C,coldim_C);
    (void)exit((int) 911);  /* in case of emergency call */
  }

/*---------------------------------------------------*\
  initialize constants from definitions in the 
  generation include file.  Initialize count which
  indexes the input/output buffer.
\*---------------------------------------------------*/
  a = A_VAL;  b = B_VAL;  c = C_VAL;
  d = D_VAL;  e = E_VAL;  f = F_VAL;
  count = 0;
/*------------------------------------------------------------------*\
  Determine end loops so that (0,0,0,0) will give the first element
\*------------------------------------------------------------------*/
  iend = ilo + (ihi-ilo) + 1;
  if (iend > rowdim_C) iend=rowdim_C;
  jend = jlo + (jhi-jlo) + 1;
  if (jend > coldim_C) jend=coldim_C;

/*-----------------------------------------------------------*\
  Compute desired element, patch, or full matrix of Matrix C
\*-----------------------------------------------------------*/

  for (i=ilo;i<iend;i++) {
    for (j=jlo;j<jend;j++) {
      sum = (double)0.0;
      for (k=0;k<crossDim;k++) {
	sum += (a*i + b*k +c)*(d*k+e*j+f);
      }
      buffer[count] = sum;
      if (UTIL_PRINTIT)
	(void)printf("C count=%d, i=%d j=%d iend=%d jend=%d value=%f\n",
		     count,i,j,iend,jend,buffer[count]);
      count++;
    }
  }
  TAU_PROFILE_STOP(t);
}
/*-------------------------------------------------------------*\
   Routine to compute matrix multiply of two matrices
   Uses the "ddot" e.g., simple loop algorithm.
\*-------------------------------------------------------------*/
void mat_mul(double *a, double *b, double *c, int rowa, int cola, int colb)
{
  int i, j, k;          /* loop counters */
  int rowc, colc, rowb; /* sizes not passed as arguments */
  double sum;           /* intermediate sum variable */
  int count;            /* access pointer for linear memory */
  TAU_PROFILE_TIMER(t, "mat_mul()", "", TAU_DEFAULT);
  TAU_PROFILE_START(t);
/*---------------------------------------*\
   define missing sizes from known sizes 
\*---------------------------------------*/
  rowb = cola;   
  rowc = rowa;  
  colc = colb;
  count = 0;
  for(i=0;i<rowc;i++) {
    for(j=0;j<colc;j++) {
      sum = (double)0.0;
      for(k=0;k<cola;k++) {
	sum += *(a + i*cola + k) * *(b + k*colb + j);
      }
      *(c + count) += sum;
      count++;
    }
  }
  TAU_PROFILE_STOP(t);
}
double diffnorm(double *a, double *b, int len, int me)
{
  int i;
  double diff, norm;
  TAU_PROFILE_TIMER(t, "diffnorm()", "", TAU_DEFAULT);
  TAU_PROFILE_START(t);
  norm = (double) 0.0;
  for (i=0;i<len;i++) {
    diff = a[i] - b[i];
    if (UTIL_PRINTIT) {
      (void)printf("a[%d] = %f, b[%d] = %f, diff = %f\n",i,a[i],i,b[i],diff);
      (void)fflush(stdout);
      (void)fflush(stdout);
      (void)fflush(stdout);
      (void)fflush(stdout);
      (void)fflush(stdout);
      (void)fflush(stdout);
      (void)fflush(stdout);
      (void)fflush(stdout);
      (void)fflush(stdout);
    }
    norm += diff*diff;
  }
  TAU_PROFILE_STOP(t);
  return norm;
}
void printf_seq(char *string,int str_len)
{
  char *s2print;
  int i, nproc, me;
  TAU_PROFILE_TIMER(t, "printf_seq()", "", TAU_DEFAULT);
  TAU_PROFILE_START(t);
  gpshmem_barrier_all();
  nproc = gpnumpes();
  me = gpmype();
  s2print = gpshmalloc(sizeof(char)*str_len);
  (void)strncpy(s2print, string, sizeof(char)*str_len); 
  gpshmem_barrier_all();
  if(me==0) {
    (void)printf("<%d> %s\n",me,string);
    for(i=1;i<nproc;i++){
      gpshmem_getmem(string,s2print,str_len,i);
      (void)printf("<%d> %s\n",i,string);
      (void)fflush(stdout);
    }
    (void)strcpy(string,s2print); /* replace string with original contents */
  }
  gpshfree(s2print);
  gpshmem_barrier_all();
  TAU_PROFILE_STOP(t);
}
void output_seq(double *A, int ilo, int ihi, int jlo, int jhi, 
		int numRows, int numCols, int fmt, double threshold)
{
  int *whotodoit, valuegot;
  int nproc, me;
  int i;
  TAU_PROFILE_TIMER(t, "output_seq()", "", TAU_DEFAULT);
  TAU_PROFILE_START(t);
  whotodoit = gpshmalloc(sizeof(int));
  nproc = gpnumpes();
  me = gpmype();
  gpshmem_barrier_all();gpshmem_barrier_all();gpshmem_barrier_all();
  if (me==0) {
    printf("Beginning output from node: %d \n",me);
    fflush(stdout);
    output(A,ilo,ihi,jlo,jhi,numRows,numCols,fmt,threshold);
    fflush(stdout);
    printf("End of    output from node: %d \n",me);
    fflush(stdout);
  }
  gpshmem_barrier_all();gpshmem_barrier_all();gpshmem_barrier_all();
  for(i=1;i<nproc;i++) {
    /*    if (me==0) *whotodoit = i;*/
    gpshmem_barrier_all();gpshmem_barrier_all();gpshmem_barrier_all();
    /*    GPSHMEM_GETMEM_C(&valuegot,whotodoit,sizeof(int),0);*/
    gpshmem_broadcast32(whotodoit,&i,1,0,0,0,nproc,&valuegot);
    valuegot = *whotodoit;
    if (me==valuegot) {
      printf("Beginning output from node: %d \n",me);
      fflush(stdout);
      output(A,ilo,ihi,jlo,jhi,numRows,numCols,fmt,threshold);
      fflush(stdout);
      printf("End of    output from node: %d \n",me);
      fflush(stdout);
    } else {
      sleep(1);
    }
    gpshmem_barrier_all();gpshmem_barrier_all();gpshmem_barrier_all();
  }
  gpshmem_barrier_all();gpshmem_barrier_all();gpshmem_barrier_all();
  gpshfree(whotodoit);
  gpshmem_barrier_all();gpshmem_barrier_all();gpshmem_barrier_all();
  TAU_PROFILE_STOP(t);
}
void output(double *A, int ilo, int ihi, int jlo, int jhi, 
	    int numRows, int numCols, int fmt, double threshold)
{
  int i, j, jb, jbmax;
  int rownonzero;
  int pntcol = 6;
  int zeromatrix = 1;
  TAU_PROFILE_TIMER(t, "output()", "", TAU_DEFAULT);
  TAU_PROFILE_START(t);
  if (fmt < 1 || fmt > 2) {
    fprintf(stderr,"void output: illegal format request either 1 or 2 \n");
    return;
  }
  if ((ihi<ilo) || (jhi<jlo)) return;
  if (ihi >= numRows) ihi = numRows - 1;
  if (jhi >= numCols) jhi = numCols - 1;
  for (i=ilo;i<=ihi;i++) {
    for (j=jlo;j<=jhi;j++) {
      if(fabs(A[(i*numCols+j)])>threshold) {
	zeromatrix = 0;
	i += numRows + 1; j += numCols + 1;  /* kick out of loop at first non zero element */
      }
    }
  }
  if (zeromatrix) {
    printf("\n zero matrix\n");    fflush(stdout);
    return;
  }
  for (j=jlo;j<=jhi;j += pntcol) {
    jbmax = j + pntcol;
    if (jbmax > (jhi+1)) jbmax = jhi+1;
    for(jb=j;jb<jbmax;jb++)
      printf(" %15d",jb);
    printf("\n");fflush(stdout);
    for(i=ilo;i<=ihi;i++) {
      rownonzero = 0;
      for(jb=j;jb<jbmax;jb++) {
	if (fabs(A[(i*numCols+jb)]) > threshold) {
	  rownonzero++;
	  jb += jbmax + 1;
	}
      }
      if (rownonzero) {
	printf("%5d ",i);
	for(jb=j;jb<jbmax;jb++) {
	  if (fmt == 1)   printf(" %15f",A[i*numCols+jb]);
	  if (fmt == 2)   printf(" %15.8e",A[i*numCols+jb]);
	}
	printf("\n");fflush(stdout);
      }
    }
    printf("\n");fflush(stdout);
  }
  TAU_PROFILE_STOP(t);
}
