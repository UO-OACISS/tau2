/*********************************************************************/
/*                  pC++/Sage++  Copyright (C) 1993,1995             */
/*  Indiana University  University of Oregon  University of Rennes   */
/*  Author : Sameer Shende, 				             */ 
/*           sameer@cs.uoregon.edu 	     			     */
/*********************************************************************/

#include <stdio.h>

#define DEFAULT_INPUT_FILENAME 	"ave.trace.bpoint"
#define MAX_PROCS	64

FILE *fp;
FILE **fpout;
int file_opened[MAX_PROCS];
main(int argc, char** argv)
{
  int i, ret;
  char *outfile_name;
  int pid, oid,  msize, bflag, iflag;
  char mtype[64], mtag[64]; /* For RWMU  event type or W_elim etc. */
  char filename[256], out_file_name[256], error_msg[256], dummy[256];
  int version_no, max_procs;
  

  /* Initialise the file descriptors */
/*
  for (i=0; i<MAX_OUT_FILE; i++)
    fpout[i] = NULL;
*/

  /* Get the input trace file name or use the default */
  if (argc > 1)
	strcpy(filename, argv[1]);
  else
  { 	printf("Usage : %s <trace file name>\n",argv[0]);
	strcpy(filename, DEFAULT_INPUT_FILENAME);
	printf("This is a utility to convert the ariadne traces for pcxx\n");
	printf("Using default input trace file = %s\n", filename);
	printf("Creates pcxx_aa[1-n].trace as the n output files\n");
  }
 
  fp = fopen(filename,"r"); 
  if (fp == (FILE *) NULL)
  {
	sprintf(error_msg,"ERROR - Cannot open file %s\n", filename);
	perror(error_msg);
	exit(1);
  }

  /* First read the version number and the max processor number */
  ret = fscanf(fp,"%d %d\n", &version_no, &max_procs);
  if (ret < 0)
  {
	perror ("fscanf error in reading version no. and processor no");
	exit(1);
  }
/*
  printf("%d %d\n", version_no, max_procs);
*/
	/* bug in pC++ tracing. wrong no. of max_procs reported */
/* bug fixed */ 
/*
  max_procs = MAX_PROCS;
*/
  fpout = (FILE **) malloc(4*max_procs);
 for(i=0; i<max_procs; i++)
  {
        /* Allocate file pointers */
        fpout[i] = (FILE *) malloc(sizeof(FILE));
        if (fpout[i] == (FILE *) NULL)
        {
                sprintf(error_msg,"ERROR - fpout malloc fails i= %d ", i);
                perror(error_msg);
                exit(1);
        }
        sprintf(out_file_name,"pcxx_aa%d.trace",i);

        /* Open the output files */
        fpout[i] = fopen(out_file_name,"w");
        if (fpout[i] == (FILE *) NULL)
        {
                sprintf(error_msg,"ERROR* cannot open file %s", out_file_name);
                perror(error_msg);
                exit(1);
        }
	file_opened[i] = 1; /* file opened */
   }


  /* Set up the output files */
#ifdef NOT_NEEDED
  for(i=0; i<max_procs; i++)
  {
	/* Allocate file pointers */
	fpout[i] = (FILE *) malloc(sizeof(FILE));
	if (fpout[i] == (FILE *) NULL)
	{
		sprintf(error_msg,"ERROR - fpout malloc fails i= %d ", i);
		perror(error_msg);  
		exit(1);
	}
	
	file_opened[i] = 0; /* file not opened */
	/* Open the output files */
   }
#endif /* NOT_NEEDED */

  while ((ret = fscanf(fp, "%d %d %s %s %d %d %d", 
	&pid, &oid, mtype, mtag, &msize, &bflag,&iflag)) != -1)
  {
	if (iflag > 0)
	{ 
	  for (i=0; i< iflag; i++) fscanf(fp, "%s", dummy);
	}
/*
	printf("%d %d %s %s %d %d %d\n", pid, oid, mtype, mtag, msize, bflag, iflag);
*/
	
	/* Now put it in n different files. */
	if (file_opened[pid] == 0)
	{
	 	sprintf(out_file_name,"pcxx_aa%d.trace",pid);
		fpout[pid] = fopen(out_file_name,"w");
		if (fpout[i] == (FILE *) NULL)
		{
	 		sprintf(error_msg,"ERROR cannot open file %s", out_file_name);
	 		perror(error_msg);
	 		exit(1);
	 	}
		file_opened[pid] = 1;
		printf("opened file %s", out_file_name);
	}
	fprintf(fpout[pid],"%d %d %s %s %d %d %d\n", pid, oid, mtype, mtag, msize, bflag, iflag);	
	/* Write it in the file */
  }
  exit(0);
	
}
