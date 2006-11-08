/************************************************************************** 
*************************************************************************** 

  File: perf2tau.c
  Description: Convertor from Perflib to TAU performance data format
  Contact: Jeff Brown (jeffb@lanl.gov) Sameer Shende (sameer@cs.uoregon.edu)
   		
*************************************************************************** 
***************************************************************************/ 

#include <perf.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <stdio.h>

int verbose = 0;

#define dprintf if (verbose) printf


enum metrics { TAU_WTIME, TAU_RSS, TAU_FPINS, TAU_VIRTUALTIME, TAU_PAGEFAULTS, TAU_PROCMEM } tau_measurement; 

/*
extern int trace_enabled, mpitrace_enabled, memtrace_enabled, countertrace_enabled, iotrace_enabled;
*/

int GetNumRoutines(struct all_context_data_struct *all_context_data_ptr)
{
  int numroutines = 0;

  while(all_context_data_ptr)
  { /* iterate over all routines and increment the counter */
    all_context_data_ptr = all_context_data_ptr->next_routine;
    numroutines ++;
  }
  return numroutines;
}

char * GetMetricName(enum metrics measurement)
{
  switch (measurement)
  {
    case TAU_WTIME :
       return "GET_TIME_OF_DAY";
    case TAU_RSS:
       return "RSS";
    case TAU_FPINS:
       return "PAPI_FP_INS";
    case TAU_VIRTUALTIME:
       return "P_VIRTUAL_TIME";
    case TAU_PAGEFAULTS:
       return "PAGE_FAULTS";
    case TAU_PROCMEM:
       return "PROC_MEM";
    default:
       return "DEFAULT";
  }
}

double GetPerformanceDataValue(enum metrics measurement, struct data_struct dt)
{
  switch (measurement)
  {
    case TAU_WTIME :
       return dt.wtime * 1e6;
    case TAU_RSS:
       return dt.rss;
    case TAU_FPINS:
       return dt.flops;
    case TAU_VIRTUALTIME:
       return dt.ptime;
    case TAU_PAGEFAULTS:
       return dt.faults;
    case TAU_PROCMEM:
       return dt.procmem;
    default:
       return 0;
  }
}
   
int WriteMetricInTauFormat(enum metrics measurement, int rank, int numroutines, struct all_context_data_struct *all_context_data_ptr )
{
  char *metric;
  char newdirname[1024];
  char filename[1024];
  FILE *fp;
 
  metric = GetMetricName(measurement);

  /* Create a new directory name */
  dprintf("METRIC: %s\n", metric);
  
  sprintf(newdirname, "./MULTI__%s", metric);

  mkdir(newdirname,S_IRWXU | S_IRGRP | S_IXGRP);

  sprintf(filename, "%s/profile.%d.0.0", newdirname, rank);
  if ((fp = fopen(filename, "w+")) == NULL) {
    perror("ERROR: writing profile file");
    exit(1);
  }
  
  fprintf(fp, "%d templated_functions_MULTI_%s\n", numroutines, metric);
  fprintf(fp, "# Name Calls Subrs Excl Incl ProfileCalls\n");

  while (all_context_data_ptr)
  { /* iterate over each routine */
     
     fprintf(fp, "\"%s\" %g %g %g %g 0 GROUP=\"TAU_DEFAULT\"\n",
	all_context_data_ptr->name,
        all_context_data_ptr->calls,
        all_context_data_ptr->child_calls,
	GetPerformanceDataValue(measurement, all_context_data_ptr->exclusive_data),
	GetPerformanceDataValue(measurement, all_context_data_ptr->inclusive_data));

     dprintf("\"%s\" %g %g %g %g 0 GROUP=\"TAU_DEFAULT\"\n",
	all_context_data_ptr->name,
        all_context_data_ptr->calls,
        all_context_data_ptr->child_calls,
	GetPerformanceDataValue(measurement, all_context_data_ptr->exclusive_data),
	GetPerformanceDataValue(measurement, all_context_data_ptr->inclusive_data));
     dprintf("*******************\n");
     dprintf("name: %s\n", all_context_data_ptr->name);
     dprintf("calls: %g\n", all_context_data_ptr->calls);
     dprintf("child_calls: %g\n", all_context_data_ptr->child_calls);
     dprintf("Exclusive metric: %g\n", GetPerformanceDataValue(measurement, all_context_data_ptr->exclusive_data));
     dprintf("Inclusive metric: %g\n", GetPerformanceDataValue(measurement, all_context_data_ptr->inclusive_data));

     all_context_data_ptr = all_context_data_ptr->next_routine;
  }
  fprintf(fp, "0 aggregates\n");
  /* fclose(fp); */
  

}
void ShowUsage(void)
{
   printf("perf2tau [data_directory] [-h]\n");
   printf("Converts perflib data to TAU format. \n");
   printf("If an argument is not specified, it checks the perf_data_directory environment variable\n");
   printf("e.g., \n");
   printf("> perf2tau timing\n");
   printf("opens perf_data.timing directory to read perflib data \n");
   printf("If no args are specified, it tries to read perf_data.<current_date> file\n");
   exit(1);
}

int main(int argc, char **argv)
{

  struct perf_forest_struct *tree_cycle_ptr;
  struct all_context_data_cycle_struct *all_context_cycle_ptr;
  struct all_context_data_rank_struct *all_context_rank_ptr;
  struct all_context_data_struct *all_context_data_ptr;
  struct header_struct *header_ptr;
  char filename[1024];
  FILE *fp;



  int maxrank, cycle, rank, numroutines, i;
  char *data_directory;
  struct tm *broken_down_time;
  time_t lt;


  for(i = 0; i < argc; i++)
  {
    switch(i) {
      case 0: 
        /*  set up run date (default is today) - yyyymmdd format */
        data_directory = getenv("perf_data_directory");
        if (!data_directory)
        {
          lt = time(NULL);
          data_directory = malloc(9);
          if (lt == -1)
            strcpy(data_directory, "00000000");
          else
          {
            broken_down_time = localtime(&lt);
            sprintf(data_directory, "%04d%02d%02d", 1900+broken_down_time->tm_year, broken_down_time->tm_mon+1, broken_down_time->tm_mday);
            sprintf(filename, "perf_data.%s/header.0", data_directory);
            if (argc == 1) {
	      if ((fp = fopen(filename, "r")) == (FILE *)NULL) ShowUsage();
              fclose(fp); 
              dprintf("After closing fp\n");
            }
          }
        }
	break;

      case 1: 
	if (strcmp(argv[i], "-h") == 0) ShowUsage();
	else {
	  if (strcmp(argv[i], "-v") == 0) verbose = 1;
	  else {
	    data_directory = argv[i]; 
            dprintf("data_directory %s\n", data_directory);
          }
        } 
        break;

      default:
	ShowUsage();
        break;
    }
    dprintf("i = %d\n", i);
  }

  /* Initialize Perf post processing library */
  Perf_Init();

  /* retrieve header that contains what kind of metrics were measured */
  header_ptr = (struct header_struct *) Perf_Build_Header(data_directory);


  tree_cycle_ptr = (struct perf_forest_struct *)Perf_Build_Tree(data_directory, PERF_ALL_CYCLES, PERF_ALL_RANKS);

  /* Gets the entire tree. Now examine aggregate data within this rank */
  all_context_cycle_ptr = (struct all_context_data_cycle_struct *)Perf_Build_All_Context_Data(tree_cycle_ptr);

  while (all_context_cycle_ptr)
  { /* cycle */
    /* Go to the last cycle. If the next cycle is null stop else continue */
    dprintf("cycle: %d\n", all_context_cycle_ptr->cycle);
    if (all_context_cycle_ptr->next_cycle) 
    {
      all_context_cycle_ptr = all_context_cycle_ptr->next_cycle;
      continue;
    }       

    /* we are at the last cycle! */
    all_context_rank_ptr = all_context_cycle_ptr->cycle_data_ptr;
    /* within this rank, we now examine the data for the routines */
    rank = 0; 
    while (all_context_rank_ptr)
    {
      all_context_data_ptr = all_context_rank_ptr->rank_data_ptr;
      numroutines = GetNumRoutines(all_context_data_ptr);
      dprintf("Rank = %d, number of routines = %d\n",
	rank, numroutines);
      while (all_context_data_ptr)
      { /* iterate over each routine */
        if (header_ptr->trace_enabled)
          WriteMetricInTauFormat(TAU_WTIME, rank, numroutines, all_context_data_ptr);
        if (header_ptr->memtrace_enabled)
        { /* NOTE: memtrace flag turns on rss and pagefault tracking */
          WriteMetricInTauFormat(TAU_RSS, rank, numroutines, all_context_data_ptr);
          WriteMetricInTauFormat(TAU_PAGEFAULTS, rank, numroutines, all_context_data_ptr);
          /* WriteMetricInTauFormat(TAU_PROCMEM, rank, numroutines, all_context_data_ptr);
          */
        }

        if (header_ptr->countertrace_enabled)
        { /* countertrace flag turns on tracking flops and processor time */
          WriteMetricInTauFormat(TAU_FPINS, rank, numroutines, all_context_data_ptr);
          WriteMetricInTauFormat(TAU_VIRTUALTIME, rank, numroutines, all_context_data_ptr);
        }

        all_context_data_ptr = all_context_data_ptr->next_routine;
      }

      all_context_rank_ptr = all_context_rank_ptr->next_rank;
      rank++;
    } /* rank */
    all_context_cycle_ptr = all_context_cycle_ptr->next_cycle;
  } /* cycle */

  return 0;
}

