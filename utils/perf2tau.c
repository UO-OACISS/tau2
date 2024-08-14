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
#include <string.h>

int verbose = 0;
int callpath = 1;  /* show callpaths */
struct header_struct *header_ptr;

#define dprintf if (verbose) printf
#define TAU_NO_COUNTER_ID 0


enum metrics { TAU_WTIME, TAU_RSS, TAU_COUNTERS, TAU_VIRTUALTIME, TAU_PAGEFAULTS, TAU_PROCMEM } tau_measurement; 

/*
extern int trace_enabled, mpitrace_enabled, memtrace_enabled, countertrace_enabled, iotrace_enabled;
*/

int GetNumRoutines(struct all_context_data_struct *all_context_data_ptr)
{
  int numroutines = 0;
  struct context_struct *context_ptr; /* are there callpaths that should be counted? */

  while(all_context_data_ptr)
  { /* iterate over all routines and increment the counter */
    if (callpath)
    { /* should we take into account all the callpaths as well? */
      context_ptr = all_context_data_ptr->context_ptr; 
      while(context_ptr && IsCallpath(context_ptr->path)) 	
      { /* iterate over all contexts or callpaths for this routine */
        numroutines++;
        context_ptr = context_ptr->next_context; /* until it is null */
      }
    } /* callpaths */

    /* now add the number for the node in the callgraph after the edges are counted */
    all_context_data_ptr = all_context_data_ptr->next_routine;
    numroutines ++; /* the given routine */
  }
  return numroutines; /* or rather, the number of entities to be counted */
}

char * GetMetricName(enum metrics measurement, int counterid)
{
  switch (measurement)
  {
    case TAU_WTIME :
       return "GET_TIME_OF_DAY";
    case TAU_RSS:
       return "RSS";
    case TAU_COUNTERS:
       return header_ptr->counter_label[counterid]; /* pick up the name of the counter*/
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

double GetPerformanceDataValue(enum metrics measurement, struct data_struct dt, int counterid)
{
  switch (measurement)
  {
    case TAU_WTIME :
       return dt.wtime * 1e6;
    case TAU_RSS:
       return dt.rss;
    case TAU_COUNTERS:
       return dt.counter[counterid];
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
   
int WriteRoutineDataInFile(FILE *fp, char *name, double numcalls, double childcalls, double excl, double incl, char *group)
{
     if (incl < excl) incl = excl; 
     /* ASSUMPTION: inclusive time is never less than exclusive time */

     fprintf(fp, "\"%s\" %g %g %g %g 0 GROUP=\"%s\"\n",
        name, numcalls, childcalls, excl, incl, group);
     dprintf("\"%s\" %g %g %g %g 0 GROUP=\"%s\"\n",
        name, numcalls, childcalls, excl, incl, group);

     dprintf("*******************\n");
     dprintf("name: %s\n", name);
     dprintf("calls: %g\n", numcalls);
     dprintf("child_calls: %g\n", childcalls);
     dprintf("Exclusive metric: %g\n", excl);
     dprintf("Inclusive metric: %g\n", incl);
     dprintf("Group : %s\n", group);
     return 0;
}

int IsCallpath(char *path)
{ /* is there a => in the name? */
  if (strstr(path, "=>") == NULL)
    return 0; /* not found */
  else 
    return 1; /* found */
}

int WriteMetricInTauFormat(enum metrics measurement, int rank, int numroutines, struct all_context_data_struct *all_context_data_ptr , int counterid)
{
  char *metric;
  char newdirname[1024];
  char filename[1024];
  FILE *fp;
  struct context_struct *context_ptr; /* are there callpaths that should be counted? */

  metric = GetMetricName(measurement, counterid);

  /* Create a new directory name */
  dprintf("METRIC: %s\n", metric);
  
  snprintf(newdirname, sizeof(newdirname),  "./MULTI__%s", metric);

  mkdir(newdirname,S_IRWXU | S_IRGRP | S_IXGRP);

  snprintf(filename, sizeof(filename),  "%s/profile.%d.0.0", newdirname, rank);
  if ((fp = fopen(filename, "w+")) == NULL) {
    perror("ERROR: writing profile file");
    exit(1);
  }
  
  fprintf(fp, "%d templated_functions_MULTI_%s\n", numroutines, metric);
  fprintf(fp, "# Name Calls Subrs Excl Incl ProfileCalls\n");

  while (all_context_data_ptr)
  { /* iterate over each routine */
     
     WriteRoutineDataInFile(fp, 
	all_context_data_ptr->name,
        all_context_data_ptr->calls,
        all_context_data_ptr->child_calls,
	GetPerformanceDataValue(measurement, all_context_data_ptr->exclusive_data, counterid),
	GetPerformanceDataValue(measurement, all_context_data_ptr->inclusive_data, counterid),
	"TAU_DEFAULT");

     if (callpath)
     { /* should we take into account all the callpaths as well? */
       context_ptr = all_context_data_ptr->context_ptr;
       while(context_ptr)        
       { /* iterate over all contexts or callpaths for this routine */
         if(IsCallpath(context_ptr->path)) {

           WriteRoutineDataInFile(fp, 
             context_ptr->path, 
             context_ptr->calls,
             context_ptr->child_calls,
	     GetPerformanceDataValue(measurement, context_ptr->exclusive_data, counterid),
	     GetPerformanceDataValue(measurement, context_ptr->inclusive_data, counterid),
	     "TAU_CALLPATH");

         }            
         context_ptr = context_ptr->next_context; /* until it is null */
       }
     } /* callpaths */

     all_context_data_ptr = all_context_data_ptr->next_routine;
  }
  fprintf(fp, "0 aggregates\n");
  fclose(fp);
  return 0;  

}
void ShowUsage(void)
{
   printf("perf2tau [data_directory] [-h] [-v] [-flat] \n");
   printf("Converts perflib data to TAU format. \n");
   printf("If an argument is not specified, it checks the PERF_DATA_DIRECTORY environment variable\n");
   printf("e.g., \n");
   printf("> perf2tau timing\n");
   printf("opens perf_data.timing directory to read perflib data \n");
   printf("If no args are specified, it tries to read perf_data.<current_date> file\n");
   printf("-h : help\n");
   printf("-v : verbose\n");
   printf("-flat: by default, callpath profiles are generated. -flat forces flat profile generation. \n");
   exit(1);
}

int main(int argc, char **argv)
{

  struct perf_forest_struct *tree_cycle_ptr;
  struct all_context_cycle_struct *all_context_cycle_ptr;
  struct all_context_rank_struct *all_context_rank_ptr;
  struct all_context_data_struct *all_context_data_ptr;
  char filename[1024];
  FILE *fp;



  int maxrank, cycle, rank, numroutines, i;
  char *data_directory;
  struct tm *broken_down_time;
  struct perflib_struct *perflib_ptr;   
  time_t lt;


  for(i = 0; i < argc; i++)
  {
    switch(i) {
      case 0: 
        /*  set up run date (default is today) - yyyymmdd format */
        data_directory = getenv("PERF_DATA_DIRECTORY");
        if (!data_directory)
        {
          lt = time(NULL);
          const int len = 9;
          data_directory = (char *) malloc(len);
          if (lt == -1)
            strncpy(data_directory,  "00000000", len); 
          else
          {
            broken_down_time = localtime(&lt);
            snprintf(data_directory, len,  "%04d%02d%02d", 1900+broken_down_time->tm_year, broken_down_time->tm_mon+1, broken_down_time->tm_mday);
            snprintf(filename, sizeof(filename), "perf_data.%s/header.0", data_directory);
            if (argc == 1) {
	      if ((fp = fopen(filename, "r")) == (FILE *)NULL) ShowUsage();
              fclose(fp); 
              dprintf("After closing fp\n");
            }
          }
        }
	break;

       default: 
	 if (strcmp(argv[i], "-h") == 0) ShowUsage();
	 if (strcmp(argv[i], "-v") == 0) verbose = 1;
	 if (strcmp(argv[i], "-flat") == 0) callpath = 0; /* no callpath profiles */
         if (argv[i][0] != '-') 
         { /* retrieve the name of the data directory */
	   data_directory = argv[i]; 
           dprintf("data_directory %s\n", data_directory);
         }
         break;
    }
    dprintf("i = %d\n", i);
  }

  /* Initialize Perf post processing library */
  perflib_ptr = (struct perflib_struct *) Perf_Initpp(data_directory);

  /* retrieve header that contains what kind of metrics were measured */
  header_ptr = perflib_ptr->header_ptr;


  tree_cycle_ptr = (struct perf_forest_struct *)Perf_Build_Forest(perflib_ptr, PERF_ALL_CYCLES, PERF_ALL_RANKS);

  /* Gets the entire tree. Now examine aggregate data within this rank */
  all_context_cycle_ptr = (struct all_context_cycle_struct *)Perf_Build_All_Context_Data(perflib_ptr, tree_cycle_ptr);

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
        if (header_ptr->profile_time_enabled)
          WriteMetricInTauFormat(TAU_WTIME, rank, numroutines, all_context_data_ptr, TAU_NO_COUNTER_ID);
        if (header_ptr->profile_memory_enabled)
        { /* NOTE: memtrace flag turns on rss and pagefault tracking */
          WriteMetricInTauFormat(TAU_RSS, rank, numroutines, all_context_data_ptr, TAU_NO_COUNTER_ID);
          WriteMetricInTauFormat(TAU_PAGEFAULTS, rank, numroutines, all_context_data_ptr, TAU_NO_COUNTER_ID);
          /* WriteMetricInTauFormat(TAU_PROCMEM, rank, numroutines, all_context_data_ptr);
          */
        }

        if (header_ptr->profile_counters_enabled)
        { /* countertrace flag turns on tracking flops and processor time */
          for(i = 0; i < header_ptr->numcounters; i++)
          { 
            WriteMetricInTauFormat(TAU_COUNTERS, rank, numroutines, all_context_data_ptr, i);
          }
          WriteMetricInTauFormat(TAU_VIRTUALTIME, rank, numroutines, all_context_data_ptr, TAU_NO_COUNTER_ID);
        }


      all_context_rank_ptr = all_context_rank_ptr->next_rank;
      rank++;
    } /* rank */
    all_context_cycle_ptr = all_context_cycle_ptr->next_cycle;
  } /* cycle */

  return 0;
}

