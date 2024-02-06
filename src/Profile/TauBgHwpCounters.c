/* We use the IBM Blue Gene/P UPC counters */
/* Acknowledgements: Scott Parker, ALCF */

#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <mpi.h>

#ifdef TAU_BGP

#include <TAU.h>

#ifdef TAU_GNU
/* *CWL* - ULLONG_MAX relies on c99 in limits.h. 
           Intrepid's GNU implementation does not appear to support 
	   this version. The best option appears to be explicitly defining
	   the value. Redefining __STDC_VERSION__ is not the way to go and is
           added here only as a note.
*/
/* #define __STDC_VERSION__ 199901L */
/* #define ULLONG_MAX 18446744073709551615 */
#define ULLONG_MAX 18446744073709551615ULL
#endif /* TAU_GNU */

#ifndef COUNTER_REDUCE
#define COUNTER_REDUCE 1
#endif /* COUNTER_REDUCE */

#include "spi/UPC.h"
#include "spi/kernel_interface.h"
#include "Profile/TauBgHwpCounters.h"

// define UPC_CONTROL_CORE as core zero since there is always a process on that core
#define COUNTER_CONTROL_CORE 0
#define ROOT_RANK 0

void Tau_Bg_hwp_counters_start(int *error) {
    *error = 0;

    // needs to be called by all processes on a node to initialize process variables and shared variables
    BGP_UPC_Initialize();

    // determine the core the process is running on
    uint32_t coreId = Kernel_PhysicalProcessorID();

    // only UPC_CONTROL_CORE configures and starts the shared counters for the node
    if (coreId == COUNTER_CONTROL_CORE) {
      int32_t configGenNum = BGP_UPC_Initialize_Counter_Config(BGP_UPC_MODE_DEFAULT, BGP_UPC_CFG_EDGE_DEFAULT);
      if (configGenNum < 0) {
        *error = COUNTER_CONFIGURATION_ERROR;       
        return;
      }

      int32_t startGenNum = BGP_UPC_Start(0);
      if (startGenNum < 0) {
        *error = COUNTER_START_ERROR;
        return;
      }
    }
   
    return;
}


void Tau_Bg_hwp_counters_stop(int* numCounters, uint64_t counters[], int* mode, int *error) {
    *numCounters = 0;
    *mode = COUNTER_MODE_NOCORES;
    *error = 0;

    // determine the core the process is running on
    uint32_t coreId = Kernel_PhysicalProcessorID();

    // stop counters if active and on control core
    if (coreId == COUNTER_CONTROL_CORE) {
      int32_t active = BGP_UPC_Check_Active();

      if (active == 1) {
        BGP_UPC_Stop();

        char configBuffer[BGP_UPC_MINIMUM_LENGTH_READ_COUNTERS_STRUCTURE];
        int32_t startGenNum = BGP_UPC_Read_Counter_Values(configBuffer,
                                 BGP_UPC_MINIMUM_LENGTH_READ_COUNTERS_STRUCTURE, BGP_UPC_READ_EXCLUSIVE);
        if (startGenNum < 0) {
          *mode = COUNTER_MODE_INVALID;
          *error = COUNTER_READ_ERROR;
          return;
        }

        BGP_UPC_Read_Counters_Struct_t *config = (BGP_UPC_Read_Counters_Struct_t*) configBuffer;
        if (config->mode == BGP_UPC_MODE_0) {
          *mode = COUNTER_MODE_CORES01;
        } else if (config->mode == BGP_UPC_MODE_1) {
          *mode = COUNTER_MODE_CORES23;
        } else {
          *mode = COUNTER_MODE_INVALID;
          *error = COUNTER_CONFIGURATION_ERROR;
          return;
        }

        startGenNum = BGP_UPC_Read_Counters(counters, BGP_UPC_MAXIMUM_LENGTH_READ_COUNTERS_ONLY, 
                                            BGP_UPC_READ_EXCLUSIVE);
        if (startGenNum < 0) {
          *error = COUNTER_READ_ERROR;
          return;
        }

        *numCounters = 256;
      } else {
        *error = COUNTER_INACTIVE_ERROR;
      }
    }

    return;
}

void Tau_Bg_hwp_counters_output(int* numCounters, uint64_t counters[], int* mode, int* error) {

    int rank;
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char metadata_string[1024];

    #ifdef COUNTER_REDUCE
      int reduceFlag = 1;
    #else
      int reduceFlag = 0;
    #endif 

    if (reduceFlag) {
      int core01Count = 0, core23Count = 0;
      int core01Flag = 0, core23Flag = 0;

      if (*mode == COUNTER_MODE_CORES01) {
        core01Flag = 1;
      } else if (*mode == COUNTER_MODE_CORES23) {
        core23Flag = 1;
      } else if (*mode == COUNTER_MODE_NOCORES) {
        ;
      } else {
        *error = COUNTER_CONFIGURATION_ERROR;
      } 

      int err;
      err = PMPI_Reduce(&core01Flag, &core01Count, 1, MPI_INT, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);
      if (err != 0) *error = COUNTER_REDUCE_ERROR;
      err = PMPI_Reduce(&core23Flag, &core23Count, 1, MPI_INT, MPI_SUM, ROOT_RANK, MPI_COMM_WORLD);
      if (err != 0) *error = COUNTER_REDUCE_ERROR;

      int i;
      uint64_t zero[COUNTER_ARRAY_LENGTH];
      for (i= 0; i < COUNTER_ARRAY_LENGTH; i++) {
        zero[i] = 0ull;
      }

      uint64_t max[COUNTER_ARRAY_LENGTH];
      for (i= 0; i < COUNTER_ARRAY_LENGTH; i++) {
        max[i] = ULLONG_MAX;
      }

      uint64_t core01Min[COUNTER_ARRAY_LENGTH], core01Av[COUNTER_ARRAY_LENGTH], core01Max[COUNTER_ARRAY_LENGTH];
      uint64_t core23Min[COUNTER_ARRAY_LENGTH], core23Av[COUNTER_ARRAY_LENGTH], core23Max[COUNTER_ARRAY_LENGTH];
 
      if (*mode == COUNTER_MODE_CORES01) {
        err = PMPI_Reduce(counters, core01Min, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_MIN, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;
        err = PMPI_Reduce(counters, core01Av, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_SUM, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;
        err = PMPI_Reduce(counters, core01Max, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_MAX, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;
  
        err = PMPI_Reduce(max, core23Min, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_MIN, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;
        err = PMPI_Reduce(zero, core23Av, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_SUM, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;
        err = PMPI_Reduce(zero, core23Max, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_MAX, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;
      } else if (*mode == COUNTER_MODE_CORES23) {
        err = PMPI_Reduce(max, core01Min, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_MIN, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;
        err = PMPI_Reduce(zero, core01Av, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_SUM, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;
        err = PMPI_Reduce(zero, core01Max, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_MAX, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;

        err = PMPI_Reduce(counters, core23Min, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_MIN, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;
        err = PMPI_Reduce(counters, core23Av, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_SUM, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;
        err = PMPI_Reduce(counters, core23Max, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_MAX, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;
      } else {
        err = PMPI_Reduce(max, core01Min, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_MIN, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;
        err = PMPI_Reduce(zero, core01Av, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_SUM, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;
        err = PMPI_Reduce(zero, core01Max, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_MAX, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;

        err = PMPI_Reduce(max, core23Min, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_MIN, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;
        err = PMPI_Reduce(zero, core23Av, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_SUM, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;
        err = PMPI_Reduce(zero, core23Max, COUNTER_ARRAY_LENGTH, MPI_UNSIGNED_LONG_LONG, MPI_MAX, ROOT_RANK,
                          MPI_COMM_WORLD);
        if (err != 0) *error = COUNTER_REDUCE_ERROR;
      }

      if (rank == ROOT_RANK) {
#ifdef TAU_WRITE_COUNTER_FILE 
        FILE *outFile = fopen(QUOTE(OUTPUT_FILE), "w");
        if (outFile == NULL) {
          *error = COUNTER_IO_ERROR;
          return;
        }

        fprintf(outFile, "core01Count = %i\n", core01Count);
        fprintf(outFile, "core23Count = %i\n\n", core23Count);
#endif /* TAU_WRITE_COUNTER_FILE */

        char eventName[BGP_UPC_MAXIMUM_LENGTH_EVENT_NAME];
        for (i= 0; i < COUNTER_ARRAY_LENGTH; i++) {
          BGP_UPC_Get_Event_Name(i, BGP_UPC_MAXIMUM_LENGTH_EVENT_NAME, eventName);
#ifdef TAU_WRITE_COUNTER_FILE 
          fprintf(outFile, "%3i  %-56s %15llu %15llu %15llu\n", i, eventName, core01Min[i], core01Max[i],
                            core01Av[i]/core01Count);
#endif /* TAU_WRITE_COUNTER_FILE */
          snprintf(metadata_string, sizeof(metadata_string), "%15llu %15llu %15llu",  core01Min[i], core01Max[i],
                            core01Av[i]/core01Count);
	  TAU_METADATA(eventName, metadata_string);

        }
     
        for (i= 0; i < COUNTER_ARRAY_LENGTH; i++) {
          BGP_UPC_Get_Event_Name(COUNTER_ARRAY_LENGTH+i, BGP_UPC_MAXIMUM_LENGTH_EVENT_NAME, eventName);
#ifdef TAU_WRITE_COUNTER_FILE 
          fprintf(outFile, "%3i  %-56s %15llu %15llu %15llu\n", COUNTER_ARRAY_LENGTH+i, eventName, core23Min[i],
                  core23Max[i], core23Av[i]/core23Count);
#endif /* TAU_WRITE_COUNTER_FILE */
          snprintf(metadata_string, sizeof(metadata_string), "%15llu %15llu %15llu", core23Min[i],
                  core23Max[i], core23Av[i]/core23Count);
	  TAU_METADATA(eventName, metadata_string);
        }

#ifdef TAU_WRITE_COUNTER_FILE 
        fclose(outFile);
#endif /* TAU_WRITE_COUNTER_FILE */
      }
    } else {
      if (*mode == COUNTER_MODE_CORES01 || *mode == COUNTER_MODE_CORES23) {
        char fileName[128];
        snprintf(fileName, sizeof(fileName),  "%s-%06i", QUOTE(OUTPUT_FILE), rank);
       
        FILE *outFile = fopen(fileName, "w");
        if (outFile == NULL) {
          fprintf(stderr, "** ERROR: error opening file %s\n", fileName);
        } else {

          char eventName[BGP_UPC_MAXIMUM_LENGTH_EVENT_NAME];
          int i;
          if (*mode == COUNTER_MODE_CORES01) {
            for (i= 0; i < COUNTER_ARRAY_LENGTH; i++) {
              BGP_UPC_Get_Event_Name(i, BGP_UPC_MAXIMUM_LENGTH_EVENT_NAME, eventName);
#ifdef TAU_WRITE_COUNTER_FILE 
              fprintf(outFile, "%3i  %-56s %15llu\n", i, eventName, counters[i]);
#endif /* TAU_WRITE_COUNTER_FILE */
            }
          } else if (*mode == COUNTER_MODE_CORES23) {
            for (i= 0; i < COUNTER_ARRAY_LENGTH; i++) {
              BGP_UPC_Get_Event_Name(COUNTER_ARRAY_LENGTH+i, BGP_UPC_MAXIMUM_LENGTH_EVENT_NAME, eventName);
#ifdef TAU_WRITE_COUNTER_FILE 
              fprintf(outFile, "%3i  %-56s %15llu\n", COUNTER_ARRAY_LENGTH+i, eventName, counters[i]);
#endif /* TAU_WRITE_COUNTER_FILE */
            }
          }

#ifdef TAU_WRITE_COUNTER_FILE 
          fclose(outFile);
#endif /* TAU_WRITE_COUNTER_FILE */
        }
      }
    }


    return;
}
#endif /* TAU_BGP */
