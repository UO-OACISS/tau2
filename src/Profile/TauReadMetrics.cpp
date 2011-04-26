/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.cs.uoregon.edu/research/tau             **
*****************************************************************************
**    Copyright 2009                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**      File            : TauReadMetrics.cpp                               **
**      Description     : TAU Profiling Package                            **
**      Contact         : tau-bugs@cs.uoregon.edu                          **
**      Documentation   : See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : Handles Reading Metrics                          **
**                                                                         **
****************************************************************************/

#include <TAU.h>

/* for getrusage */
#ifndef TAU_WINDOWS
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#endif /* TAU_WINDOWS */

#ifdef TAU_BGL
/* header files for BlueGene/L */
#include <bglpersonality.h>
#include <rts.h>
#endif /* TAU_BGL */

#ifdef TAU_BGP
#  ifdef BGP_TIMERS
/* header files for BlueGene/P */
#    include <bgp_personality.h>
#    include <bgp_personality_inlines.h>
#    include <kernel_interface.h>
#  endif /* TAU_BGPTIMERS */
#endif /* TAU_BGP */

#ifdef TAU_PAPI
extern "C" {
#include "papi.h"
}
#endif /* TAU_PAPI */

#include "Profile/CuptiLayer.h"

#ifdef TAUKTAU_SHCTR
#include "Profile/KtauCounters.h"
#endif //TAUKTAU_SHCTR

#ifdef TAU_MPI
extern TauUserEvent *TheSendEvent(void);
extern TauUserEvent *TheRecvEvent(void);
extern TauUserEvent *TheBcastEvent(void);
extern TauUserEvent *TheReduceEvent(void);
extern TauUserEvent *TheReduceScatterEvent(void);
extern TauUserEvent *TheScanEvent(void);
extern TauUserEvent *TheAllReduceEvent(void);
extern TauUserEvent *TheAlltoallEvent(void);
extern TauUserEvent *TheScatterEvent(void);
extern TauUserEvent *TheGatherEvent(void);
extern TauUserEvent *TheAllgatherEvent(void);
#endif /* TAU_MPI */



/* null clock that always returns 0 */
void metric_read_nullClock(int tid, int idx, double values[]) {
  values[idx] = 0;
}

/* logical clock that just increments on each request */
void metric_read_logicalClock(int tid, int idx, double values[]) {
  static long long value = 0;
  values[idx] = value++;
}

#ifdef TAU_LINUX_TIMERS
extern "C" double TauGetMHzRatings(void);

extern "C" unsigned long long getLinuxHighResolutionTscCounter(void);
#endif

void metric_read_linuxtimers(int tid, int idx, double values[]) {
#ifdef TAU_LINUX_TIMERS
  static double ratings = TauGetMHzRatings();
  values[idx] = (double)getLinuxHighResolutionTscCounter() / ratings;
#endif
}

double TauWindowsUsecD(void);


/* user defined clock */

static double userClock[TAU_MAX_THREADS];

void metric_write_userClock(int tid, double value) {
  userClock[tid] = value;
}

void metric_read_userClock(int tid, int idx, double values[]) {
  values[idx] = userClock[tid];
}


/* clock that uses gettimeofday */
void metric_read_gettimeofday(int tid, int idx, double values[]) {
#ifdef TAU_WINDOWS
  values[idx] = TauWindowsUsecD();
#else
  struct timeval tp;
  gettimeofday(&tp, 0);
  values[idx] = ((double)tp.tv_sec * 1e6 + tp.tv_usec);
#endif
}

/* bgl/bgp timers */
void metric_read_bgtimers(int tid, int idx, double values[]) {
#ifdef TAU_BGL
  static double bgl_clockspeed = 0.0;

  if (bgl_clockspeed == 0.0) {
    BGLPersonality mybgl;
    rts_get_personality(&mybgl, sizeof(BGLPersonality));
    bgl_clockspeed = 1.0e6 / (double)BGLPersonality_clockHz(&mybgl);
  }
  values[idx] = (rts_get_timebase() * bgl_clockspeed);
#endif /* TAU_BGL */

#ifdef TAU_BGP
#ifdef BGP_TIMERS
  static double bgp_clockspeed = 0.0;

  if (bgp_clockspeed == 0.0) {
    _BGP_Personality_t mybgp;
    Kernel_GetPersonality(&mybgp, sizeof(_BGP_Personality_t));
    bgp_clockspeed = 1.0 / (double)BGP_Personality_clockMHz(&mybgp);
  }
  values[idx] =  (_bgp_GetTimeBase() * bgp_clockspeed);
#else /* TAU_BGPTIMERS */
  printf("TAU: Error: You must specify -BGPTIMERS at configure time\n");
  values[idx] = 0;
#endif /* TAU_BGPTIMERS */
#endif /* TAU_BGP */
}

/* cray timers */
void metric_read_craytimers(int tid, int idx, double values[]) {
#ifdef  CRAY_TIMERS
#ifdef TAU_CATAMOUNT /* for Cray XT3 */
  values[idx] = dclock() * 1.0e6;
#else /* for Cray X1 */
  long long tick = _rtc();
  values[idx] = (double)tick / HZ;
#endif /* TAU_CATAMOUNT */
#endif /* CRAY_TIMERS */
}

/* cputime from getrusage */
void metric_read_cputime(int tid, int idx, double values[]) {
#ifndef TAU_WINDOWS
  struct rusage current_usage;
  getrusage(RUSAGE_SELF, &current_usage);
  values[idx] = (current_usage.ru_utime.tv_sec + current_usage.ru_stime.tv_sec) * 1e6
                + (current_usage.ru_utime.tv_usec + current_usage.ru_stime.tv_usec);
#endif
}

/* message size "timer" */
void metric_read_messagesize(int tid, int idx, double values[]) {
#ifdef TAU_MPI
  values[idx] = TheSendEvent()->GetSumValue(tid)
                + TheRecvEvent()->GetSumValue(tid)
                + TheBcastEvent()->GetSumValue(tid)
                + TheReduceEvent()->GetSumValue(tid)
                + TheReduceScatterEvent()->GetSumValue(tid)
                + TheScanEvent()->GetSumValue(tid)
                + TheAllReduceEvent()->GetSumValue(tid)
                + TheAlltoallEvent()->GetSumValue(tid)
                + TheScatterEvent()->GetSumValue(tid)
                + TheGatherEvent()->GetSumValue(tid)
                + TheAllgatherEvent()->GetSumValue(tid);
  //Currently TAU_EVENT_DATATYPE is a double.
#endif //TAU_MPI
}

/* PAPI_VIRTUAL_TIME */
void metric_read_papivirtual(int tid, int idx, double values[]) {
#ifdef TAU_PAPI
  values[idx] = PAPI_get_virt_usec();
#endif // TAU_PAPI
}

/* PAPI wallclock time */
void metric_read_papiwallclock(int tid, int idx, double values[]) {
  static long long oldvalue = 0L;
  static long long offset = 0;
  long long newvalue = 0L;
#ifdef TAU_PAPI
  newvalue = PAPI_get_real_usec();
  if (newvalue < oldvalue) {
    offset += UINT_MAX;
    DEBUGPROFMSG("WARNING: papi counter overflow. Fixed in TAU! new = "
                 << newvalue << " old = " << oldvalue << " offset = " << offset << endl;
                 );
    DEBUGPROFMSG("Returning " << newvalue + offset << endl;
                 );
  }
  oldvalue = newvalue;
  values[idx] = newvalue + offset;
#endif // TAU_PAPI
}

/* PAPI metrics */
void metric_read_papi(int tid, int idx, double values[]) {
#ifdef TAU_PAPI
  int numPapiValues;
  long long *papiValues = PapiLayer::getAllCounters(tid, &numPapiValues);

  if (papiValues) {
    for (int i = 0; i < numPapiValues; i++) {
      values[idx + i] = papiValues[i];
    }
  }
#endif /* TAU_PAPI */
}

/* KTAU metrics */
void metric_read_ktau(int tid, int idx, double values[]) {
#ifdef TAUKTAU
  extern double KTauGetMHz(void);

  int numKtauValues;
  long long *ktauValues = KtauCounters::getAllCounters(tid, &numKtauValues);

  if (ktauValues) {
    for (int i = 0; i < numKtauValues; i++) {
      //sometimes due to double-precision issues the below
      //division can result in very small negative exclusive
      //times. Currently there is no fix implemented for this.
      //The thing to do maybe is to add a check in Profiler.cpp
      //to make sure no negative values are set.
      if (KtauCounters::counterType[i] != KTAU_SHCTR_TYPE_NUM) {
        values[idx + i] = ktauValues[i] / KTauGetMHz();
      } else {
        values[idx + i] = ktauValues[i];
      }
    }
  }

#endif
}

#define CPU_THREAD 0

double gpu_timestamp[TAU_MAX_THREADS];

void metric_set_gpu_timestamp(int tid, double value)
{
	gpu_timestamp[tid] = value;
}

void metric_read_cudatime(int tid, int idx, double values[]) {

  //get time from the CPU clock
  if (tid == CPU_THREAD)
  { 
    struct timeval tp;
    gettimeofday(&tp, 0);
    values[idx] = ((double)tp.tv_sec * 1e6 + tp.tv_usec);
  }
  // get time from the callback API 
  else
  {
    values[idx] = gpu_timestamp[tid];
  }
}

void metric_read_cupti(int tid, int idx, double values[])
{

	//printf("is the cupti layer is initialized? %d\n", Tau_CuptiLayer_is_initialized());
	if (Tau_CuptiLayer_is_initialized())
	{
		uint64_t* counterDataBuffer = (uint64_t*) malloc
			(Tau_CuptiLayer_get_num_events()*sizeof(uint64_t));
		Tau_CuptiLayer_read_counters(counterDataBuffer);

		if (counterDataBuffer)
		{
			for (int i=0; i<Tau_CuptiLayer_get_num_events(); i++)
			{
				values[idx + i] = (double) counterDataBuffer[i];
				//printf("cupti value %d is: %lf.\n", i, values[idx + i]);
			}
		}
		free(counterDataBuffer);
	}
	else
	{
		for (int i=0; i<Tau_CuptiLayer_get_num_events(); i++)
		{
			values[idx + i] = 0;
		}
	}
}
