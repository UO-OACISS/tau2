/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2009  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**	File 		: TauReadMetrics.cpp		        	   **
**	Description 	: TAU Profiling Package				   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : Handles Reading Metrics                          **
**                                                                         **
****************************************************************************/


#include <TAU.h>

#ifndef TAU_WINDOWS
/* for getrusage */
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#endif /* TAU_WINDOWS */

#ifdef BGL_TIMERS
/* header files for BlueGene/L */
#include <bglpersonality.h>
#include <rts.h>
#endif // BGL_TIMERS

#ifdef BGP_TIMERS
/* header files for BlueGene/P */
#include <bgp_personality.h>
#include <bgp_personality_inlines.h>
#include <kernel_interface.h>
#endif // BGP_TIMERS


#ifdef TAU_MPI
extern TauUserEvent* TheSendEvent(void);
extern TauUserEvent* TheRecvEvent(void);
extern TauUserEvent* TheBcastEvent(void);
extern TauUserEvent* TheReduceEvent(void);
extern TauUserEvent* TheReduceScatterEvent(void);
extern TauUserEvent* TheScanEvent(void);
extern TauUserEvent* TheAllReduceEvent(void);
extern TauUserEvent* TheAlltoallEvent(void);
extern TauUserEvent* TheScatterEvent(void);
extern TauUserEvent* TheGatherEvent(void);
extern TauUserEvent* TheAllgatherEvent(void);
#endif /* TAU_MPI */



// null clock that always returns 0
void metric_read_nullClock(int tid, int idx, double values[]) {
  values[idx] = 0;
}

// logical clock that just increments on each request
void metric_read_logicalClock(int tid, int idx, double values[]) {
  static long long value = 0;
  values[idx] = value++;
}

double TauWindowsUsecD(void);

// clock that uses gettimeofday
void metric_read_gettimeofday(int tid, int idx, double values[]) {
#ifdef TAU_WINDOWS

#else
  struct timeval tp;
  gettimeofday (&tp, 0);
  values[idx] = ((double)tp.tv_sec * 1e6 + tp.tv_usec);
#endif
}


// bgl/bgp timers
void metric_read_bgtimers(int tid, int idx, double values[]) {

#ifdef TAU_BGL
  static double bgl_clockspeed = 0.0;
  
  if (bgl_clockspeed == 0.0) {
    BGLPersonality mybgl;
    rts_get_personality(&mybgl, sizeof(BGLPersonality));
    bgl_clockspeed = 1.0e6/(double)BGLPersonality_clockHz(&mybgl);
  }
  values[idx] = (rts_get_timebase() * bgl_clockspeed);
#endif /* TAU_BGL */

#ifdef TAU_BGP
  static double bgp_clockspeed = 0.0;

  if (bgp_clockspeed == 0.0) {
    _BGP_Personality_t mybgp;
    Kernel_GetPersonality(&mybgp, sizeof(_BGP_Personality_t));
    bgp_clockspeed = 1.0/(double)BGP_Personality_clockMHz(&mybgp);
  }
  values[idx] =  (_bgp_GetTimeBase() * bgp_clockspeed);
#endif /* TAU_BGP */
}


// cray timers
void metric_read_craytimers(int tid, int idx, double values[]) {
#ifdef  CRAY_TIMERS
#ifdef TAU_CATAMOUNT /* for Cray XT3 */
  values[idx] = dclock()*1.0e6;
#else /* for Cray X1 */
  long long tick = _rtc();
  values[idx] = (double) tick/HZ;
#endif /* TAU_CATAMOUNT */
#endif /* CRAY_TIMERS */
}

// cputime from getrusage
void metric_read_cputime(int tid, int idx, double values[]) {
#ifndef TAU_WINDOWS
  struct rusage current_usage;
  getrusage (RUSAGE_SELF, &current_usage);
  values[idx] = (current_usage.ru_utime.tv_sec + current_usage.ru_stime.tv_sec)* 1e6 
  + (current_usage.ru_utime.tv_usec + current_usage.ru_stime.tv_usec);
#endif
}


// message size "timer"
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
#endif//TAU_MPI
}


void metric_read_papivirtual(int tid, int idx, double values[]) {
#ifdef TAU_PAPI
  values[idx] = PAPI_get_virt_usec();
#endif // TAU_PAPI
}


void metric_read_papiwallclock(int tid, int idx, double values[]) {

static long long oldvalue = 0L;
static long long offset = 0;
long long newvalue = 0L;
#ifdef TAU_PAPI
  newvalue = PAPI_get_real_usec();
  if (newvalue < oldvalue) {
    offset += UINT_MAX;
    DEBUGPROFMSG("WARNING: papi counter overflow. Fixed in TAU! new = "
	 <<newvalue <<" old = " <<oldvalue<<" offset = "<<offset <<endl;);
    DEBUGPROFMSG("Returning "<<newvalue + offset<<endl;);
  }
  oldvalue = newvalue;
  values[idx] = newvalue + offset; 
#endif // TAU_PAPI
}


void metric_read_papi(int tid, int idx, double values[]) {
#ifdef TAU_PAPI
  int numPapiValues;
  long long *papiValues = PapiLayer::getAllCounters(tid, &numPapiValues);


  if (papiValues) {
    for(int i=0; i<numPapiValues; i++) {
      values[idx+i] = papiValues[i];
    }
  }
#endif /* TAU_PAPI */
}

