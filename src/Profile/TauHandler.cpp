/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/paracomp/tau    **
*****************************************************************************
**    Copyright 2004  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/***************************************************************************
**	File 		: TauHandler.cpp				  **
**	Description 	: TAU Profiling Package				  **
**	Author		: Sameer Shende					  **
**	Contact		: sameer@cs.uoregon.edu sameer@acl.lanl.gov 	  **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files
//////////////////////////////////////////////////////////////////////

#ifndef TAU_WINDOWS
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#endif
#include <fcntl.h>
#include <errno.h>

#include <signal.h>
#include <Profile/Profiler.h>
#include <Profile/TauMemory.h>

#include <Profile/TauPluginInternals.h>

extern "C" int Tau_track_mpi_t_here(void);
//////////////////////////////////////////////////////////////////////
// Routines
//////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////
// Is TAU tracking memory events? Set to true/false.
//////////////////////////////////////////////////////////////////////
bool& TheIsTauTrackingMemory(void) {
  static bool isit = false; /* TAU is not tracking memory */
  return isit;
}

/////////////////////////////////////////////////////////////////////
// Is TAU tracking memory resident set size and high water mark
// from /proc/self/status?
//////////////////////////////////////////////////////////////////////
bool& TheIsTauTrackingMemoryRSSandHWM(void) {
  static bool isit = false; /* TAU is not tracking memory */
  return isit;
}

//////////////////////////////////////////////////////////////////////
// Is TAU tracking memory headroom events? Set to true/false.
//////////////////////////////////////////////////////////////////////
bool& TheIsTauTrackingMemoryHeadroom(void) {
  static bool isit = false; /* TAU is not tracking memory headroom */
  return isit;
}

//////////////////////////////////////////////////////////////////////
// Is TAU tracking power events? Set to true/false.
//////////////////////////////////////////////////////////////////////
bool& TheIsTauTrackingPower(void) {
  static bool isit = false; /* TAU is not tracking power */
  return isit;
}

//////////////////////////////////////////////////////////////////////
// Is TAU tracking system load events? Set to true/false.
//////////////////////////////////////////////////////////////////////
bool& TheIsTauTrackingLoad(void) {
  static bool isit = false; /* TAU is not tracking load */
  return isit;
}

//////////////////////////////////////////////////////////////////////
// Start tracking memory
//////////////////////////////////////////////////////////////////////
int TauEnableTrackingMemory(void) {
  // Set tracking to true
  TheIsTauTrackingMemory() = true;
  return 1;
}

//////////////////////////////////////////////////////////////////////
// Start tracking memory
//////////////////////////////////////////////////////////////////////
int TauEnableTrackingMemoryHeadroom(void) {
  // Set tracking to true
  TheIsTauTrackingMemoryHeadroom() = true;
  return 1;
}

//////////////////////////////////////////////////////////////////////
// Start tracking power
//////////////////////////////////////////////////////////////////////
int TauEnableTrackingPower(void) {
  // Set tracking to true
  TheIsTauTrackingPower() = true;
  return 1;
}

//////////////////////////////////////////////////////////////////////
// Start tracking load
//////////////////////////////////////////////////////////////////////
int TauEnableTrackingLoad(void) {
  // Set tracking to true
  TheIsTauTrackingLoad() = true;
  return 1;
}

//////////////////////////////////////////////////////////////////////
// Start tracking memory rss and hwm
//////////////////////////////////////////////////////////////////////
int TauEnableTrackingMemoryRSSandHWM(void) {
  // Set tracking to true
  TheIsTauTrackingMemoryRSSandHWM() = true;
  return 1;
}

//////////////////////////////////////////////////////////////////////
// Stop tracking memory
//////////////////////////////////////////////////////////////////////
int TauDisableTrackingMemory(void) {
  TheIsTauTrackingMemory() = false;
  return 0;
}

//////////////////////////////////////////////////////////////////////
// Stop tracking memory RSS and HWM
//////////////////////////////////////////////////////////////////////
int TauDisableTrackingMemoryRSSandHWM(void) {
  TheIsTauTrackingMemoryRSSandHWM() = false;
  return 0;
}

//////////////////////////////////////////////////////////////////////
// Stop tracking memory headroom
//////////////////////////////////////////////////////////////////////
int TauDisableTrackingMemoryHeadroom(void) {
  TheIsTauTrackingMemoryHeadroom() = false;
  return 0;
}

//////////////////////////////////////////////////////////////////////
// Stop tracking power
//////////////////////////////////////////////////////////////////////
int TauDisableTrackingPower(void) {
  // Set tracking to false
  TheIsTauTrackingPower() = false;
  return 0;
}

//////////////////////////////////////////////////////////////////////
// Stop tracking load
//////////////////////////////////////////////////////////////////////
int TauDisableTrackingLoad(void) {
  // Set tracking to false
  TheIsTauTrackingLoad() = false;
  return 0;
}

//////////////////////////////////////////////////////////////////////
// Set interrupt interval
//////////////////////////////////////////////////////////////////////
int& TheTauInterruptInterval(void) {
  static int interval = TauEnv_get_interval(); /* interrupt every 10 seconds */
  return interval;
}

//////////////////////////////////////////////////////////////////////
// Set interrupt interval
//////////////////////////////////////////////////////////////////////
void TauSetInterruptInterval(int interval) {
  /* Set the interval */
  TheTauInterruptInterval() = interval;
}

int Tau_read_cray_power_events(int fd, long long int *value)  {
  char buf[2048];
  int ret, bytesread;
  if (fd > 0) {
    ret = lseek(fd, 0, SEEK_SET);
    if (ret < 0) {
      perror("lseek failure:");
      *value = 0;
      return ret;
    }
  }
  else {
    *value = 0;
    return -1;
  }
  bytesread = read(fd, buf, 2048);
  if (bytesread == -1) {
    perror("Error reading from Cray power events");
    return bytesread;
  }
  ret = sscanf(buf, "%lld", value);
  return ret;
}

int Tau_read_load_event(int fd, double *value)  {
  char buf[2048] = {0};
  int ret, bytesread;
  if (fd > 0) {
    ret = lseek(fd, 0, SEEK_SET);
    if (ret < 0) {
      perror("lseek failure:");
      *value = 0;
      return ret;
    }
  }
  else {
    *value = 0;
    return -1;
  }
  bytesread = read(fd, buf, 2048);
  if (bytesread == -1) {
    perror("Error reading from system load events");
    return bytesread;
  }
  ret = sscanf(buf, "%lf", value);
  //TAU_VERBOSE("LOAD: buf = %s Value = %g\n", buf, *value);
  return ret;
}

int Tau_open_system_file(const char *filename) {

  int fd = open(filename, O_RDONLY);
  return fd;
}

void TauTriggerCrayPowerEvent(int fd, const char *event_name)  {
  long long int value;
  int tid = 0;
  if (TauEnv_get_thread_per_gpu_stream()) {
    tid = RtsLayer::myThread();
  }
  if (fd) {
    Tau_read_cray_power_events(fd, &value);
    if (value > 0) {
      Tau_trigger_context_event_thread(event_name, (double) value, tid);
      TAU_VERBOSE("Triggered %s with %lld\n", event_name, value);
    }
  }
}

void TauTriggerCrayPowerEvents(const char * prefix = nullptr) {
/*
  static int power_fd=Tau_open_cray_file("/sys/cray/pm_counters/power");
  static int accel_power_fd=Tau_open_cray_file("/sys/cray/pm_counters/accel_power");
  static int accel_energy_fd=Tau_open_cray_file("/sys/cray/pm_counters/accel_energy");
  static int energy_fd=Tau_open_cray_file("/sys/cray/pm_counters/energy");

  // this does not take into account the freshness file
  TauTriggerCrayPowerEvent(power_fd, "Node Power (in Watts)");
  TauTriggerCrayPowerEvent(accel_power_fd, "Accelerator Device Power (in Watts)");
  TauTriggerCrayPowerEvent(energy_fd, "Node Energy (in Joules)");
  TauTriggerCrayPowerEvent(accel_energy_fd, "Accel Energy (in Joules)");
*/
}

void TauTriggerPowerEvent(bool in_signal_handler, const char * prefix = nullptr) {
  //printf("Inside TauTriggerPowerEvent\n");
#ifdef TAU_CRAYCNL
  TauTriggerCrayPowerEvents(prefix);
#else
#ifdef TAU_PAPI_PERF_RAPL
  PapiLayer::triggerRAPLPowerEvents(in_signal_handler);
#endif /* TAU_PAPI */
#endif /* TAU_CRAYCNL */
}

//////////////////////////////////////////////////////////////////////
// TAU's trigger load
//////////////////////////////////////////////////////////////////////

/* Assume the first call comes from initialization, not the handler.
 * This function is used only once, to initialize the user event,
 * so that we don't have to allocate memory during the signal
 * handler. */
void * TauTriggerLoadEvent_helper(const char * prefix) {
  std::string tmpstr{prefix == nullptr ? "" : prefix};
  if (TauEnv_get_tracing()) {
    tmpstr += "System load (x100)";
    return Tau_get_userevent(tmpstr.c_str());
  } else {
    tmpstr += "System load";
    return Tau_get_userevent(tmpstr.c_str());
  }
}

void TauTriggerLoadEvent(bool use_context, const char * prefix = nullptr) {
  double value;
  /* maintain this from call-to-call so we don't have to
   * allocate during a signal handler */
  static void *ue = TauTriggerLoadEvent_helper(prefix);
  static int fd = Tau_open_system_file("/proc/loadavg");
  int tid = 0;
  if (TauEnv_get_thread_per_gpu_stream()) {
    tid = RtsLayer::myThread();
  }
  if (fd) {
    Tau_read_load_event(fd, &value);
    //Do not bother with recording the load if TAU is uninitialized.
    if (Tau_init_check_initialized() && TheSafeToDumpData()) {
      if (TauEnv_get_tracing()) {
          if(use_context) {
            Tau_trigger_context_event_thread("System load (x100)", value*100, tid);
          } else {
            Tau_userevent_thread(ue, value*100, tid);
          }
      }
      else  {
          if(use_context) {
            Tau_trigger_context_event_thread("System load", value, tid);
          } else {
            Tau_userevent_thread(ue, value, tid);
          }
      }
    }
  }
}

extern "C" int Tau_trigger_memory_rss_hwm(bool use_context, const char * prefix = nullptr);
//////////////////////////////////////////////////////////////////////
// TAU's alarm signal handler
//////////////////////////////////////////////////////////////////////
void TauAlarmHandler(int signum) {
    //printf("In %s\n", __func__);


   /* Check and see if we're tracking memory events */
  // these are never safe!  It's not signal safe...
/*
  if (TheIsTauTrackingMemory()) {
      TauAllocation::TriggerHeapMemoryUsageEvent();
  }

  if (TheIsTauTrackingMemoryHeadroom()) {
    TauAllocation::TriggerMemoryHeadroomEvent();
  }
*/

  if (TheIsTauTrackingPower()) {
    // yes, we are in a signal handler
    TauTriggerPowerEvent(true);
  }

  if (TheIsTauTrackingLoad()) {
    // no, don't use the context (not signal safe)
    TauTriggerLoadEvent(false);
  }

  if (TauEnv_get_track_mpi_t_pvars()) {
    Tau_track_mpi_t_here();
  }

  if (TheIsTauTrackingMemoryRSSandHWM()) {
    // no, don't use the context (not signal safe)
    Tau_trigger_memory_rss_hwm(false);
  }

  /* Set alarm for the next interrupt */
#ifndef TAU_WINDOWS
  alarm(TheTauInterruptInterval());
#endif
  /*Invoke plugins only if both plugin path and plugins are specified*/
  if(Tau_plugins_enabled.interrupt_trigger) {
    Tau_plugin_event_interrupt_trigger_data_t plugin_data;
    plugin_data.signum = signum;
    plugin_data.tid = Tau_get_thread();
    Tau_util_invoke_callbacks(TAU_PLUGIN_EVENT_INTERRUPT_TRIGGER, "*", &plugin_data);
  }
}

void TauSetupHandler(void) {
    /* set up the ONE timer that will execute the handler for all
     * the triggered counters.  ONE. */
#ifndef TAU_WINDOWS
  struct sigaction new_action, old_action;
  memset(&old_action, 0, sizeof(struct sigaction));
  memset(&new_action, 0, sizeof(struct sigaction));

  // call the handler once, at startup.  This will pre-allocate some
  // necessary data structures for us, so they don't have to be created
  // during the signal processing.
  TauAlarmHandler(SIGINT);

  // set signal handler
  new_action.sa_handler = TauAlarmHandler;

  new_action.sa_flags = 0;
  sigaction(SIGALRM, NULL, &old_action);
  if (old_action.sa_handler != SIG_IGN) {
    /* by default it is set to ignore */
    sigaction(SIGALRM, &new_action, NULL);
  }

  /* activate alarm */
  alarm(TheTauInterruptInterval());
#endif
}

//////////////////////////////////////////////////////////////////////
// Track Memory
//////////////////////////////////////////////////////////////////////
void TauTrackMemoryUtilization(bool allocated) {
//////////////////////////////////////////////////////////////////////
// Argument: allocated. TauTrackMemoryUtilization can keep track of memory
// allocated or memory free (headroom to grow). Accordingly, it is true
// for tracking memory allocated, and false to check the headroom
//////////////////////////////////////////////////////////////////////
  // Are we tracking memory or headroom. Check the allocated argument.
  if (allocated) {
    TheIsTauTrackingMemory() = true;
  } else {
    TheIsTauTrackingMemoryHeadroom() = true;
  }
}

//////////////////////////////////////////////////////////////////////
// Track Power
//////////////////////////////////////////////////////////////////////
void TauTrackPower(const char * prefix) {
  /* Enable tracking power by default */
  static int flag = TauEnableTrackingPower();
  // use the variable to prevent compiler complaints
  if (!flag) {};

  /* Check and see if we're *still* tracking memory events */
  if (TheIsTauTrackingPower()) {
    // not in a signal handler, but don't use context
    TauTriggerPowerEvent(true, prefix);
  }
}

//////////////////////////////////////////////////////////////////////
// Track Load
//////////////////////////////////////////////////////////////////////
void TauTrackLoad(const char * prefix) {
  /* Enable tracking power by default */
  static int flag = TauEnableTrackingLoad();
  // use the variable to prevent compiler complaints
  if (!flag) {};

  /* Check and see if we're *still* tracking memory events */
  if (TheIsTauTrackingLoad()) {
    // don't use context
    TauTriggerLoadEvent(false, prefix);
  }
}
//////////////////////////////////////////////////////////////////////
// Track MPI_T
//////////////////////////////////////////////////////////////////////
extern "C" void Tau_track_mpi_t(void) {
    // nothing to do?
}


//////////////////////////////////////////////////////////////////////
// Track memory resident set size (RSS) and high water mark (hwm)
//////////////////////////////////////////////////////////////////////
void TauTrackMemoryFootPrint(const char * prefix) {
  /* Enable tracking memory by default */
  static int flag = TauEnableTrackingMemoryRSSandHWM();
  // use the variable to prevent compiler complaints
  if (!flag) {};

  /* Check and see if we're *still* tracking memory events */
  if (TheIsTauTrackingMemoryRSSandHWM()) {
    Tau_trigger_memory_rss_hwm(false, prefix);
  }
}

//////////////////////////////////////////////////////////////////////
// Track Memory events at this location in the source code
//////////////////////////////////////////////////////////////////////
void TauTrackMemoryFootPrintHere(void) {
  /* Enable tracking memory by default */
  static int flag = TauEnableTrackingMemoryRSSandHWM();
  // use the variable to prevent compiler complaints
  if (!flag) {};

  /* Check and see if we're *still* tracking memory events */
  if (TheIsTauTrackingMemoryRSSandHWM()) {
    Tau_trigger_memory_rss_hwm(true);
  }
}


//////////////////////////////////////////////////////////////////////
// Track Memory events at this location in the source code
//////////////////////////////////////////////////////////////////////
void TauTrackMemoryHere(const char * prefix) {
  /* Enable tracking memory by default */
  static int flag = TauEnableTrackingMemory();
  // use the variable to prevent compiler complaints
  if (!flag) {};

  /* Check and see if we're *still* tracking memory events */
  if (TheIsTauTrackingMemory()) {
    TauAllocation::TriggerHeapMemoryUsageEvent(prefix);
  }
}

//////////////////////////////////////////////////////////////////////
// Track Memory headroom events at this location in the source code
//////////////////////////////////////////////////////////////////////
void TauTrackMemoryHeadroomHere(void) {
  /* Enable tracking memory by default */
  static int flag = TauEnableTrackingMemoryHeadroom();
  // use the variable to prevent compiler complaints
  if (!flag) {};

  /* Check and see if we're *still* tracking memory events */
  if (TheIsTauTrackingMemoryHeadroom()) {
    TauAllocation::TriggerMemoryHeadroomEvent();
  }
}

//////////////////////////////////////////////////////////////////////
// Track Power events at this location in the source code
//////////////////////////////////////////////////////////////////////
void TauTrackPowerHere(void) {
  /* Enable tracking power by default */
  static int flag = TauEnableTrackingPower();
  // use the variable to prevent compiler complaints
  if (!flag) {};

  /* Check and see if we're *still* tracking memory events */
  if (TheIsTauTrackingPower()) {
    // no, not in a signal handler
    TauTriggerPowerEvent(false);
  }
}

//////////////////////////////////////////////////////////////////////
// Track Load events at this location in the source code
//////////////////////////////////////////////////////////////////////
void TauTrackLoadHere(void) {
  /* Enable tracking power by default */
  static int flag = TauEnableTrackingLoad();
  // use the variable to prevent compiler complaints
  if (!flag) {};

  /* Check and see if we're *still* tracking memory events */
  if (TheIsTauTrackingLoad()) {
    // use context
    TauTriggerLoadEvent(true);
  }
}

/***************************************************************************
 * $RCSfile: TauHandler.cpp,v $   $Author: amorris $
 * $Revision: 1.24 $   $Date: 2010/05/14 22:21:04 $
 * POOMA_VERSION_ID: $Id: TauHandler.cpp,v 1.24 2010/05/14 22:21:04 amorris Exp $
 ***************************************************************************/







