/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/

#include <TAU.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <thread>
#include <fstream>
#include <string>

#include <Profile/TauEnv.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauCollate.h>
#include <Profile/TauUtil.h>
#include <Profile/TauXML.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <signal.h>
#include <semaphore.h>

#ifdef TAU_MPI
#include <mpi.h>

#include <list>
#include <vector>

#include <Profile/TauPlugin.h>
#include <Profile/TauMemory.h>

#include <Profile/TauTrace.h>
#include <sys/resource.h>

#include <sys/types.h>
#include <unistd.h>

using namespace std;
std::vector<std::thread> thread_vec;
extern "C" int Tau_open_system_file(const char *filename);
extern "C" int Tau_read_load_event(int fd, double *value);

int done = 0; 

int Tau_plugin_event_pre_end_of_execution(Tau_plugin_event_pre_end_of_execution_data_t *data) {

  done = 1;

  for(auto it = thread_vec.begin(); it != thread_vec.end(); it++) 
    it->join();

  fprintf(stderr, "Asynchronous plugin exiting...\n");

  return 0;
}

void Tau_plugin_do_work(void * data) {
  double value = 0;
  static int fd = Tau_open_system_file("/proc/loadavg");

  while(!done) {
      value = 0;
      if (fd) {
        Tau_read_load_event(fd, &value);
    
       //Do not bother with recording the load if TAU is uninitialized. 
        if (Tau_init_check_initialized()) {
            value = value*100;
        } else {
          value = 0;
        }
      }
      struct rusage r_usage;
      getrusage(RUSAGE_SELF,&r_usage);
      fprintf(stderr, "Load and Max Memory usage = %lf, %ld\n", value, r_usage.ru_maxrss);
      sleep(2);
  }
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
  Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);
  cb->PreEndOfExecution = Tau_plugin_event_pre_end_of_execution;

  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);
  void * data = NULL;
  thread_vec.push_back(std::thread(Tau_plugin_do_work, data));

  return 0;
}

#endif

