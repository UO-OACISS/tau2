/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/

#include <TAU.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>

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

int done = 0;

int Tau_plugin_event_end_of_execution(Tau_plugin_event_end_of_execution_data_t *data) {

  return 0;
}

void Tau_plugin_do_work(void * data) {
   struct rusage r_usage;
   getrusage(RUSAGE_SELF,&r_usage);
   fprintf(stderr, "Max Memory usage = %ld\n", r_usage.ru_maxrss);
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
  Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
  TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

  cb->StartAsyncPlugin = Tau_plugin_do_work;

  TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);

  return 0;
}

#endif

