
#include "Profile/TauSOS.h"
#include "Profile/Profiler.h"
#include "Profile/UserEvent.h"
#include "Profile/TauMetrics.h"
#include "Profile/TauMetaData.h"
#include <TauUtil.h>

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <set>
#include <stdexcept>
#include <cassert>
#include "stdio.h"
//#include "error.h"
#include "errno.h"
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#include <signal.h>
#include <algorithm>
#include <iterator>

#ifdef TAU_MPI
#include <mpi.h>
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <dlfcn.h>
#endif

#include "sos.h"

#define CONVERT_TO_USEC 1.0/1000000.0 // hopefully the compiler will precompute this.

SOS_pub *tau_sos_pub = NULL;
unsigned long fi_count = 0;
unsigned long ue_count = 0;
static bool done = false;
static SOS_runtime * _runtime = NULL;
pthread_mutex_t _my_mutex; // for initialization, termination
pthread_cond_t _my_cond; // for timer
pthread_t worker_thread;
bool _threaded = false;
int daemon_rank = 0;
bool shutdown_daemon = false;
int period_microseconds = 2000000;
unsigned long int instance_guid = 0UL;

void init_lock(void) {
    if (!_threaded) return;
    pthread_mutexattr_t Attr;
    pthread_mutexattr_init(&Attr);
    pthread_mutexattr_settype(&Attr, PTHREAD_MUTEX_ERRORCHECK);
    int rc;
    if ((rc = pthread_mutex_init(&_my_mutex, &Attr)) != 0) {
        errno = rc;
        perror("pthread_mutex_init error");
        exit(1);
    }
    if ((rc = pthread_cond_init(&_my_cond, NULL)) != 0) {
        errno = rc;
        perror("pthread_cond_init error");
        exit(1);
    }
}

extern "C" void * Tau_sos_thread_function(void* data) {
    /* Set the wakeup time (ts) to 2 seconds in the future. */
    struct timespec ts;
    struct timeval  tp;

    while (!done) {
        // wait x microseconds for the next batch.
        gettimeofday(&tp, NULL);
        const int one_second = 1000000;
        // first, add the period to the current microseconds
        int tmp_usec = tp.tv_usec + period_microseconds;
        int flow_sec = 0;
        if (tmp_usec > one_second) { // did we overflow?
            flow_sec = tmp_usec / one_second; // how many seconds?
            tmp_usec = tmp_usec % one_second; // get the remainder
        }
        ts.tv_sec  = (tp.tv_sec + flow_sec);
        ts.tv_nsec = (1000 * tmp_usec);
        pthread_mutex_lock(&_my_mutex);
        int rc = pthread_cond_timedwait(&_my_cond, &_my_mutex, &ts);
        if (rc == ETIMEDOUT) {
            TAU_VERBOSE("%d Sending data from TAU thread...\n", RtsLayer::myNode()); fflush(stderr);
            TAU_SOS_send_data();
            TAU_VERBOSE("%d Done.\n", RtsLayer::myNode()); fflush(stderr);
        } else if (rc == EINVAL) {
            TAU_VERBOSE("Invalid timeout!\n"); fflush(stderr);
        } else if (rc == EPERM) {
            TAU_VERBOSE("Mutex not locked!\n"); fflush(stderr);
        }
    }
    // unlock after being signalled.
    pthread_mutex_unlock(&_my_mutex);
    pthread_exit((void*)0L);
}

void TAU_SOS_make_pub() {
        char pub_name[SOS_DEFAULT_STRING_LEN] = {0};
        char app_version[SOS_DEFAULT_STRING_LEN] = {0};

        TAU_VERBOSE("[TAU_SOS_make_pub]: Creating new pub...\n");

#ifdef TAU_MPI
        int rank;
        int commsize;
        PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
        PMPI_Comm_size(MPI_COMM_WORLD, &commsize);
        _runtime->config.comm_rank = rank;
        _runtime->config.comm_size = commsize;
#endif

/* Fixme! Replace these with values from TAU metadata. */
        sprintf(pub_name, "TAU_SOS_SUPPORT");
        sprintf(app_version, "v0.alpha");
/* Fixme! Replace these with values from TAU metadata. */
        // old API:
        SOS_pub_create(_runtime, &tau_sos_pub, pub_name, SOS_NATURE_DEFAULT);
        // new API:
        // SOS_pub_init(_runtime, &tau_sos_pub, pub_name, SOS_NATURE_DEFAULT);

        strcpy(tau_sos_pub->prog_ver, app_version);
        tau_sos_pub->meta.channel       = 1;
        tau_sos_pub->meta.layer         = SOS_LAYER_LIB;
        // tau_sos_pub->meta.pri_hint      = SOS_PRI_IMMEDIATE;
        // tau_sos_pub->meta.scope_hint    = SOS_SCOPE_SELF;
        // tau_sos_pub->meta.retain_hint   = SOS_RETAIN_SESSION;

        TAU_VERBOSE("[TAU_SOS_make_pub]:   ... done.  (pub->guid == %ld)\n", tau_sos_pub->guid);
        TAU_VERBOSE("[TAU_SOS_make_pub]: Announcing the pub...\n");
        SOS_announce(tau_sos_pub);
	// all processes in this MPI execution should agree on the session.
#ifdef TAU_MPI
	if (rank == 0) {
        instance_guid = tau_sos_pub->guid;
	}
	PMPI_Bcast( &instance_guid, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD );
#else
    instance_guid = tau_sos_pub->guid;
#endif
    SOS_pack(tau_sos_pub, "TAU::MPI::INSTANCE_ID", SOS_VAL_TYPE_LONG, &instance_guid);
}

void TAU_SOS_do_fork(std::string forkCommand) {
    std::istringstream iss(forkCommand);
    std::vector<std::string> tokens;
    copy(std::istream_iterator<std::string>(iss),
         std::istream_iterator<std::string>(),
         std::back_inserter(tokens));
    const char **args = (const char **)calloc(tokens.size()+1, sizeof(char*));
    for (int i = 0; i < tokens.size() ; i++) {
        args[i] = tokens[i].c_str();
    }
    int rc = execvp(args[0],const_cast<char* const*>(args));
    if (rc < 0) {
        perror("\nError in execvp! Failed to spawn SOS client.  Things are gonna go sideways...");
    }
    // exit the daemon spawn!
    //std::cout << "Daemon exited!" << std::endl;
    _exit(0);
}

void TAU_SOS_fork_exec_sosd_shutdown(void) {
#ifdef TAU_MPI
    // first, figure out who should fork a daemon on this node
    int i, rank;
    PMPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if (rank == daemon_rank) {
        int pid = vfork();
        if (pid == 0) {
            char* forkCommand;
            forkCommand = getenv ("SOS_FORK_SHUTDOWN");
            if (forkCommand) {
                std::cout << "Rank " << rank << " stopping SOS daemon(s): " << forkCommand << std::endl;
                std::string foo(forkCommand);
                TAU_SOS_do_fork(foo);
            } else {
                std::cout << "Please set the SOS_FORK_SHUTDOWN environment variable to stop SOS in the background." << std::endl;
            }
        }
    }
    //
    // wait until it is running
    //
    //wait(2);
#endif
}

void TAU_SOS_send_shutdown_message(void) {
#ifdef TAU_MPI
    int i, rank;
    PMPI_Comm_rank(MPI_COMM_WORLD,&rank);
    SOS_buffer     *buffer;
    SOS_msg_header  header;
    int offset;
    if (rank == daemon_rank) {
        TAU_VERBOSE("Waiting for SOS to flush...\n");
		sleep(1);

        SOS_buffer_init(_runtime, &buffer);

        header.msg_size = -1;
        header.msg_type = SOS_MSG_TYPE_SHUTDOWN;
        header.msg_from = _runtime->my_guid;
        header.ref_guid = 0;

        offset = 0;
        SOS_buffer_pack(buffer, &offset, "iigg",
                header.msg_size,
                header.msg_type,
                header.msg_from,
                header.ref_guid);

        header.msg_size = offset;
        offset = 0;
        SOS_buffer_pack(buffer, &offset, "i", header.msg_size);

        TAU_VERBOSE("Sending SOS_MSG_TYPE_SHUTDOWN ...\n");

        SOS_send_to_daemon(buffer, buffer);

        SOS_buffer_destroy(buffer);
    }
#endif
}

void TAU_SOS_fork_exec_sosd(void) {
#ifdef TAU_MPI
    // first, figure out who should fork a daemon on this node
    int i, rank, size;
    PMPI_Comm_rank(MPI_COMM_WORLD,&rank);
    PMPI_Comm_size(MPI_COMM_WORLD,&size);
    // get my hostname
    const int hostlength = 128;
    char hostname[hostlength] = {0};
    gethostname(hostname, sizeof(char)*hostlength);
    //std::cout << hostname << std::endl;
    // make array for all hostnames
    char * allhostnames = (char*)calloc(hostlength*size, sizeof(char));
    // copy my name into the big array
    char * host_index = allhostnames + (hostlength * rank);
    strncpy(host_index, hostname, hostlength);
    // get all hostnames
    PMPI_Allgather(hostname, hostlength, MPI_CHAR, allhostnames, 
                   hostlength, MPI_CHAR, MPI_COMM_WORLD);
    daemon_rank = 0;
    // point to the head of the array
    host_index = allhostnames;
    // find the lowest rank with my hostname
    for (i = 0 ; i < size ; i++) {
        //printf("%d:%d comparing '%s' to '%s'\n", rank, size, hostname, host_index);
        if (strncmp(hostname, host_index, hostlength) == 0) {
            daemon_rank = i;
        }
        host_index = host_index + hostlength;
    }
    // fork the daemon
    if (rank == daemon_rank) {
        int pid = vfork();
        if (pid == 0) {
            char* forkCommand = NULL;
            char* ranks_per_node = NULL;
            char* offset = NULL;
            forkCommand = getenv ("SOS_FORK_COMMAND");
            //std::cout << "forkCommand " << forkCommand << std::endl;
            ranks_per_node = getenv ("SOS_APP_RANKS_PER_NODE");
            //std::cout << "ranks_per_node " << ranks_per_node << std::endl;
            offset = getenv ("SOS_LISTENER_RANK_OFFSET");
            //std::cout << "offset " << offset << std::endl;
            if (forkCommand) {
                std::string custom_command(forkCommand);
                size_t index = 0;
                index = custom_command.find("@LISTENER_RANK@", index);
                if (index != std::string::npos) {
                    if (ranks_per_node) {
                        int rpn = atoi(ranks_per_node);
                        int listener_rank = rank / rpn;
                        if(offset) {
                            int off = atoi(offset);
                            listener_rank = listener_rank + off;
                        }
                        std::stringstream ss;
                        ss << listener_rank;
                        custom_command.replace(index,15,ss.str());
                    }
                }
                std::cout << "SOS Listener not found, Rank " << rank << " spawning SOS daemon(s): " << custom_command << std::endl;
                TAU_SOS_do_fork(custom_command);
            } else {
                std::cerr << "Please set the SOS_FORK_COMMAND environment variable to spawn SOS in the background." << std::endl;
            }
        }
    }
    //
    // wait until it is running
    //
    //wait(2);
#endif
}

void program_path(char* exe_name)
{
#ifdef __APPLE__
    uint32_t size = 4098;
    if (_NSGetExecutablePath(exe_name, &size) != 0) {
        strcpy(exe_name,"");
    }
#else
    if (readlink("/proc/self/exe", exe_name, PATH_MAX) == -1) {
        strcpy(exe_name,"");
    }
#endif
}

char ** fix_arguments(int *argc) {
#if 1
  char ** argv = NULL;
  argv = (char**)(malloc(sizeof(char**)));
  char exe_name[2048] = {0};
  program_path(exe_name);
  argv[0] = strdup(exe_name);
  *argc = 1; 
#else
  char ** argv = NULL;
  FILE *f = fopen("/proc/self/cmdline", "r");
  if (f) {
    char line[4096];

    std::string os;
    while (Tau_util_readFullLine(line, f)) {
      if (os.length() != 0) {
        os.append(" ");
      }
      os.append(line);
    }
    fclose(f);

    //how many tokens?
    char * token = strtok (const_cast<char*>(os.c_str())," ");
    while (token != NULL)
    {
      *argc = *argc + 1;
      token = strtok (NULL, " ");
    }

    if (argv == NULL) {
      // allocate a list of strings
      argv = (char**)(malloc(sizeof(char**) * (*argc)));
    }

    //get the tokens
    token = strtok (const_cast<char*>(os.c_str())," ");
    int i = 0;
    while (token != NULL)
    {
      argv[i] = strdup(token);
      token = strtok (NULL, " ");
    }

  }
#endif
  return argv;
}

extern "C" void TAU_SOS_init(int * argc, char *** argv, bool threaded) {
    static bool initialized = false;
    int my_argc = 0;
    char ** my_argv = NULL;
    TAU_VERBOSE("TAU_SOS_init()...\n");
    if (!TauEnv_get_sos_enabled()) { TAU_VERBOSE("*** SOS NOT ENABLED! ***\n"); return; }
    if (!initialized) {
        _threaded = threaded > 0 ? true : false;
        init_lock();
        // if runtime returns null, wait a bit and try again. If 
        // we fail "too many" times, give an error and continue
        _runtime = NULL;
        TAU_VERBOSE("TAU_SOS_init() trying to connect...\n");
        if (argc == NULL || argv == NULL || *argc == 0) {
          my_argv = fix_arguments(&my_argc);
        } else {
          my_argc = *argc;
          my_argv = *argv;
        }
        // Old API:
        SOS_init(&my_argc, &my_argv, &_runtime, SOS_ROLE_CLIENT, SOS_RECEIVES_NO_FEEDBACK, NULL);
        // New API:
        //SOS_init(&_runtime, SOS_ROLE_CLIENT, SOS_RECEIVES_NO_FEEDBACK, NULL);
        if(_runtime == NULL) {
            TAU_VERBOSE("Unable to connect to SOS daemon. Spawning...\n");
            TAU_SOS_fork_exec_sosd();
            shutdown_daemon = true;
        }
        int repeat = 3;
        while(_runtime == NULL) {
            sleep(2);
            _runtime = NULL;
            TAU_VERBOSE("TAU_SOS_init() trying to connect...\n");
            // Old API:
            SOS_init(&my_argc, &my_argv, &_runtime, SOS_ROLE_CLIENT, SOS_RECEIVES_NO_FEEDBACK, NULL);
            // New API:
            //SOS_init(&_runtime, SOS_ROLE_CLIENT, SOS_RECEIVES_NO_FEEDBACK, NULL);
            if (_runtime != NULL) {
                TAU_VERBOSE("Connected to SOS daemon. Continuing...\n");
                break;
            } else if (--repeat < 0) { 
                TAU_VERBOSE("Unable to connect to SOS daemon. Failing...\n");
                return;
            }
        }

        if (_threaded && TauEnv_get_sos_periodic()) {
			period_microseconds = TauEnv_get_sos_period();
            TAU_VERBOSE("Spawning thread for SOS.\n");
            int ret = pthread_create(&worker_thread, NULL, &Tau_sos_thread_function, NULL);
            if (ret != 0) {
                errno = ret;
                perror("Error: pthread_create (1) fails\n");
                exit(1);
            }
        }
        initialized = true;
        /* Fixme! Insert all the data that was collected into Metadata */
    }
    Tau_metadata_push_to_sos();
    //SOS_announce(tau_sos_pub);
    SOS_publish(tau_sos_pub);
}

extern "C" void TAU_SOS_init_simple(void) {
    bool threads = false;
#if defined(PTHREADS) || defined(TAU_OPENMP)
    //threads = true;
#endif
    int argc = 0;
    char **argv = NULL;
    TAU_SOS_init(&argc, &argv, threads);
}

extern "C" void TAU_SOS_stop_worker(void) {
    if (_runtime == NULL) { return; }
    //printf("%s\n", __func__); fflush(stdout);
    pthread_mutex_lock(&_my_mutex);
    done = true;
    pthread_mutex_unlock(&_my_mutex);
    if (_threaded && TauEnv_get_sos_periodic()) {
        TAU_VERBOSE("TAU SOS thread joining...\n"); fflush(stderr);
        pthread_cond_signal(&_my_cond);
        int ret = pthread_join(worker_thread, NULL);
        if (ret != 0) {
            switch (ret) {
                case ESRCH:
                    // already exited.
                    break;
                case EINVAL:
                    // Didn't exist?
                    break;
                case EDEADLK:
                    // trying to join with itself?
                    break;
                default:
                    errno = ret;
                    perror("Warning: pthread_join failed\n");
                    break;
            }
        }
        pthread_cond_destroy(&_my_cond);
        pthread_mutex_destroy(&_my_mutex);
    }
}

extern "C" void TAU_SOS_finalize(void) {
    static bool finalized = false;
    // only thread 0 should finalize
    if (RtsLayer::myThread() > 0) { return; }
    // no runtime? quit.
    if (_runtime == NULL) { return; }
    // no SOS enabled? quit.
    if (!TauEnv_get_sos_enabled()) { return; }
    //printf("%s\n", __func__); fflush(stdout);
    // Already finalized? quit.
    if (finalized) return;
    finalized = true;
    if (!done) {
        TAU_SOS_stop_worker();
    }
    // flush any outstanding packs
    TAU_SOS_send_data();
#ifdef TAU_MPI
	// wait for ALL RANKS to get to this point.  They should be done
	// sending all data to the listener.
    TAU_VERBOSE("Waiting for SOS clients to rendez-vous...\n");
	PMPI_Barrier(MPI_COMM_WORLD);
#endif
    // shutdown the daemon, if necessary
    if (shutdown_daemon) {
        TAU_SOS_send_shutdown_message();
        // shouldn't be necessary, but sometimes the shutdown message is ignored?
        //TAU_SOS_fork_exec_sosd_shutdown();
    }
    SOS_finalize(_runtime);
}

extern "C" int TauProfiler_updateAllIntermediateStatistics(void);
extern "C" Profiler * Tau_get_current_profiler(void);

extern "C" void Tau_SOS_pack_current_timer(const char * event_name) {
    if (_runtime == NULL) { return; }
    // first time?
    if (tau_sos_pub == NULL) {
        RtsLayer::LockDB();
        // protect against race conditions
        if (tau_sos_pub == NULL) {
            TAU_SOS_make_pub();
        }
        RtsLayer::UnLockDB();
    }
    if (done) { return; }
    // get the current profiler
    Profiler * p = Tau_get_current_profiler();
    // get the current time
    double current;
    int tid = RtsLayer::myThread();
    RtsLayer::getUSecD(tid, &current);
    // assume time is the first counter!
    // also assume it is in microseconds!
    double value = (current - p->StartTime[0]) * CONVERT_TO_USEC;
    RtsLayer::LockDB();
    SOS_pack(tau_sos_pub, event_name, SOS_VAL_TYPE_DOUBLE, &value);
    RtsLayer::UnLockDB();
}

extern "C" void Tau_SOS_pack_string(const char * name, char * value) {
    if (_runtime == NULL) { return; }
    if (done) { return; }
    // first time?
    if (tau_sos_pub == NULL) {
        RtsLayer::LockDB();
        // protect against race conditions
        if (tau_sos_pub == NULL) {
            TAU_SOS_make_pub();
        }
        RtsLayer::UnLockDB();
    }
    std::stringstream ss;
    ss << "TAU::" << RtsLayer::myThread() << "::Metadata::" << name;
    RtsLayer::LockDB();
    SOS_pack(tau_sos_pub, ss.str().c_str(), SOS_VAL_TYPE_STRING, value);
    RtsLayer::UnLockDB();
}

extern "C" void Tau_SOS_pack_double(const char * name, double value) {
    if (_runtime == NULL) { return; }
    if (done) { return; }
    // first time?
    if (tau_sos_pub == NULL) {
        RtsLayer::LockDB();
        // protect against race conditions
        if (tau_sos_pub == NULL) {
            TAU_SOS_make_pub();
        }
        RtsLayer::UnLockDB();
    }
    std::stringstream ss;
    ss << "TAU::" << RtsLayer::myThread() << "::Metadata::" << name;
    // TAU_VERBOSE("SOS: %s = '%s'\n", name, value);
    RtsLayer::LockDB();
    SOS_pack(tau_sos_pub, ss.str().c_str(), SOS_VAL_TYPE_DOUBLE, &value);
    RtsLayer::UnLockDB();
}

extern "C" void Tau_SOS_pack_integer(const char * name, int value) {
    if (_runtime == NULL) { return; }
    if (done) { return; }
    // first time?
    if (tau_sos_pub == NULL) {
        RtsLayer::LockDB();
        // protect against race conditions
        if (tau_sos_pub == NULL) {
            TAU_SOS_make_pub();
        }
        RtsLayer::UnLockDB();
    }
    std::stringstream ss;
    ss << "TAU::" << RtsLayer::myThread() << "::Metadata::" << name;
    // TAU_VERBOSE("SOS: %s = '%s'\n", name, value);
    RtsLayer::LockDB();
    SOS_pack(tau_sos_pub, ss.str().c_str(), SOS_VAL_TYPE_INT, &value);
    RtsLayer::UnLockDB();
}

bool get_low_res_counter_name(const char *name, std::string &out_name) {
	// eliminate context counters
	if (strstr(name," : ") != NULL) {
		return false;
	}
	if (strstr(name," => ") != NULL) {
		return false;
	}
    if (strstr(name,"Message size for ") != NULL) {
		out_name.assign("Collective Bytes Sent");
		return true;
	}
    if (strstr(name,"Message size received from all nodes") != NULL) {
		out_name.assign("MPI Receive Bytes");
		return true;
	}
	/*
    if (strstr(name,"Message size received in wait") != NULL) {
		out_name.assign("MPI Receive Bytes");
		return true;
	}
	*/
    if (strstr(name,"Message size sent to all nodes") != NULL) {
		out_name.assign("MPI Send Bytes");
		return true;
	}
    if (strstr(name,"Message size sent to node ") != NULL) {
		return false;
	}
    if (strstr(name,"Bytes Read") != NULL) {
		out_name.assign("IO Bytes Read");
		return true;
	}
    if (strstr(name,"Bytes Written") != NULL) {
		out_name.assign("IO Bytes Written");
		return true;
	}
    if (strstr(name,"ADIOS data size") != NULL) {
		out_name.assign("ADIOS data size");
		return true;
	}
    out_name.assign(name);
	return true;
}

extern "C" void TAU_SOS_send_data(void) {
    if (_runtime == NULL) { return; }
    // first time?
    if (tau_sos_pub == NULL) {
        RtsLayer::LockDB();
        // protect against race conditions
        if (tau_sos_pub == NULL) {
            TAU_SOS_make_pub();
        }
        RtsLayer::UnLockDB();
    }
    assert(tau_sos_pub);
    //TauTrackPowerHere(); // get a power measurement
    Tau_global_incr_insideTAU();
    // get the most up-to-date profile information
    TauProfiler_updateAllIntermediateStatistics();

    // get the FunctionInfo database, and iterate over it
    std::vector<FunctionInfo*>::const_iterator it;
  const char **counterNames;
  int numCounters;
  TauMetrics_getCounterList(&counterNames, &numCounters);
  //printf("Num Counters: %d, Counter[0]: %s\n", numCounters, counterNames[0]);
  RtsLayer::LockDB();

  std::map<std::string, std::vector<double>* > low_res_timer_map;
  std::map<std::string, std::vector<double>* >::iterator timer_map_it;

  //foreach: TIMER
  for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
    FunctionInfo *fi = *it;
    // get the number of calls
    int tid = 0; // todo: get ALL thread data.
    int calls;
    double inclusive, exclusive;
    calls = 0;
    inclusive = 0.0;
    exclusive = 0.0;

	std::vector<double> *tmp_vec = NULL;
    if (TauEnv_get_sos_high_resolution() == 0) {
	  std::string group_name(fi->GetAllGroups());
      timer_map_it = low_res_timer_map.find(group_name);
	  if (timer_map_it == low_res_timer_map.end()) {
	    // create a vector with space for count and inclusive/exclusive for each metric
	    tmp_vec = new std::vector<double>((Tau_Global_numCounters*2) + 1, 0.0);
	    // add it to the map
	    low_res_timer_map.insert(std::pair<std::string, std::vector<double>* >(group_name, tmp_vec));
	  } else {
	    tmp_vec = timer_map_it->second;
	  }
	}

    //foreach: THREAD
    for (tid = 0; tid < RtsLayer::getTotalThreads(); tid++) {
        calls = fi->GetCalls(tid);
		int vec_index = 0;

        if (TauEnv_get_sos_high_resolution()) {
          std::stringstream calls_str;
          calls_str << "TAU::" << tid << "::calls::" << fi->GetAllGroups() << "::" << fi->GetName();
          const std::string& tmpcalls = calls_str.str();
          SOS_pack(tau_sos_pub, tmpcalls.c_str(), SOS_VAL_TYPE_INT, &calls);
		} else {
		  (*tmp_vec)[vec_index++] += (double)calls;
		}

        // todo - subroutines
        // iterate over metrics 
        std::stringstream incl_str;
        std::stringstream excl_str;
        for (int m = 0; m < Tau_Global_numCounters; m++) {
            inclusive = fi->getDumpInclusiveValues(tid)[m];
            exclusive = fi->getDumpExclusiveValues(tid)[m];
            if (TauEnv_get_sos_high_resolution()) {
                incl_str.str(std::string());
                incl_str << "TAU::" << tid << "::inclusive_" << counterNames[m] << "::" << fi->GetName();
                const std::string& tmpincl = incl_str.str();
                excl_str.str(std::string());
                excl_str << "TAU::" << tid << "::exclusive_" << counterNames[m] << "::" << fi->GetName();
                const std::string& tmpexcl = excl_str.str();
                SOS_pack(tau_sos_pub, tmpincl.c_str(), SOS_VAL_TYPE_DOUBLE, &inclusive);
                SOS_pack(tau_sos_pub, tmpexcl.c_str(), SOS_VAL_TYPE_DOUBLE, &exclusive);
			} else {
		        (*tmp_vec)[vec_index++] += inclusive;
		        (*tmp_vec)[vec_index++] += exclusive;
			}
        }
    }
  }

  if (TauEnv_get_sos_high_resolution() == 0) {
    for (timer_map_it = low_res_timer_map.begin() ; timer_map_it != low_res_timer_map.end() ; timer_map_it++) {
	  int vec_index = 0;
	  std::string group_name = timer_map_it->first;
	  std::vector<double> * values = timer_map_it->second;
      std::stringstream tmp_ss;
      tmp_ss << "TAU::0::calls::" << group_name;
      std::string tmp_name = tmp_ss.str();
	  double tmp_val = (*values)[vec_index++] / RtsLayer::getTotalThreads();
      SOS_pack(tau_sos_pub, tmp_name.c_str(), SOS_VAL_TYPE_DOUBLE, &tmp_val);
      for (int m = 0; m < Tau_Global_numCounters; m++) {
        tmp_ss.str("");
        tmp_ss.clear();
        tmp_ss << "TAU::0::inclusive_" << counterNames[m] << "::" << group_name;
        tmp_name = tmp_ss.str();
	    tmp_val = (*values)[vec_index++] / RtsLayer::getTotalThreads();
        SOS_pack(tau_sos_pub, tmp_name.c_str(), SOS_VAL_TYPE_DOUBLE, &tmp_val);
        tmp_ss.str("");
        tmp_ss.clear();
        tmp_ss << "TAU::0::exclusive_" << counterNames[m] << "::" << group_name;
        tmp_name = tmp_ss.str();
	    tmp_val = (*values)[vec_index++] / RtsLayer::getTotalThreads();
        SOS_pack(tau_sos_pub, tmp_name.c_str(), SOS_VAL_TYPE_DOUBLE, &tmp_val);
	  }
	  delete values;
	}
	low_res_timer_map.clear();
  }

  std::map<std::string, double> low_res_counter_map;
  std::map<std::string, double>::iterator counter_map_it;

  // do the same with counters.
  //std::vector<tau::TauUserEvent*>::const_iterator it2;
  tau::AtomicEventDB::iterator it2;
  int numEvents;
  double max, min, mean, sumsqr;
  std::stringstream tmp_str;
  for (it2 = tau::TheEventDB().begin(); it2 != tau::TheEventDB().end(); it2++) {
    tau::TauUserEvent *ue = (*it2);
	double tmp_accum = 0.0;
	std::string counter_name;

    if (TauEnv_get_sos_high_resolution() == 0) {
	  // if not a counter we want to keep, continue
	  if (!get_low_res_counter_name(ue->GetName().c_str(), counter_name)) {
	    continue;
	  }
      counter_map_it = low_res_counter_map.find(counter_name);
	  if (counter_map_it == low_res_counter_map.end()) {
	    // add it to the map
	    low_res_counter_map.insert(std::pair<std::string, double>(counter_name, tmp_accum));
	  } else {
	    tmp_accum = counter_map_it->second;
	  }
	}

    int tid = 0;
    for (tid = 0; tid < RtsLayer::getTotalThreads(); tid++) {
      if (ue && ue->GetNumEvents(tid) == 0) continue;
      //if (ue && ue->GetWriteAsMetric()) continue;
      numEvents = ue->GetNumEvents(tid);
      mean = ue->GetMean(tid);
      if (TauEnv_get_sos_high_resolution()) {
        max = ue->GetMax(tid);
        min = ue->GetMin(tid);
        sumsqr = ue->GetSumSqr(tid);
        tmp_str << "TAU::" << tid << "::NumEvents::" << ue->GetName();
        SOS_pack(tau_sos_pub, tmp_str.str().c_str(), SOS_VAL_TYPE_INT, &numEvents);
        tmp_str.str(std::string());
        tmp_str << "TAU::" << tid << "::Max::" << ue->GetName();
        SOS_pack(tau_sos_pub, tmp_str.str().c_str(), SOS_VAL_TYPE_DOUBLE, &max);
        tmp_str.str(std::string());
        tmp_str << "TAU::" << tid << "::Min::" << ue->GetName();
        SOS_pack(tau_sos_pub, tmp_str.str().c_str(), SOS_VAL_TYPE_DOUBLE, &min);
        tmp_str.str(std::string());
        tmp_str << "TAU::" << tid << "::Mean::" << ue->GetName();
        SOS_pack(tau_sos_pub, tmp_str.str().c_str(), SOS_VAL_TYPE_DOUBLE, &mean);
        tmp_str.str(std::string());
        tmp_str << "TAU::" << tid << "::SumSqr::" << ue->GetName();
        SOS_pack(tau_sos_pub, tmp_str.str().c_str(), SOS_VAL_TYPE_DOUBLE, &sumsqr);
        tmp_str.str(std::string());
      } else {
	    tmp_accum += mean * numEvents;
	    low_res_counter_map[counter_name] = tmp_accum;
	  }
    }
  }
  if (TauEnv_get_sos_high_resolution() == 0) {
    for (counter_map_it = low_res_counter_map.begin() ; counter_map_it != low_res_counter_map.end() ; counter_map_it++) {
	  std::string counter_name = counter_map_it->first;
	  double value = counter_map_it->second;
      std::stringstream tmp_ss;
	  // this is "total", because we take the product of count * mean
      tmp_ss << "TAU::0::Total::" << counter_name;
      std::string tmp_name = tmp_ss.str();
	  value = value / RtsLayer::getTotalThreads();
      SOS_pack(tau_sos_pub, tmp_name.c_str(), SOS_VAL_TYPE_DOUBLE, &value);
	}
	low_res_counter_map.clear();
  }

  if ((ue_count + fi_count) > SOS_DEFAULT_ELEM_MAX) {
      TAU_VERBOSE("DANGER, WILL ROBINSON! EXCEEDING MAX ELEMENTS IN SOS. Bad things might happen?\n");
  }
  RtsLayer::UnLockDB();
  TAU_VERBOSE("[TAU_SOS_send_data]: Publishing the values...\n");
  TAU_VERBOSE("MY RANK IS: %d/%d\n", _runtime->config.comm_rank, _runtime->config.comm_size);
  SOS_publish(tau_sos_pub);
  TAU_VERBOSE("[TAU_SOS_send_data]:   ...done.\n");
  Tau_global_decr_insideTAU();
}

