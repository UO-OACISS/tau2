#if defined(TAU_SOS)

#include "TauSOS.h"
#include "Profile/Profiler.h"
#include "Profile/UserEvent.h"
#include "Profile/TauMetrics.h"
#include "Profile/TauMetaData.h"
#include "Profile/TauUtil.h"
#include "Profile/TauAPI.h"

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <set>
#include <stdexcept>
#include <cassert>
#include "stdio.h"
#include "errno.h"
#include <pthread.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <signal.h>
#include <algorithm>
#include <iterator>
#include <fcntl.h>


#ifdef TAU_MPI
#include <mpi.h>
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <dlfcn.h>
#endif

#ifndef TAU_MAX_METRICS
#define TAU_MAX_METRICS 25 //Temporary
#endif

#include "sos.h"

SOS_pub *tau_sos_pub = NULL;
unsigned long fi_count = 0;
unsigned long ue_count = 0;
static bool done = false;
static SOS_runtime * _runtime = NULL;
static int listener_pid = 0;
pthread_mutex_t _my_mutex; // for initialization, termination
pthread_cond_t _my_cond; // for timer
pthread_t worker_thread;
bool _threaded = false;
int daemon_rank = 0;
int my_rank = 0;
int listener_rank = 0;
int comm_size = 1;
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

void * Tau_sos_thread_function(void* data) {
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
            TAU_VERBOSE("[TAU_SOS]%d Sending data from TAU thread...\n", RtsLayer::myNode()); fflush(stderr);
            TAU_SOS_send_data(false);
            TAU_VERBOSE("[TAU_SOS]%d Done.\n", RtsLayer::myNode()); fflush(stderr);
        } else if (rc == EINVAL) {
            TAU_VERBOSE("[TAU_SOS]Invalid timeout!\n"); fflush(stderr);
        } else if (rc == EPERM) {
            TAU_VERBOSE("[TAU_SOS]Mutex not locked!\n"); fflush(stderr);
        }
    }
    // unlock after being signalled.
    pthread_mutex_unlock(&_my_mutex);
    pthread_exit((void*)0L);
	return(NULL);
}

void TAU_SOS_make_pub() {
        char pub_name[SOS_DEFAULT_STRING_LEN] = {0};
        char app_version[SOS_DEFAULT_STRING_LEN] = {0};

        TAU_VERBOSE("[TAU_SOS_make_pub]: Creating new pub...\n");

#ifdef TAU_MPI
        _runtime->config.comm_rank = my_rank;
        _runtime->config.comm_size = comm_size;
#endif

        sprintf(pub_name, "TAU_SOS_SUPPORT");
        sprintf(app_version, "v0.alpha");
        SOS_pub_init(_runtime, &tau_sos_pub, pub_name, SOS_NATURE_DEFAULT);
        SOS_pub_config(tau_sos_pub, SOS_PUB_OPTION_CACHE, thePluginOptions().env_sos_cache_depth);

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
	if (my_rank == 0) {
        instance_guid = tau_sos_pub->guid;
	}
	PMPI_Bcast( &instance_guid, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD );
#else
    instance_guid = tau_sos_pub->guid;
#endif
    SOS_pack(tau_sos_pub, "TAU:MPI:INSTANCE_ID", SOS_VAL_TYPE_LONG, &instance_guid);
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
    if (my_rank == daemon_rank) {
        int pid = vfork();
        if (pid == 0) {
            char* forkCommand;
            forkCommand = getenv ("SOS_FORK_SHUTDOWN");
            if (forkCommand) {
                std::cout << "Rank " << my_rank << " stopping SOS daemon(s): " << forkCommand << std::endl;
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
    SOS_buffer     *buffer;
    SOS_msg_header  header;
    int offset;
    SOS_buffer_init(_runtime, &buffer);

    header.msg_size = -1;
    header.msg_type = SOS_MSG_TYPE_SHUTDOWN;
    header.msg_from = _runtime->my_guid;
    header.ref_guid = 0;

    offset = 0;
    SOS_buffer_pack(buffer, &offset, (char*)"iigg",
            header.msg_size,
            header.msg_type,
            header.msg_from,
            header.ref_guid);

    header.msg_size = offset;
    offset = 0;
    SOS_buffer_pack(buffer, &offset, (char*)"i", header.msg_size);
    TAU_VERBOSE("[TAU_SOS]Sending SOS_MSG_TYPE_SHUTDOWN ...\n");
    SOS_send_to_daemon(buffer, buffer);
    SOS_buffer_destroy(buffer);
	char * exporting = getenv("SOS_EXPORT_DB_AT_EXIT");
	if (exporting != NULL) {
    	TAU_VERBOSE("[TAU_SOS]Waiting %d seconds for SOS to write (if necessary)...\n", thePluginOptions().env_sos_shutdown_delay);
		sleep(thePluginOptions().env_sos_shutdown_delay);
	}
#endif
}

bool TAU_SOS_fork_exec_sosd(void) {
#ifdef TAU_MPI
    // first, figure out who should fork a daemon on this node
    int i;
    // get my hostname
    const int hostlength = 128;
    char hostname[hostlength] = {0};
    gethostname(hostname, sizeof(char)*hostlength);
    //std::cout << hostname << std::endl;
    // make array for all hostnames
    char * allhostnames = (char*)calloc(hostlength * comm_size, sizeof(char));
    // copy my name into the big array
    char * host_index = allhostnames + (hostlength * my_rank);
    strncpy(host_index, hostname, hostlength);
    // get all hostnames
    PMPI_Allgather(hostname, hostlength, MPI_CHAR, allhostnames,
                   hostlength, MPI_CHAR, MPI_COMM_WORLD);
    daemon_rank = 0;
    // point to the head of the array
    host_index = allhostnames;
    // find the lowest rank with my hostname
    for (i = 0 ; i < comm_size ; i++) {
        //printf("%d:%d comparing '%s' to '%s'\n", rank, size, hostname, host_index);
        if (strncmp(hostname, host_index, hostlength) == 0) {
            daemon_rank = i;
        }
        host_index = host_index + hostlength;
    }
    char* forkCommand = NULL;
    char* ranks_per_node = NULL;
    char* offset = NULL;
    forkCommand = getenv ("SOS_FORK_COMMAND");
    //std::cout << "forkCommand " << forkCommand << std::endl;
    ranks_per_node = getenv ("SOS_APP_RANKS_PER_NODE");
    //std::cout << "ranks_per_node " << ranks_per_node << std::endl;
    offset = getenv ("SOS_LISTENER_RANK_OFFSET");
    //std::cout << "offset " << offset << std::endl;
    if (!forkCommand || !ranks_per_node || !offset) {
        if (my_rank == 0) {
            std::cerr << "Please set the SOS_FORK_COMMAND, SOS_APP_RANKS_PER_NODE, and SOS_LISTENER_RANK_OFFSET environment variables to spawn SOS in the background." << std::endl;
            std::cerr << "SOS Listener not found, SOS plugin not configured." << std::endl;
        }
        return false;
    }
    // fork the daemon
    if (my_rank == daemon_rank) {
        int outfd = 0;
        int errfd = 0;
        /* This is disabled unless you need to log the output from listeners */
        if (ranks_per_node && false) {
            int rpn = atoi(ranks_per_node);
            std::stringstream ss;
            ss << "listener." << my_rank / rpn << ".out";
            if ((outfd = open(ss.str().c_str(), O_CREAT|O_TRUNC|O_WRONLY, 0644)) < 0) {
                perror(ss.str().c_str());    /* open failed */
                exit(1);
            }
            std::cout << "writing output of the command " << forkCommand << " to " << ss.str() << std::endl;
            std::stringstream ss2;
            ss2 << "listener." << my_rank / rpn << ".err";
            if ((errfd = open(ss2.str().c_str(), O_CREAT|O_TRUNC|O_WRONLY, 0644)) < 0) {
                perror(ss2.str().c_str());    /* open failed */
                exit(1);
            }
            std::cout << "writing error of the command " << forkCommand << " to " << ss2.str() << std::endl;
        }
        int listener_pid = vfork();
        if (listener_pid == 0) {
            if (forkCommand) {
                std::string custom_command(forkCommand);
                size_t index = 0;
                index = custom_command.find("@LISTENER_RANK@", index);
                if (index != std::string::npos) {
                    if (ranks_per_node) {
                        int rpn = atoi(ranks_per_node);
                        listener_rank = my_rank / rpn;
                        if(offset) {
                            int off = atoi(offset);
                            listener_rank = listener_rank + off;
                        }
                        std::stringstream ss;
                        ss << listener_rank;
                        custom_command.replace(index,15,ss.str());
                    }
                }
                std::cout << "SOS Listener not found, Rank " << my_rank << " spawned SOS daemon(s): " << custom_command << std::endl;
                if (outfd > 0) {
                    dup2(outfd, STDOUT_FILENO);    /* fd becomes the standard output */
                }
                if (errfd > 0) {
                    dup2(errfd, STDERR_FILENO);    /* fd becomes the standard output */
                }
                TAU_SOS_do_fork(custom_command);
            }
        }
    }
    //
    // wait until it is running
    //
    //wait(2);
#endif
    return true;
}

/*********************************************************************
 * Parse a boolean value
 ********************************************************************/
static int parse_bool(const char *str, int default_value = 0) {
  if (str == NULL) {
    return default_value;
  }
  static char strbuf[128];
  char *ptr = strbuf;
  strncpy(strbuf, str, 128);
  while (*ptr) {
    *ptr = tolower(*ptr);
    ptr++;
  }
  if (strcmp(strbuf, "yes") == 0  ||
      strcmp(strbuf, "true") == 0 ||
      strcmp(strbuf, "on") == 0 ||
      strcmp(strbuf, "1") == 0) {
    return 1;
  } else {
    return 0;
  }
}

/*********************************************************************
 * Parse an integer value
 ********************************************************************/
static int parse_int(const char *str, int default_value = 0) {
  if (str == NULL) {
    return default_value;
  }
  int tmp = atoi(str);
  if (tmp < 0) {
    return default_value;
  }
  return tmp;
}

void TAU_SOS_parse_environment_variables(void) {
    char * tmp = NULL;
    tmp = getenv("TAU_SOS");
    if (parse_bool(tmp, TAU_SOS_DEFAULT)) {
        thePluginOptions().env_sos_enabled = 1;
    }
    tmp = getenv("TAU_SOS_TRACE_ADIOS");
    if (parse_bool(tmp, TAU_SOS_TRACE_ADIOS_DEFAULT)) {
      thePluginOptions().env_sos_trace_adios = 1;
    }
    tmp = getenv("TAU_SOS_TRACING");
    if (parse_bool(tmp, TAU_SOS_TRACING_DEFAULT)) {
        thePluginOptions().env_sos_tracing = 1;
    }
    tmp = getenv("TAU_SOS_CACHE_DEPTH");
    thePluginOptions().env_sos_cache_depth = parse_int(tmp, TAU_SOS_CACHE_DEPTH_DEFAULT);
    tmp = getenv("TAU_SOS_PERIODIC");
    if (parse_bool(tmp, TAU_SOS_PERIODIC_DEFAULT)) {
      thePluginOptions().env_sos_periodic = 1;
      tmp = getenv("TAU_SOS_PERIOD");
      thePluginOptions().env_sos_period = parse_int(tmp, TAU_SOS_PERIOD_DEFAULT);
    }
    tmp = getenv("TAU_SOS_SHUTDOWN_DELAY_SECONDS");
    thePluginOptions().env_sos_shutdown_delay = parse_int(tmp, TAU_SOS_SHUTDOWN_DELAY_DEFAULT);
    tmp = getenv("TAU_SOS_SELECTION_FILE");
    if (tmp != NULL) {
      Tau_SOS_parse_selection_file(tmp);
    }
    // also needed:
    // - disable profile output (trace only)
}

void Tau_SOS_parse_selection_file(const char * filename) {
    std::ifstream file(filename);
    std::string str;
    bool including_timers = false;
    bool excluding_timers = false;
    bool including_counters = false;
    bool excluding_counters = false;
    thePluginOptions().env_sos_use_selection = true;
    while (std::getline(file, str)) {
        // trim right whitespace
        str.erase(str.find_last_not_of(" \n\r\t")+1);
        // trim left whitespace
        str.erase(0, str.find_first_not_of(" \n\r\t"));
        // skip blank lines
        if (str.size() == 0) {
            continue;
        }
        // skip comments
        if (str.find("#", 0) == 0) {
            continue;
        }
        if (str.compare("BEGIN_INCLUDE_TIMERS") == 0) {
            including_timers = true;
        } else if (str.compare("END_INCLUDE_TIMERS") == 0) {
            including_timers = false;
        } else if (str.compare("BEGIN_EXCLUDE_TIMERS") == 0) {
            excluding_timers = true;
        } else if (str.compare("END_EXCLUDE_TIMERS") == 0) {
            excluding_timers = false;
        } else if (str.compare("BEGIN_INCLUDE_COUNTERS") == 0) {
            including_counters = true;
        } else if (str.compare("END_INCLUDE_COUNTERS") == 0) {
            including_counters = false;
        } else if (str.compare("BEGIN_EXCLUDE_COUNTERS") == 0) {
            excluding_counters = true;
        } else if (str.compare("END_EXCLUDE_COUNTERS") == 0) {
            excluding_counters = false;
        } else {
            if (including_timers) {
                if (str.find("#") == string::npos && str.find("?") == string::npos) {
                    thePluginOptions().included_timers.insert(str);
                } else {
                    thePluginOptions().included_timers_with_wildcards.insert(str);
                }
            } else if (excluding_timers) {
                if (str.find("#") == string::npos && str.find("?") == string::npos) {
                    thePluginOptions().excluded_timers.insert(str);
                } else {
                    thePluginOptions().excluded_timers_with_wildcards.insert(str);
                }
            } else if (including_counters) {
                if (str.find("#") == string::npos && str.find("?") == string::npos) {
                    thePluginOptions().included_counters.insert(str);
                } else {
                    thePluginOptions().included_counters_with_wildcards.insert(str);
                }
            } else if (excluding_counters) {
                if (str.find("#") == string::npos && str.find("?") == string::npos) {
                    thePluginOptions().excluded_counters.insert(str);
                } else {
                    thePluginOptions().excluded_counters_with_wildcards.insert(str);
                }
            } else {
                std::cerr << "Warning, selection outside of include/exclude section: "
                    << str << std::endl;
            }
        }
    }
}

bool TAU_SOS_init() {
    static bool initialized = false;
    TAU_VERBOSE("[TAU_SOS]TAU_SOS_init()...\n");
    if (!initialized) {
        my_rank = RtsLayer::myNode();
        comm_size = tau_totalnodes(0,1);
        if (thePluginOptions().env_sos_periodic) {
            _threaded = true;
        } else {
            _threaded = false;
        }
        init_lock();
        // if runtime returns null, wait a bit and try again. If
        // we fail "too many" times, give an error and continue
        _runtime = NULL;
        TAU_VERBOSE("[TAU_SOS]TAU_SOS_init() trying to connect...\n");
        SOS_init(&_runtime, SOS_ROLE_CLIENT, SOS_RECEIVES_NO_FEEDBACK, NULL);
        if(_runtime == NULL) {
            TAU_VERBOSE("[TAU_SOS]Unable to connect to SOS daemon. Spawning...\n");
            shutdown_daemon = TAU_SOS_fork_exec_sosd();
            if (!shutdown_daemon) {
                // failed.  Don't claim initialized.
                _runtime = NULL;
                return false;
            }
        }
        int repeat = 3;
        while(_runtime == NULL) {
            sleep(2);
            _runtime = NULL;
            TAU_VERBOSE("TAU_SOS_init() trying to connect...\n");
            SOS_init(&_runtime, SOS_ROLE_CLIENT, SOS_RECEIVES_NO_FEEDBACK, NULL);
            if (_runtime != NULL) {
                TAU_VERBOSE("[TAU_SOS]Connected to SOS daemon. Continuing...\n");
                break;
            } else if (--repeat < 0) {
                TAU_VERBOSE("[TAU_SOS]Unable to connect to SOS daemon. Failing...\n");
                return false;
            }
        }

        if (thePluginOptions().env_sos_periodic) {
            period_microseconds = thePluginOptions().env_sos_period;
            TAU_VERBOSE("[TAU_SOS]Spawning thread for SOS.\n");
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
    // if we have a runtime, all is good.
    return (!(_runtime == NULL));
}

void TAU_SOS_stop_worker(void) {
    if (_runtime == NULL) { return; }
    //printf("%s\n", __func__); fflush(stdout);
    pthread_mutex_lock(&_my_mutex);
    done = true;
    pthread_mutex_unlock(&_my_mutex);
    if (thePluginOptions().env_sos_periodic) {
        TAU_VERBOSE("[TAU_SOS]TAU SOS thread joining...\n"); fflush(stderr);
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

void TAU_SOS_finalize(void) {
    static bool finalized = false;
    // only thread 0 should finalize
    if (RtsLayer::myThread() > 0) { return; }
    // no runtime? quit.
    if (_runtime == NULL) { return; }
    // no SOS enabled? quit.
    if (!thePluginOptions().env_sos_enabled) { return; }
    //printf("%s\n", __func__); fflush(stdout);
    // Already finalized? quit.
    if (finalized) return;
    finalized = true;
    if (!done) {
        TAU_SOS_stop_worker();
    }
    // flush any outstanding packs
    TAU_SOS_send_data(true);
    // shutdown the daemon, if necessary
    if (shutdown_daemon) {
        if (my_rank == daemon_rank) {
            TAU_VERBOSE("[TAU_SOS]Waiting %d seconds for SOS to flush...\n", thePluginOptions().env_sos_shutdown_delay);
		    sleep(thePluginOptions().env_sos_shutdown_delay);
			printf("[TAU_SOS]TAU: rank %d sending shutdown message to listener %d...\n", my_rank, listener_rank);
            TAU_SOS_send_shutdown_message();
			int returnStatus = 0;
			pid_t retval = 0;
			if (listener_pid != 0) {
				// wait for zombie children, just in case.
			    retval = waitpid((pid_t) (-1), &returnStatus, 0);
			} else {
				retval = waitpid(listener_pid, &returnStatus, 0);
			}
            if (WIFEXITED(returnStatus)) {
                printf("listener %d exited, status=%d\n", listener_rank, WEXITSTATUS(returnStatus));
            } else if (WIFSIGNALED(returnStatus)) {
                printf("listener %d killed by signal %d\n", listener_rank, WTERMSIG(returnStatus));
            } else if (WIFSTOPPED(returnStatus)) {
                printf("listener %d stopped by signal %d\n", listener_rank, WSTOPSIG(returnStatus));
            } else if (WIFCONTINUED(returnStatus)) {
                printf("listener %d continued\n", listener_rank);
            }
			if (retval < 0) {
				perror("waitpid error: ");
        		fprintf(stderr, "WARNING! SOS listener %d did not exit normally!\n", listener_rank);
			}
        }
        // shouldn't be necessary, but sometimes the shutdown message is ignored?
        //TAU_SOS_fork_exec_sosd_shutdown();
    }
    SOS_finalize(_runtime);
}

void Tau_SOS_pack_current_timer(const char * event_name) {
    if (_runtime == NULL) { return; }
    // first time?
    if (tau_sos_pub == NULL) {
        TAU_VERBOSE("[TAU_SOS_send_data]:  RtsLayer::LockDB()\n");
        RtsLayer::LockDB();
        // protect against race conditions
        if (tau_sos_pub == NULL) {
            TAU_SOS_make_pub();
        }
        TAU_VERBOSE("[TAU_SOS_send_data]:  RtsLayer::UnLockDB()\n");
        RtsLayer::UnLockDB();
    }
    if (done) { return; }
    // get the current profiler
    tau::Profiler * p = Tau_get_current_profiler();
    // get the current time
    double current[TAU_MAX_METRICS];//TODO: DYNATHREAD. Can't initialize this to TAU_MAX_THREADS. TAU_MAX_METRICS may be insufficient.
    int tid = RtsLayer::myThread();
    RtsLayer::getUSecD(tid, current);
    // assume time is the first counter!
    // also convert it to microseconds
    double value = (current[0] - p->StartTime[0]) * CONVERT_TO_USEC;
    // if (strlen(event_name) > 256) { printf("long string, %d: '%s'\n", strlen(event_name), event_name); }
    SOS_pack(tau_sos_pub, event_name, SOS_VAL_TYPE_DOUBLE, &value);
}

void Tau_SOS_pack_string(const char * name, char * value) {
    if (_runtime == NULL) { return; }
    if (done) { return; }
    // first time?
    if (tau_sos_pub == NULL) {
        TAU_VERBOSE("[TAU_SOS_send_data]:  RtsLayer::LockDB()\n");
        RtsLayer::LockDB();
        // protect against race conditions
        if (tau_sos_pub == NULL) {
            TAU_SOS_make_pub();
        }
        TAU_VERBOSE("[TAU_SOS_send_data]:  RtsLayer::UnLockDB()\n");
        RtsLayer::UnLockDB();
    }
    // if (strlen(name) > 256) { printf("long string, %d: '%s'\n", strlen(name), name); }
    SOS_pack(tau_sos_pub, name, SOS_VAL_TYPE_STRING, value);
}

void Tau_SOS_pack_double(const char * name, double value) {
    if (_runtime == NULL) { return; }
    if (done) { return; }
    // first time?
    if (tau_sos_pub == NULL) {
        TAU_VERBOSE("[TAU_SOS_send_data]:  RtsLayer::LockDB()\n");
        RtsLayer::LockDB();
        // protect against race conditions
        if (tau_sos_pub == NULL) {
            TAU_SOS_make_pub();
        }
        TAU_VERBOSE("[TAU_SOS_send_data]:  RtsLayer::UnLockDB()\n");
        RtsLayer::UnLockDB();
    }
    // if (strlen(name) > 256) { printf("long string, %d: '%s'\n", strlen(name), name); }
    SOS_pack(tau_sos_pub, name, SOS_VAL_TYPE_DOUBLE, &value);
}

void Tau_SOS_pack_integer(const char * name, int value) {
    if (_runtime == NULL) { return; }
    if (done) { return; }
    // first time?
    if (tau_sos_pub == NULL) {
        TAU_VERBOSE("[TAU_SOS_send_data]:  RtsLayer::LockDB()\n");
        RtsLayer::LockDB();
        // protect against race conditions
        if (tau_sos_pub == NULL) {
            TAU_SOS_make_pub();
        }
        TAU_VERBOSE("[TAU_SOS_send_data]:  RtsLayer::UnLockDB()\n");
        RtsLayer::UnLockDB();
    }
    // if (strlen(name) > 256) { printf("long string, %d: '%s'\n", strlen(name), name); }
    SOS_pack(tau_sos_pub, name, SOS_VAL_TYPE_INT, &value);
}

void Tau_SOS_pack_long(const char * name, long int value) {
    if (_runtime == NULL) { return; }
    if (done) { return; }
    // first time?
    if (tau_sos_pub == NULL) {
        TAU_VERBOSE("[TAU_SOS_send_data]:  RtsLayer::LockDB()\n");
        RtsLayer::LockDB();
        // protect against race conditions
        if (tau_sos_pub == NULL) {
            TAU_SOS_make_pub();
        }
        TAU_VERBOSE("[TAU_SOS_send_data]:  RtsLayer::UnLockDB()\n");
        RtsLayer::UnLockDB();
    }
    // if (strlen(name) > 256) { printf("long string, %d: '%s'\n", strlen(name), name); }
    SOS_pack(tau_sos_pub, name, SOS_VAL_TYPE_LONG, &value);
}

/* Necessary to use const char * because UserEvents use TauSafeString objects,
 * not std::string. We use the "if_empty" parameter to tell us how to treat
 * an empty set.  For exclude lists, it's false, for include lists, it's true */
bool Tau_SOS_contains(std::set<std::string>& myset,
        const char * key, bool if_empty) {
    // if the set has contents, and we are in the set, then return true.
    std::string _key(key);
    if (myset.size() == 0) {
        return if_empty;
    } else if (myset.find(_key) == myset.end()) {
        return false;
    }
    // otherwise, return false.
    return true;
}

void TAU_SOS_pack_profile() {
    Tau_global_incr_insideTAU();
    // get the most up-to-date profile information
    TauProfiler_updateAllIntermediateStatistics();
    TAU_VERBOSE("[TAU_SOS_send_data]:  RtsLayer::LockDB()\n");
    RtsLayer::LockDB();
    /* Copy the function info database so we can release the lock */
    std::vector<FunctionInfo*> tmpTimers(TheFunctionDB());
    // use the normal copy constructor.
    //tau::AtomicEventDB tmpCounters(tau::TheEventDB());
    TAU_VERBOSE("[TAU_SOS_send_data]:  RtsLayer::UnLockDB()\n");
    RtsLayer::UnLockDB();

    // get the FunctionInfo database, and iterate over it
    std::vector<FunctionInfo*>::const_iterator it;
    const char **counterNames;
    int numCounters;
    TauMetrics_getCounterList(&counterNames, &numCounters);
    //printf("Num Counters: %d, Counter[0]: %s\n", numCounters, counterNames[0]);

    std::map<std::string, std::vector<double>* > low_res_timer_map;
    std::map<std::string, std::vector<double>* >::iterator timer_map_it;

    //foreach: TIMER
    for (it = tmpTimers.begin(); it != tmpTimers.end(); it++) {
        FunctionInfo *fi = *it;
        /* First, check to see if we are including/excluding this timer */
        if (skip_timer(fi->GetName())) {
            continue;
        }
        // get the number of calls
        int tid = 0; // todo: get ALL thread data.
        int calls;
        double inclusive, exclusive;
        calls = 0;
        inclusive = 0.0;
        exclusive = 0.0;

        //foreach: THREAD
        for (tid = 0; tid < RtsLayer::getTotalThreads(); tid++) {
            calls = fi->GetCalls(tid);
            // skip this timer if this thread didn't call it.
            // for data-reduction reasons.
            if (calls == 0) continue;
            std::stringstream calls_str;
            calls_str << "TAU_TIMER:" << tid << ":calls:" << fi->GetAllGroups() << ":" << fi->GetName();
            const std::string& tmpcalls = calls_str.str();
            // if (strlen(tmpcalls.c_str()) > 256) { printf("long string, %d: '%s'\n", strlen(tmpcalls.c_str()), tmpcalls.c_str()); }
            SOS_pack(tau_sos_pub, tmpcalls.c_str(), SOS_VAL_TYPE_INT, &calls);

            // todo - subroutines
            // iterate over metrics
            std::stringstream incl_str;
            std::stringstream excl_str;
            for (int m = 0; m < Tau_Global_numCounters; m++) {
                inclusive = fi->getDumpInclusiveValues(tid)[m];
                exclusive = fi->getDumpExclusiveValues(tid)[m];
                incl_str.str(std::string());
                incl_str << "TAU_TIMER:" << tid << ":inclusive_" << counterNames[m] << ":" << fi->GetAllGroups() << ":" << fi->GetName();
                const std::string& tmpincl = incl_str.str();
                excl_str.str(std::string());
                excl_str << "TAU_TIMER:" << tid << ":exclusive_" << counterNames[m] << ":" << fi->GetAllGroups() << ":" << fi->GetName();
                const std::string& tmpexcl = excl_str.str();
                // if (strlen(tmpexcl.c_str()) > 256) { printf("long string, %d: '%s'\n", strlen(tmpexcl.c_str()), tmpexcl.c_str()); }
                // if (strlen(tmpincl.c_str()) > 256) { printf("long string, %d: '%s'\n", strlen(tmpincl.c_str()), tmpincl.c_str()); }
                SOS_pack(tau_sos_pub, tmpincl.c_str(), SOS_VAL_TYPE_DOUBLE, &inclusive);
                SOS_pack(tau_sos_pub, tmpexcl.c_str(), SOS_VAL_TYPE_DOUBLE, &exclusive);
            }
        }
    }

    tau::AtomicEventDB::const_iterator counterIterator;
    std::map<std::string, double> low_res_counter_map;
    std::map<std::string, double>::iterator counter_map_it;

    // do the same with counters.
    int numEvents;
    double max, min, mean, sumsqr;
    std::stringstream tmp_str;
    TAU_VERBOSE("[TAU_SOS_send_data]:  RtsLayer::LockDB()\n");
    RtsLayer::LockDB();
    for (counterIterator = tau::TheEventDB().begin();
         counterIterator != tau::TheEventDB().end(); counterIterator++) {
        tau::TauUserEvent *ue = (*counterIterator);
        if (ue == NULL) continue;
        /* First, check to see if we are including/excluding this counter */
        if (skip_counter(ue->GetName().c_str())) {
            continue;
        }
	    std::string counter_name;

        int tid = 0;
        for (tid = 0; tid < RtsLayer::getTotalThreads(); tid++) {
            if (ue->GetNumEvents(tid) == 0) continue;
            //if (ue && ue->GetWriteAsMetric()) continue;
            numEvents = ue->GetNumEvents(tid);
            mean = ue->GetMean(tid);
            max = ue->GetMax(tid);
            min = ue->GetMin(tid);
            sumsqr = ue->GetSumSqr(tid);
            tmp_str << "TAU_COUNTER:" << tid << ":NumEvents:" << ue->GetName();
            // if (strlen(tmp_str.str().c_str()) > 256) { printf("long string, %d: '%s'\n", strlen(tmp_str.str().c_str()), tmp_str.str().c_str()); }
            SOS_pack(tau_sos_pub, tmp_str.str().c_str(), SOS_VAL_TYPE_INT, &numEvents);
            tmp_str.str(std::string());
            tmp_str << "TAU_COUNTER:" << tid << ":Max:" << ue->GetName();
            // if (strlen(tmp_str.str().c_str()) > 256) { printf("long string, %d: '%s'\n", strlen(tmp_str.str().c_str()), tmp_str.str().c_str()); }
            SOS_pack(tau_sos_pub, tmp_str.str().c_str(), SOS_VAL_TYPE_DOUBLE, &max);
            tmp_str.str(std::string());
            tmp_str << "TAU_COUNTER:" << tid << ":Min:" << ue->GetName();
            // if (strlen(tmp_str.str().c_str()) > 256) { printf("long string, %d: '%s'\n", strlen(tmp_str.str().c_str()), tmp_str.str().c_str()); }
            SOS_pack(tau_sos_pub, tmp_str.str().c_str(), SOS_VAL_TYPE_DOUBLE, &min);
            tmp_str.str(std::string());
            tmp_str << "TAU_COUNTER:" << tid << ":Mean:" << ue->GetName();
            // if (strlen(tmp_str.str().c_str()) > 256) { printf("long string, %d: '%s'\n", strlen(tmp_str.str().c_str()), tmp_str.str().c_str()); }
            SOS_pack(tau_sos_pub, tmp_str.str().c_str(), SOS_VAL_TYPE_DOUBLE, &mean);
            tmp_str.str(std::string());
            tmp_str << "TAU_COUNTER:" << tid << ":SumSqr:" << ue->GetName();
            // if (strlen(tmp_str.str().c_str()) > 256) { printf("long string, %d: '%s'\n", strlen(tmp_str.str().c_str()), tmp_str.str().c_str()); }
            SOS_pack(tau_sos_pub, tmp_str.str().c_str(), SOS_VAL_TYPE_DOUBLE, &sumsqr);
            tmp_str.str(std::string());
        }
    }
    TAU_VERBOSE("[TAU_SOS_send_data]:  RtsLayer::UnLockDB()\n");
    RtsLayer::UnLockDB();

    if ((ue_count + fi_count) > SOS_DEFAULT_ELEM_MAX) {
        TAU_VERBOSE("[TAU_SOS]DANGER, WILL ROBINSON! EXCEEDING MAX ELEMENTS IN SOS. Bad things might happen?\n");
    }
}

extern "C" int Tau_open_status(void);
extern "C" int Tau_read_status(int fd, long long * rss, long long * hwm);

void TAU_SOS_send_data(bool finalizing) {
    // have we initialized?
    if (_runtime == NULL) {
        fprintf(stderr, "ERROR! No SOS runtime found, did you initialize?\n");
        return;
    }
    // Do we have a pub handle?
    if (tau_sos_pub == NULL) {
        TAU_VERBOSE("[TAU_SOS_send_data]:  RtsLayer::LockDB()\n");
        RtsLayer::LockDB();
        // protect against race conditions
        if (tau_sos_pub == NULL) {
            TAU_SOS_make_pub();
        }
        TAU_VERBOSE("[TAU_SOS_send_data]:  RtsLayer::UnLockDB()\n");
        RtsLayer::UnLockDB();
    }
    // Make sure we have a pub handle */
    assert(tau_sos_pub);
    /* records the heap, with no context, even though it says "here". */
    Tau_track_memory_here();
    /* records the rss/hwm, without context. */
    //Tau_track_memory_rss_and_hwm();
    /* records the load, without context */
    Tau_track_load();
    /* Only send a profile update if we aren't tracing */
    if (finalizing ||
        (!thePluginOptions().env_sos_tracing &&
	     !thePluginOptions().env_sos_trace_adios)) {
        TAU_SOS_pack_profile();
    }
    static int frame = 0;
    TAU_VERBOSE("[TAU_SOS_send_data]: Publishing the values for frame %d...\n", frame++);
    TAU_VERBOSE("[TAU_SOS_send_data]MY RANK IS: %d/%d\n", _runtime->config.comm_rank, _runtime->config.comm_size);
    SOS_announce(tau_sos_pub);
    SOS_publish(tau_sos_pub);
    TAU_VERBOSE("[TAU_SOS_send_data]:   ...done.\n");
    Tau_global_decr_insideTAU();
}

// C++ program to implement wildcard
// pattern matching algorithm
// from: https://www.geeksforgeeks.org/wildcard-pattern-matching/
#if defined(__APPLE__) && defined(__clang__)
// do nothing
#else
#include <bits/stdc++.h>
#endif
using namespace std;

// Function that matches input str with
// given wildcard pattern
bool strmatch(const char str[], const char pattern[],
              int n, int m)
{
    // empty pattern can only match with
    // empty string
    if (m == 0)
        return (n == 0);

    // lookup table for storing results of
    // subproblems
    bool lookup[n + 1][m + 1]; // = {false};
	// PGI compiler doesn't like initialization during declaration...
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= m; j++) {
            lookup[i][j] = false;
		}
	}

    // initailze lookup table to false
    //memset(lookup, false, sizeof(lookup));

    // empty pattern can match with empty string
    lookup[0][0] = true;

    // Only '#' can match with empty string
    for (int j = 1; j <= m; j++)
        if (pattern[j - 1] == '#')
            lookup[0][j] = lookup[0][j - 1];

    // fill the table in bottom-up fashion
    for (int i = 1; i <= n; i++)
    {
        for (int j = 1; j <= m; j++)
        {
            // Two cases if we see a '#'
            // a) We ignore ‘#’ character and move
            //    to next  character in the pattern,
            //     i.e., ‘#’ indicates an empty sequence.
            // b) '#' character matches with ith
            //     character in input
            if (pattern[j - 1] == '#')
                lookup[i][j] = lookup[i][j - 1] ||
                               lookup[i - 1][j];

            // Current characters are considered as
            // matching in two cases
            // (a) current character of pattern is '?'
            // (b) characters actually match
            else if (pattern[j - 1] == '?' ||
                    str[i - 1] == pattern[j - 1])
                lookup[i][j] = lookup[i - 1][j - 1];

            // If characters don't match
            else lookup[i][j] = false;
        }
    }

    return lookup[n][m];
}

bool skip_timer(const char * key) {
    // are we filtering at all?
    if (!thePluginOptions().env_sos_use_selection) {
        return false;
    }
    // check to see if this label is excluded
    if (Tau_SOS_contains(thePluginOptions().excluded_timers, key, false)) {
        return true;
    // check to see if this label is included
    } else if (Tau_SOS_contains(thePluginOptions().included_timers, key, false)) {
        return false;
    } else {
        // check to see if it's in the excluded wildcards
        for (std::set<std::string>::iterator
                it=thePluginOptions().excluded_timers_with_wildcards.begin();
             it!=thePluginOptions().excluded_timers_with_wildcards.end(); ++it) {
            if (strmatch(key, it->c_str(), strlen(key), it->length())) {
                // make the lookup faster next time
                thePluginOptions().excluded_timers.insert(key);
                return true;
            }
        }
        // check to see if it's in the included wildcards
        for (std::set<std::string>::iterator
                it=thePluginOptions().included_timers_with_wildcards.begin();
             it!=thePluginOptions().included_timers_with_wildcards.end(); ++it) {
            if (strmatch(key, it->c_str(), strlen(key), it->length())) {
                // make the lookup faster next time
                thePluginOptions().included_timers.insert(key);
                return false;
            }
        }
    }
    // neither included nor excluded?
    // do we have an inclusion list? If so, then skip (because we didn't match it).
    if (!thePluginOptions().included_timers.empty() ||
        !thePluginOptions().included_timers_with_wildcards.empty()) {
        return true;
    }
    // by default, don't skip it.
    return false;
}

bool skip_counter(const char * key) {
    // are we filtering at all?
    if (!thePluginOptions().env_sos_use_selection) {
        return false;
    }
    // check to see if this label is excluded
    if (Tau_SOS_contains(thePluginOptions().excluded_counters, key, false)) {
        return true;
    // check to see if this label is included
    } else if (Tau_SOS_contains(thePluginOptions().included_counters, key, false)) {
        return false;
    } else {
        // check to see if it's in the excluded wildcards
        for (std::set<std::string>::iterator
                it=thePluginOptions().excluded_counters_with_wildcards.begin();
             it!=thePluginOptions().excluded_counters_with_wildcards.end(); ++it) {
            if (strmatch(key, it->c_str(), strlen(key), it->length())) {
                // make the lookup faster next time
                thePluginOptions().excluded_counters.insert(key);
                return true;
            }
        }
        // check to see if it's in the included wildcards
        for (std::set<std::string>::iterator
                it=thePluginOptions().included_counters_with_wildcards.begin();
             it!=thePluginOptions().included_counters_with_wildcards.end(); ++it) {
            if (strmatch(key, it->c_str(), strlen(key), it->length())) {
                // make the lookup faster next time
                thePluginOptions().included_counters.insert(key);
                return false;
            }
        }
    }
    // neither included nor excluded?
    // do we have an inclusion list? If so, then skip (because we didn't match it).
    if (!thePluginOptions().included_counters.empty() ||
        !thePluginOptions().included_counters_with_wildcards.empty()) {
        return true;
    }
    return false;
}

#endif // TAU_SOS
