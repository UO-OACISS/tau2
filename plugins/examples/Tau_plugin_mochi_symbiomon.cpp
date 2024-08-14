/***************************************************************************
 * *   Plugin Testing
 * *   This plugin will provide iterative output of TAU profile data to an
 * *   MOCHI service.
 * *
 * *************************************************************************/

/* Have the TAU configure set -DTAU_MOCHI when you want this plugin */
#if defined(TAU_MOCHI)

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <assert.h>
#include <ctype.h>
#include <unistd.h>
#include <mutex>

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>
#include <Profile/TauMetaData.h>

#ifdef TAU_MPI
#include<mpi.h>
#endif

#include <margo.h>
#include <symbiomon/symbiomon-metric.h>
#include <symbiomon/symbiomon-common.h>
#include <symbiomon/symbiomon-server.h>

/* So that we can keep track of how long this plugin takes */
#include "Tau_scoped_timer.h"

static bool enabled{false};
static bool initialized{false};
static bool opened{false};
static bool done{false};
static margo_instance_id mid;
static symbiomon_provider_t provider;
static int reduction_frequency = 0;

#define RESERVE(container, size) container.reserve(size)
#define UPDATE_KEY(container, val) container.push_back(val)
#define UPDATE_VAL(container, val) container.push_back(val)

pthread_mutex_t _my_mutex; // for initialization, termination

/* These are useful if you want to tell Mochi what the name of the program
 * is or if you want to get the path to the executable. */

char *_program_path()
{
#if defined(__APPLE__)
    return NULL;
#else
    char path[PATH_MAX] = {0};
    if (readlink("/proc/self/exe", path, PATH_MAX) == -1) {
        return(NULL);
    }
    std::string tmp(path);
    size_t i = tmp.rfind('/', tmp.length());
    if (i != string::npos) {
        //sprintf(path, "%s", tmp.substr(i+1, tmp.length() - i).c_str());
        path[i] = '\0';
    }
    return strdup(path);
#endif
}

char *_program_name()
{
#if defined(__APPLE__)
    return NULL;
#else
    char path[PATH_MAX] = {0};
    if (readlink("/proc/self/exe", path, PATH_MAX) == -1) {
        return(NULL);
    }
    std::string tmp(path);
    size_t i = tmp.rfind('/', tmp.length());
    if (i != string::npos) {
        snprintf(path, sizeof(path),  "%s", tmp.substr(i+1, tmp.length() - i).c_str());
    }
    return strdup(path);
#endif
}

void Tau_dump_mochi_metadata() {
    tau::plugins::ScopedTimer(__func__);
    int tid = RtsLayer::myThread();
    int nodeid = TAU_PROFILE_GET_NODE();
    for (MetaDataRepo::iterator it = Tau_metadata_getMetaData(tid).begin();
         it != Tau_metadata_getMetaData(tid).end(); it++) {
        std::stringstream ss;
        /* Build a metadata key */
        ss << "TAU:" << nodeid << ":" << tid << ":MetaData:" << it->first.name;
        switch(it->second->type) {
            case TAU_METADATA_TYPE_STRING:
                /* do something with  it->second->data.cval */
                break;
            case TAU_METADATA_TYPE_INTEGER:
                /* do something with  it->second->data.ival */
                break;
            case TAU_METADATA_TYPE_DOUBLE:
                /* do something with  it->second->data.dval */
                break;
            case TAU_METADATA_TYPE_TRUE:
                /* do something with  "true" */
                break;
            case TAU_METADATA_TYPE_FALSE:
                /* do something with  "false" */
                break;
            case TAU_METADATA_TYPE_NULL:
                /* do something with  "null" */
                break;
            default:
                break;
        }
    }
}

void Tau_plugin_mochi_init_mochi(void) {
    int ret;
    /* initialize the local mochi provider */
    mid = margo_init("ofi+verbs://", MARGO_SERVER_MODE, 0, -1);
    if (mid == MARGO_INSTANCE_NULL)
    {
	std::cerr << "Error: margo_init()" << std::endl;
	ret = -1;
        return;
    }

    struct symbiomon_provider_args args = SYMBIOMON_PROVIDER_ARGS_INIT;
    args.push_finalize_callback = 0;
    //args.pool = pool;

    ret = symbiomon_provider_register(mid, 42, &args, &provider);
    if (ret != 0)
    {
	std::cerr << "Error: symbiomon_provider_register()" << std::endl;
	margo_finalize(mid);
	ret = -1;
        return;
    }

    if (ret == 0) { initialized = true; }
    assert(ret == 0);
}

void Tau_plugin_mochi_open_file(void) {
    /* open mochi connection */

    int ret = 0;
    if (ret == 0) { opened = true;}
    assert(ret == 0);
}

/* Convenience function if you want to strip the source info */
void shorten_timer_name(std::string& name) {
    std::size_t index = name.find(" [{");
    if (index != std::string::npos) {
        name = name.substr(0,index);
    }
}

/* Iterates over the FunctionInfo DB and EventDB to compile a list of
 * per-thread metrics and counters. */
void Tau_plugin_mochi_write_variables() {
    static uint32_t iteration_counter = 0;

    RtsLayer::LockDB();
    /* Copy the function info database so we can release the lock */
    std::vector<FunctionInfo*> tmpTimers(TheFunctionDB());
    RtsLayer::UnLockDB();

    int numThreadsLocal = RtsLayer::getTotalThreads();
    /* "counters" here are metrics.  Sorry for the confusion. */
    const char **counterNames;
    std::vector<int> numCounters = {0};
    TauMetrics_getCounterList(&counterNames, &(numCounters[0]));

    std::map<std::string, std::vector<double> >::iterator timer_map_it;

    int rank = RtsLayer::myNode();
    int size = tau_totalnodes(0,1);

    stringstream rank_;
    rank_ << rank;
    symbiomon_taglist_t taglist, taglist2;
    symbiomon_taglist_create(&taglist, 1, (rank_.str()).c_str());
    symbiomon_taglist_create(&taglist2, 2, (rank_.str()).c_str(), "min");

    //foreach: TIMER
    std::vector<FunctionInfo*>::const_iterator it;
    for (it = tmpTimers.begin(); it != tmpTimers.end(); it++) {
        FunctionInfo *fi = *it;

        stringstream ss;
        std::string shortName(fi->GetName());
        shorten_timer_name(shortName);
        ss << shortName;
        ss << "_Calls";
        symbiomon_metric_t m;
	symbiomon_metric_create_with_reduction("tau2", (ss.str()).c_str(), SYMBIOMON_TYPE_COUNTER, (ss.str()).c_str(), taglist, &m, provider, SYMBIOMON_REDUCTION_OP_MAX);

        // assign real data
        double total_tid_calls = 0;
        for (int tid = 0; tid < numThreadsLocal; tid++) {
            /* build a name with ss, tid */
            /* write the fi->GetCalls(tid) value */
            total_tid_calls += (double)fi->GetCalls(tid);
        }

	symbiomon_metric_update(m, total_tid_calls);

        for (int m = 0 ; m < numCounters.size() ; m++) {
            symbiomon_metric_t inc, inc_min, exc, exc_min;
            stringstream incl, incl_min;
            stringstream excl, excl_min;
            incl << shortName;
            incl_min << shortName;
            incl << "_Inclusive_";
            incl_min << "_Inclusive_min_";
            incl << counterNames[m];
            incl_min << counterNames[m];
	    symbiomon_metric_create_with_reduction("tau2", (incl.str()).c_str(), SYMBIOMON_TYPE_TIMER, (incl.str()).c_str(), taglist, &inc, provider, SYMBIOMON_REDUCTION_OP_MAX);
	    symbiomon_metric_create_with_reduction("tau2", (incl_min.str()).c_str(), SYMBIOMON_TYPE_TIMER, (incl_min.str()).c_str(), taglist, &inc_min, provider, SYMBIOMON_REDUCTION_OP_MIN);
            excl << shortName;
            excl_min << shortName;
            excl <<  "_Exclusive_";
            excl_min <<  "_Exclusive_min_";
            excl << counterNames[m];
            excl_min << counterNames[m];
	    symbiomon_metric_create_with_reduction("tau2", (excl.str()).c_str(), SYMBIOMON_TYPE_TIMER, (excl.str()).c_str(), taglist, &exc, provider, SYMBIOMON_REDUCTION_OP_MAX);
	    symbiomon_metric_create_with_reduction("tau2", (excl_min.str()).c_str(), SYMBIOMON_TYPE_TIMER, (excl_min.str()).c_str(), taglist, &exc_min, provider, SYMBIOMON_REDUCTION_OP_MIN);
            // assign real data
            double inc_time = 0, exc_time = 0;
            for (int tid = 0; tid < numThreadsLocal; tid++) {
                if(fi->GetCalls(tid) == 0) {
		  inc_time += 0.0;
		  exc_time += 0.0;
                } else {
                  inc_time += (fi->getDumpInclusiveValues(tid)[m]);
                  exc_time += (fi->getDumpExclusiveValues(tid)[m]);
                }

       }
	    symbiomon_metric_update(inc, inc_time);
	    symbiomon_metric_update(inc_min, inc_time);
            //std::cout << (incl.str()).c_str() << "_" << rank << " has value " << inc_time << std::endl;
	    symbiomon_metric_update(exc, exc_time);
	    symbiomon_metric_update(exc_min, exc_time);
        }
    }

    /* Lock the counter map */
    RtsLayer::LockDB();
    tau::AtomicEventDB::const_iterator it2;
    std::map<std::string, std::vector<double> >::iterator counter_map_it;

    // do the same with counters.
    for (it2 = tau::TheEventDB().begin(); it2 != tau::TheEventDB().end(); it2++) {
        tau::TauUserEvent *ue = (*it2);
        if (ue == NULL) continue;
        std::string counter_name(ue->GetName().c_str());
	symbiomon_metric_t numevents_m, mean_m, min_m, max_m, sumsquares_m;
        std::stringstream ss, mean, min, max, sumsqr;
        ss << counter_name << "_NumEvents";
        mean << counter_name << "_Mean";
        min << counter_name << "_Min";
        max << counter_name << "_Max";
        sumsqr << counter_name << "_SumSquares";

	symbiomon_metric_create_with_reduction("tau2", (ss.str()).c_str(), SYMBIOMON_TYPE_COUNTER, (ss.str()).c_str(), taglist, &numevents_m, provider, SYMBIOMON_REDUCTION_OP_MAX);
	symbiomon_metric_create_with_reduction("tau2", (mean.str()).c_str(), SYMBIOMON_TYPE_GAUGE, (mean.str()).c_str(), taglist, &mean_m, provider, SYMBIOMON_REDUCTION_OP_MAX);
	symbiomon_metric_create_with_reduction("tau2", (min.str()).c_str(), SYMBIOMON_TYPE_GAUGE, (min.str()).c_str(), taglist, &min_m, provider, SYMBIOMON_REDUCTION_OP_MAX);
	symbiomon_metric_create_with_reduction("tau2", (max.str()).c_str(), SYMBIOMON_TYPE_GAUGE, (max.str()).c_str(), taglist, &max_m, provider, SYMBIOMON_REDUCTION_OP_MAX);
	symbiomon_metric_create_with_reduction("tau2", (sumsqr.str()).c_str(), SYMBIOMON_TYPE_GAUGE, (sumsqr.str()).c_str(), taglist, &sumsquares_m, provider, SYMBIOMON_REDUCTION_OP_MAX);

        // assign real data
        double numevents_val = 0, mean_val = 0, min_val = 0, max_val = 0, sumsqr_val = 0;
        for (int tid = 0; tid < numThreadsLocal; tid++) {
            numevents_val += ((double)ue->GetNumEvents(tid));
            mean_val += ((double)ue->GetMean(tid));
            min_val += ((double)ue->GetMin(tid));
            max_val += ((double)ue->GetMax(tid));
            sumsqr_val += ((double)ue->GetSumSqr(tid));
        }

	symbiomon_metric_update(numevents_m, numevents_val);
	symbiomon_metric_update(mean_m, mean_val);
	symbiomon_metric_update(min_m, min_val);
	symbiomon_metric_update(max_m, max_val);
	symbiomon_metric_update(sumsquares_m, sumsqr_val);
    }

    /* unlock the counter map */
    RtsLayer::UnLockDB();

    /*Aggregate all local metrics at a user-provided reduction frequency*/
    if((iteration_counter % reduction_frequency) == 0) {
        symbiomon_metric_reduce_all(provider);
	MPI_Barrier(MPI_COMM_WORLD);
	/* Perform global aggregation if rank 0*/
	if(!rank) {
            fprintf(stderr, "SYMBIOMON: Performing metric aggregation.\n");
	    symbiomon_metric_global_reduce_all(provider, size);
	}
    }
    iteration_counter += 1;
}

int Tau_plugin_mochi_dump(Tau_plugin_event_dump_data_t* data) {
    if (!enabled) return 0;
    TAU_VERBOSE("TAU PLUGIN MOCHI: dump\n");
    tau::plugins::ScopedTimer(__func__);

    if (!initialized) {
        Tau_plugin_mochi_init_mochi();
    }

    Tau_global_incr_insideTAU();
    // get the most up-to-date profile information
    TauProfiler_updateAllIntermediateStatistics();
    Tau_global_decr_insideTAU();

    if (!opened) {
       Tau_plugin_mochi_open_file();
    }

    if (opened) {
        Tau_plugin_mochi_write_variables();
    }

    return 0;
}

/* This is a weird event, not sure what for */
int Tau_plugin_finalize(Tau_plugin_event_function_finalize_data_t* data) {
    return 0;
}

/* This happens from MPI_Finalize, before MPI is torn down. */
int Tau_plugin_mochi_pre_end_of_execution(Tau_plugin_event_pre_end_of_execution_data_t* data) {
    if (!enabled) return 0;
    TAU_VERBOSE("TAU PLUGIN MOCHI Pre-Finalize\n"); fflush(stdout);
#ifdef TAU_MPI
    Tau_plugin_event_dump_data_t dummy;
    dummy.tid = 0;
    /* write final data */
    Tau_plugin_mochi_dump(&dummy);
    if (opened) {
        /* close mochi if it has to close before MPI_Finalize */
        opened = false;
    }
    enabled = false;
#endif
    margo_finalize(mid);
    return 0;
}

/* This happens after MPI_Init, and after all TAU metadata variables have been
 * read */
int Tau_plugin_mochi_post_init(Tau_plugin_event_post_init_data_t* data) {
    if (!enabled) return 0;
    Tau_plugin_mochi_init_mochi();
    Tau_plugin_mochi_open_file();
    return 0;
}

/* This happens from Profiler.cpp, when data is written out. */
int Tau_plugin_mochi_end_of_execution(Tau_plugin_event_end_of_execution_data_t* data) {
    if (!enabled || data->tid != 0) return 0;
    TAU_VERBOSE("TAU PLUGIN MOCHI Finalize\n"); fflush(stdout);
#ifndef TAU_MPI
    if (opened) {
        Tau_plugin_event_dump_data_t dummy;
        dummy.tid = 0;
        /* write final data */
        Tau_plugin_mochi_dump(&dummy);
        bpWriter.Close();
        opened = false;
    }
    enabled = false;
#endif
    margo_finalize(mid);
    return 0;
}

/* Function to remove all spaces from a given string */
void removeSpaces(char *str) {
     // To keep track of non-space character count
     int count = 0;
     for (int i = 0; str[i]; i++)
         if (str[i] != ' ') str[count++] = str[i];
             str[count] = '\0';
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
    Tau_plugin_callbacks_t cb;
    TAU_VERBOSE("TAU PLUGIN MOCHI Init\n"); fflush(stdout);
    /* Create the callback object */
    TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(&cb);
    /* Required event support */
    cb.Dump = Tau_plugin_mochi_dump;
    cb.PostInit = Tau_plugin_mochi_post_init;
    cb.PreEndOfExecution = Tau_plugin_mochi_pre_end_of_execution;
    cb.EndOfExecution = Tau_plugin_mochi_end_of_execution;

    /* Register the callback object */
    TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(&cb, id);
    enabled = true;
    char * red_freq_str = getenv("TAU_SYMBIOMON_REDUCTION_FREQUENCY");
    if(red_freq_str != NULL) {
	    reduction_frequency = atoi(red_freq_str);
	    fprintf(stderr, "Reduction frequency is: %d\n", reduction_frequency);
    } else {
	    reduction_frequency = 1;
	    fprintf(stderr, "Reduction frequency is: %d\n", reduction_frequency);
    }
    return 0;
}


#endif // TAU_MOCHI
