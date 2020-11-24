/***************************************************************************
 * *   Plugin Testing
 * *   This plugin will provide iterative output of TAU profile data to an
 * *   MOCHI service.
 * *
 * *************************************************************************/

/* Have the TAU configure set -DTAU_MOCHI when you want this plugin */
#if defined(TAU_MOCHI)

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <assert.h>

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>
#include <Profile/TauMetaData.h>
/* So that we can keep track of how long this plugin takes */
#include "Tau_scoped_timer.h"

static bool enabled{false};
static bool initialized{false};
static bool opened{false};
static bool done{false};
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
        sprintf(path, "%s", tmp.substr(i+1, tmp.length() - i).c_str());
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
    /* initialize mochi client */
    initialized = true;
}

void Tau_plugin_mochi_open_file(void) {
    /* open mochi connection */
    opened = true;
}

/* Convenience function if you want to strip the source info */
void shorten_timer_name(std::string& name) {
    std::size_t index = name.find(" [{");
    if (index != std::string::npos) {
        name = name.substr(0,index);
    }
}

void Tau_plugin_mochi_write_variables() {
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

    //foreach: TIMER
    std::vector<FunctionInfo*>::const_iterator it;
    for (it = tmpTimers.begin(); it != tmpTimers.end(); it++) {
        FunctionInfo *fi = *it;

        stringstream ss;
        std::string shortName(fi->GetName());
        shorten_timer_name(shortName);
        ss << shortName << " / Calls";

        // assign real data
        for (int tid = 0; tid < numThreadsLocal; tid++) {
            /* build a name with ss, tid */
            /* write the fi->GetCalls(tid) value */
        }
        
        for (int m = 0 ; m < numCounters ; m++) {
            stringstream incl;
            stringstream excl;
            incl << shortName << " / Inclusive " << counterNames[m];
            excl << shortName << " / Exclusive " << counterNames[m];
            // assign real data
            for (int tid = 0; tid < numThreadsLocal; tid++) {
                if (fi->GetCalls(tid) == 0) {
                    /* write zeros */
                } else {
                    /* build a name with incl, tid */
                    /* write the fi->getDumpInclusiveValues(tid)[m] value */
                    /* build a name with excl, tid */
                    /* write the fi->getDumpExclusiveValues(tid)[m] value */
                }
            }
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
        std::stringstream ss, mean, min, max, sumsqr;
        ss << counter_name << " / Num Events";
        mean << counter_name << " / Mean";
        min << counter_name << " / Min";
        max << counter_name << " / Max";
        sumsqr << counter_name << " / Sum Squares";

        // assign real data
        for (int tid = 0; tid < numThreadsLocal; tid++) {
            /* as above, build a key and write out:
            ue->GetNumEvents(tid);
            ue->GetMean(tid);
            ue->GetMax(tid);
            ue->GetMin(tid);
            ue->GetSumSqr(tid);
            */
        }
    }
    /* unlock the counter map */
    RtsLayer::UnLockDB();
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
    return 0;
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
    return 0;
}


#endif // TAU_MOCHI
