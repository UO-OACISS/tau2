/****************************************************************************
 **			TAU Portable Profiling Package			   **
 **			http://www.cs.uoregon.edu/research/tau	           **
 *****************************************************************************
 **    Copyright 1997-2017	          			   	   **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/***************************************************************************
 **	File 		: TauKokkos.cpp					  **
 **	Description 	: TAU Profiling Interface for Kokkos. Use the env **
 **                       var KOKKOS_PROFILE_LIBRARY to point to libTAU.so**
 **	Contact		: tau-bugs@cs.uoregon.edu 		 	  **
 **	Documentation	: See http://www.cs.uoregon.edu/research/tau      **
 ***************************************************************************/


//////////////////////////////////////////////////////////////////////
// Include Files
//////////////////////////////////////////////////////////////////////

#ifdef TAU_DOT_H_LESS_HEADERS
#include <cstdio>
#include <inttypes.h>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <stack>
#include <iostream>
using namespace std;
#endif /* TAU_DOT_H_LESS_HEADERS */
#include <stdlib.h>

#include <TAU.h>
#include "Profile/Profiler.h"
#include "Profile/UserEvent.h"
#include "Profile/TauMetrics.h"
#include "Profile/TauMetaData.h"

extern "C" {

/* Data Structures */

    typedef struct ps_tool_timer_data
    {
        unsigned int num_timers;
        unsigned int num_threads;
        unsigned int num_metrics;
        char **timer_names;
        char **metric_names;
        double *values;
    } ps_tool_timer_data_t;

    typedef struct ps_tool_counter_data
    {
        unsigned int num_counters;
        unsigned int num_threads;
        char **counter_names;
        double *num_samples;
        double *value_total;
        double *value_min;
        double *value_max;
        double *value_sumsqr;
    } ps_tool_counter_data_t;

    typedef struct ps_tool_metadata
    {
        unsigned int num_values;
        char **names;
        char **values;
    } ps_tool_metadata_t;

/* Function pointers */

void ps_tool_initialize(void) {
    int _argc = 1;
    const char *_dummy = "";
    char *_argv[1];
    _argv[0] = (char *)(_dummy);
    Tau_init(_argc, _argv);
#ifndef TAU_MPI
    Tau_set_node(0);
#endif
    /* Disable throttling, because if users use ps_tool_stop_current(),
     * throttling will cause Tau_start() to do nothing for throttled events,
     * but Tau_global_stop() will stop the timer on the stop of the stack */
    TauEnv_set_throttle(0);
    Tau_create_top_level_timer_if_necessary();
}

void ps_tool_register_thread(void) {
    Tau_register_thread();
    Tau_create_top_level_timer_if_necessary();
}

void Tau_profile_exit_all_threads();

void ps_tool_finalize(void) {
#ifndef TAU_MPI
    //Tau_destructor_trigger();
    //Tau_profile_exit_all_threads();
    Tau_exit("stub exiting");
#endif
}

void ps_tool_pause_measurement(void) {
    Tau_disable_instrumentation();
}

void ps_tool_resume_measurement(void) {
    Tau_enable_instrumentation();
}

void ps_tool_dump_data(void) {
#if 0
    const char **counterNames;
    int numCounters;
    TauMetrics_getCounterList(&counterNames, &numCounters);

    TauTrackMemoryFootPrintHere();
    TauTrackMemoryHere();
    //TauTrackMemoryHeadroomHere();
    if (numCounters == 1) {
        TauTrackPowerHere();
    }
    TauTrackLoadHere();
#endif
    Tau_dump();
}

void * ps_tool_timer_create(const char * name) {
    return Tau_pure_search_for_function(name, 1);
}

void ps_tool_timer_start(const void * timer) {
    Tau_start_timer((FunctionInfo*)timer, 0, Tau_get_thread());
}

void ps_tool_timer_stop(const void * timer) {
    Tau_stop_timer((FunctionInfo*)timer, Tau_get_thread());
}

void ps_tool_start_string(const char * name) {
    Tau_pure_start_task_group(name, Tau_get_thread(), "TAU_PERFSTUBS");
}

void ps_tool_stop_string(const char * name) {
    Tau_pure_stop_task(name, Tau_get_thread());
}

void ps_tool_stop_current(void) {
    Tau_global_stop();
}

void ps_tool_set_parameter(const char * parameter_name, int64_t parameter_value) {
    Tau_profile_param1l(parameter_value, parameter_name);
}

void ps_tool_dynamic_phase_start(const char * name, int index) {
    Tau_dynamic_start(name, index);
}

void ps_tool_dynamic_phase_stop(const char * name, int index) {
    Tau_dynamic_stop(name, index);
}

void * ps_tool_create_counter(const char * name) {
    return Tau_get_userevent(name);
}

void ps_tool_sample_counter(const void * counter, double value) {
    Tau_userevent((void*)counter, value);
}

void ps_tool_set_metadata(const char * name, const char * value) {
    Tau_metadata(name, value);
}

void ps_tool_get_timer_data(ps_tool_timer_data_t *timer_data)
{
    memset(timer_data, 0, sizeof(ps_tool_timer_data_t));
    // get the most up-to-date profile information
    TauProfiler_updateAllIntermediateStatistics();
    RtsLayer::LockDB();

    const char **counterNames;
    int numCounters;
    TauMetrics_getCounterList(&counterNames, &numCounters);
    int trueMetrics = (numCounters * 2) + 1;

    timer_data->num_timers = TheFunctionDB().size();
    timer_data->num_threads = RtsLayer::getTotalThreads();
    timer_data->num_metrics = trueMetrics;
    timer_data->timer_names = (char **)(calloc(TheFunctionDB().size(),
        sizeof(char *)));
    timer_data->metric_names = (char **)(calloc(trueMetrics,
        sizeof(char *)));
    timer_data->values = (double *)(calloc(TheFunctionDB().size() *
        RtsLayer::getTotalThreads() * trueMetrics, sizeof(double)));
    timer_data->metric_names[0] = strdup("Calls");
    for (int m = 0; m < numCounters; m++) {
        std::stringstream ss;
        ss << "Inclusive_" << counterNames[m];
        timer_data->metric_names[(m*2)+1] = strdup(ss.str().c_str());
        std::stringstream ss2;
        ss2 << "Exclusive_" << counterNames[m];
        timer_data->metric_names[(m*2)+2] = strdup(ss2.str().c_str());
    }

    // get the FunctionInfo database, and iterate over it
    std::vector<FunctionInfo*>::const_iterator it;
    //foreach: TIMER
    int t_index = 0;
    int v_index = 0;
    for (it = TheFunctionDB().begin(); it != TheFunctionDB().end(); it++) {
        FunctionInfo *fi = *it;
        timer_data->timer_names[t_index++] = strdup(fi->GetName());
        for (int tid = 0; tid < RtsLayer::getTotalThreads(); tid++) {
            timer_data->values[v_index++] = fi->GetCalls(tid);
            for (int m = 0; m < numCounters; m++) {
                timer_data->values[v_index++] =
                    fi->getDumpInclusiveValues(tid)[m];
                timer_data->values[v_index++] =
                    fi->getDumpExclusiveValues(tid)[m];
            }
        }
    }
    RtsLayer::UnLockDB();
    return;
}

    void ps_tool_free_timer_data(ps_tool_timer_data_t *timer_data)
    {
        if (timer_data == NULL)
        {
            return;
        }
        if (timer_data->timer_names != NULL)
        {
            free(timer_data->timer_names);
            timer_data->timer_names = NULL;
        }
        if (timer_data->metric_names != NULL)
        {
            free(timer_data->metric_names);
            timer_data->metric_names = NULL;
        }
        if (timer_data->values != NULL)
        {
            free(timer_data->values);
            timer_data->values = NULL;
        }
    }

void ps_tool_get_counter_data(ps_tool_counter_data_t *counter_data)
{
    memset(counter_data, 0, sizeof(ps_tool_counter_data_t));
    RtsLayer::LockDB();
    tau::AtomicEventDB tmpCounters(tau::TheEventDB());
    RtsLayer::UnLockDB();
    tau::AtomicEventDB::const_iterator counterIterator;
    counter_data->num_counters = tmpCounters.size();
    counter_data->num_threads = RtsLayer::getTotalThreads();
    counter_data->counter_names = (char **)(calloc(
        tmpCounters.size() * RtsLayer::getTotalThreads(), sizeof(char *)));
    counter_data->num_samples = (double *)(calloc(
        tmpCounters.size() * RtsLayer::getTotalThreads(), sizeof(double)));
    counter_data->value_total = (double *)(calloc(
        tmpCounters.size() * RtsLayer::getTotalThreads(), sizeof(double)));
    counter_data->value_min = (double *)(calloc(
        tmpCounters.size() * RtsLayer::getTotalThreads(), sizeof(double)));
    counter_data->value_max = (double *)(calloc(
        tmpCounters.size() * RtsLayer::getTotalThreads(), sizeof(double)));
    counter_data->value_sumsqr = (double *)(calloc(
        tmpCounters.size() * RtsLayer::getTotalThreads(), sizeof(double)));
    int c_index = 0;
    int v_index = 0;
    for (counterIterator = tmpCounters.begin();
         counterIterator != tmpCounters.end(); counterIterator++) {
        tau::TauUserEvent *ue = (*counterIterator);
        if (ue == NULL) continue;
        counter_data->counter_names[c_index++] = strdup(ue->GetName().c_str());
        for (int tid = 0; tid < RtsLayer::getTotalThreads(); tid++) {
            counter_data->num_samples[v_index] = ue->GetNumEvents(tid);
            counter_data->value_total[v_index] = ue->GetSum(tid);
            counter_data->value_max[v_index] = ue->GetMax(tid);
            counter_data->value_min[v_index] = ue->GetMin(tid);
            counter_data->value_sumsqr[v_index] = ue->GetSumSqr(tid);
            v_index++;
        }
    }
    return;
}

    void ps_tool_free_counter_data(ps_tool_counter_data_t *counter_data)
    {
        if (counter_data == NULL)
        {
            return;
        }
        if (counter_data->counter_names != NULL)
        {
            free(counter_data->counter_names);
            counter_data->counter_names = NULL;
        }
        if (counter_data->num_samples != NULL)
        {
            free(counter_data->num_samples);
            counter_data->num_samples = NULL;
        }
        if (counter_data->value_total != NULL)
        {
            free(counter_data->value_total);
            counter_data->value_total = NULL;
        }
        if (counter_data->value_min != NULL)
        {
            free(counter_data->value_min);
            counter_data->value_min = NULL;
        }
        if (counter_data->value_max != NULL)
        {
            free(counter_data->value_max);
            counter_data->value_max = NULL;
        }
        if (counter_data->value_sumsqr != NULL)
        {
            free(counter_data->value_sumsqr);
            counter_data->value_sumsqr = NULL;
        }
    }

void ps_tool_get_metadata(ps_tool_metadata_t *metadata)
{
    memset(metadata, 0, sizeof(ps_tool_metadata_t));
    metadata->num_values = 0;
    for (int tid = 0; tid < RtsLayer::getTotalThreads() ; tid++) {
        metadata->num_values = metadata->num_values +
            Tau_metadata_getMetaData(tid).size();
    }
    metadata->names = (char **)(calloc(metadata->num_values, sizeof(char *)));
    metadata->values = (char **)(calloc(metadata->num_values, sizeof(char *)));
    int v_index = 0;
    for (int tid = 0; tid < RtsLayer::getTotalThreads() ; tid++) {
        for (MetaDataRepo::iterator it = Tau_metadata_getMetaData(tid).begin();
            it != Tau_metadata_getMetaData(tid).end(); it++) {
            std::stringstream ss;
            ss << "Thread " << tid << ":" << it->first.name;
            metadata->names[v_index] = strdup(ss.str().c_str());
            std::stringstream ss2;
            switch(it->second->type) {
                case TAU_METADATA_TYPE_STRING:
                    metadata->values[v_index] = strdup(it->second->data.cval);
                    break;
                case TAU_METADATA_TYPE_INTEGER:
                    ss2 << it->second->data.ival;
                    metadata->values[v_index] = strdup(ss2.str().c_str());
                    break;
                case TAU_METADATA_TYPE_DOUBLE:
                    ss2 << it->second->data.dval;
                    metadata->values[v_index] = strdup(ss2.str().c_str());
                    break;
                case TAU_METADATA_TYPE_TRUE:
                    metadata->values[v_index] = strdup("true");
                    break;
                case TAU_METADATA_TYPE_FALSE:
                    metadata->values[v_index] = strdup("false");
                    break;
                case TAU_METADATA_TYPE_NULL:
                    metadata->values[v_index] = strdup("(null)");
                    break;
                default:
                    break;
            }
            v_index++;
        }
    }
    return;
}

    void ps_tool_free_metadata(ps_tool_metadata_t *metadata)
    {
        if (metadata == NULL)
        {
            return;
        }
        if (metadata->names != NULL)
        {
            free(metadata->names);
            metadata->names = NULL;
        }
        if (metadata->values != NULL)
        {
            free(metadata->values);
            metadata->values = NULL;
        }
    }

} // extern "C"

