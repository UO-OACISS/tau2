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
#include <sdskv-client.h>

/* So that we can keep track of how long this plugin takes */
#include "Tau_scoped_timer.h"

static bool enabled{false};
static bool initialized{false};
static bool opened{false};
static bool done{false};
static margo_instance_id mid;
static hg_addr_t svr_addr;
static std::string svr_addr_str;
static sdskv_database_id_t db_id;
static std::string db_name;
static uint8_t mplex_id;
static sdskv_client_t sdskv_cl;
static sdskv_provider_handle_t provider_handle;
static char *svr_addr_file;


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
    char *proto;
    char *colon;
    int ret;

    /* initialize Mercury using the transport portion of the destination
     * address (i.e., the part before the first : character if present)
     */
    std::ifstream fp( svr_addr_file );
    std::getline(fp, svr_addr_str);

    proto = strdup(svr_addr_str.c_str());
    assert(proto);
    colon = strchr(proto, ':');
    if(colon)
        *colon = '\0';
    mid = margo_init(proto, MARGO_CLIENT_MODE, 0, 0);
    if (mid == MARGO_INSTANCE_NULL)
    {
	std::cerr << "Error: margo_init()" << std::endl;
	ret = -1;
        return;
    }

    ret = sdskv_client_init(mid, &sdskv_cl);
    if (ret != 0)
    {
	std::cerr << "Error: sdskv_client_init()" << std::endl;
	margo_finalize(mid);
	ret = -1;
        return;
    }

    if (ret == 0) { initialized = true; }
    assert(ret == 0);
}

void Tau_plugin_mochi_open_file(void) {
    /* open mochi connection */

    int ret;
    hg_return_t hret = margo_addr_lookup(mid, svr_addr_str.c_str(), &svr_addr);
    if (hret != HG_SUCCESS)
    {
	std::cerr << "Error: margo_addr_lookup()" << std::endl;
	sdskv_client_finalize(sdskv_cl);
	margo_finalize(mid);
        ret = -1;
	return;
    }

    ret = sdskv_provider_handle_create(sdskv_cl, svr_addr, mplex_id, &provider_handle);

    if (ret != 0)
    {
	std::cerr << "Error: sdskv_provider_handle_create()" << std::endl;
	margo_addr_free(mid, svr_addr);
	sdskv_client_finalize(sdskv_cl);
	margo_finalize(mid);
        ret = -1;
	return;
    }

    ret = sdskv_open(provider_handle, db_name.c_str(), &db_id);
    if (ret != 0)
    {
	std::cerr << "Error: could not open database " <<  db_name << std::endl;
	sdskv_provider_handle_release(provider_handle);
	margo_addr_free(mid, svr_addr);
	sdskv_client_finalize(sdskv_cl);
	margo_finalize(mid);
        ret = -1 ;
	return;
    }
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
 * per-thread metrics and counters.
 * Subsequently, contacts the remote Mochi SDSKV db to write metrics and counters values
 * on a per-thread basis using two respective sdskv_put_multi() RPC calls. Note that put_multi()
 * is more efficient than a sequence of put() calls as in saves the unnecessary roundtrip RPC times */
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
    std::vector<std::string>  m_keys;
    std::vector<std::string>  m_vals;
    std::vector<hg_size_t>   m_ksizes, m_vsizes;
    std::vector<const void*> m_kptrs, m_vptrs;

    size_t metric_size = 3*numThreadsLocal*tmpTimers.size();

    RESERVE(m_keys, metric_size);
    RESERVE(m_vals, metric_size);
    RESERVE(m_ksizes, metric_size);
    RESERVE(m_kptrs, metric_size);
    RESERVE(m_vsizes, metric_size);
    RESERVE(m_vptrs, metric_size);

    int rank = RtsLayer::myNode();

    //foreach: TIMER
    std::vector<FunctionInfo*>::const_iterator it;
    for (it = tmpTimers.begin(); it != tmpTimers.end(); it++) {
        FunctionInfo *fi = *it;

        stringstream ss;
        std::string shortName(fi->GetName());
        shorten_timer_name(shortName);
        ss << shortName;
        ss << "_Calls_";
        ss << rank;

        // assign real data
        for (int tid = 0; tid < numThreadsLocal; tid++) {
            /* build a name with ss, tid */
            /* write the fi->GetCalls(tid) value */
            ss << "_";
            ss << tid;
            UPDATE_KEY(m_keys, ss.str());
            stringstream val;
            val << (double)fi->GetCalls(tid);
            UPDATE_VAL(m_vals, val.str());
        }

        for (int m = 0 ; m < numCounters.size() ; m++) {
            stringstream incl;
            stringstream excl;
            incl << shortName;
            incl << "_Inclusive_";
            incl << counterNames[m] << "_";
            incl << rank;
            excl << shortName;
            excl <<  "_Exclusive_";
            excl << counterNames[m];
            excl << rank << "_";
            // assign real data
            for (int tid = 0; tid < numThreadsLocal; tid++) {
                /* build a name with incl, tid */
                /* write the fi->getDumpInclusiveValues(tid)[m] value */
                /* build a name with excl, tid */
                /* write the fi->getDumpExclusiveValues(tid)[m] value */
                incl << "_" << tid;
		excl << "_" << tid;
                stringstream val1, val2;
                if(fi->GetCalls(tid) == 0) {
                  val1 << 0.0;
                  val2 << 0.0;
                } else {
                  val1 << (fi->getDumpInclusiveValues(tid)[m]);
                  val2 << (fi->getDumpExclusiveValues(tid)[m]);
                }
                UPDATE_KEY(m_keys, incl.str());
                UPDATE_VAL(m_vals, val1.str());
                UPDATE_KEY(m_keys, excl.str());
                UPDATE_VAL(m_vals, val2.str());
            }
        }
    }

    for (int i = 0 ; i < 3*numThreadsLocal*tmpTimers.size(); i++) {
        m_ksizes[i] = m_keys[i].size();
        m_kptrs[i]  = m_keys[i].data();
        m_vsizes[i] = m_vals[i].size();
        m_vptrs[i]  = m_vals[i].data();
    }

    //Make a sdskv-put-multi call to the Mochi db
    int ret = sdskv_put_multi(provider_handle, db_id, 3*numThreadsLocal*tmpTimers.size(), m_kptrs.data(), m_ksizes.data(), m_vptrs.data(), m_vsizes.data());
    assert(ret == 0);

    /* Lock the counter map */
    RtsLayer::LockDB();
    tau::AtomicEventDB::const_iterator it2;
    std::map<std::string, std::vector<double> >::iterator counter_map_it;
    std::vector<std::string>  m_counter_keys;
    std::vector<double>  m_counter_vals;
    std::vector<hg_size_t>   m_counter_ksizes, m_counter_vsizes;
    std::vector<const void*> m_counter_kptrs, m_counter_vptrs;

    size_t counter_size = 5*numThreadsLocal*tau::TheEventDB().size();

    RESERVE(m_counter_keys, counter_size);
    RESERVE(m_counter_vals, counter_size);
    RESERVE(m_counter_ksizes, counter_size);
    RESERVE(m_counter_vsizes, counter_size);
    RESERVE(m_counter_kptrs, counter_size);
    RESERVE(m_counter_vptrs, counter_size);

    // do the same with counters.
    for (it2 = tau::TheEventDB().begin(); it2 != tau::TheEventDB().end(); it2++) {
        tau::TauUserEvent *ue = (*it2);
        if (ue == NULL) continue;
        std::string counter_name(ue->GetName().c_str());
        std::stringstream ss, mean, min, max, sumsqr;
        ss << counter_name << "_NumEvents";
        mean << counter_name << "_Mean";
        min << counter_name << "_Min";
        max << counter_name << "_Max";
        sumsqr << counter_name << "_SumSquares";

        // assign real data
        for (int tid = 0; tid < numThreadsLocal; tid++) {
            /* as above, build a key and write out:
            ue->GetNumEvents(tid);
            ue->GetMean(tid);
            ue->GetMax(tid);
            ue->GetMin(tid);
            ue->GetSumSqr(tid);
            */
            ss << "_" << tid; mean << "_" << tid;
            min << "_" << tid; max << "_" << tid;
            sumsqr << "_" << tid;
            UPDATE_KEY(m_counter_keys, ss.str());
            UPDATE_KEY(m_counter_keys, mean.str());
            UPDATE_KEY(m_counter_keys, min.str());
            UPDATE_KEY(m_counter_keys, max.str());
            UPDATE_KEY(m_counter_keys, sumsqr.str());

            double num_val, mean_val, min_val, max_val, sumsqr_val;
            num_val = ((double)ue->GetNumEvents(tid));
            mean_val = ((double)ue->GetMean(tid));
            min_val = ((double)ue->GetMin(tid));
            max_val = ((double)ue->GetMax(tid));
            sumsqr_val = ((double)ue->GetSumSqr(tid));

            UPDATE_VAL(m_counter_vals, num_val);
            UPDATE_VAL(m_counter_vals, mean_val);
            UPDATE_VAL(m_counter_vals, min_val);
            UPDATE_VAL(m_counter_vals, max_val);
            UPDATE_VAL(m_counter_vals, sumsqr_val);

        }
    }

    double *counter_vptrs = m_counter_vals.data();
    for (int i = 0 ; i < 5*numThreadsLocal*tau::TheEventDB().size(); i++) {
        m_counter_ksizes[i] = m_counter_keys[i].size();
        m_counter_kptrs[i]  = m_counter_keys[i].data();
        m_counter_vsizes[i] = sizeof(double);
        m_counter_vptrs[i]  = (void*)&counter_vptrs[i];
    }

    /* unlock the counter map */
    RtsLayer::UnLockDB();

    //Make a sdskv-put-multi call to the Mochi db
    ret = sdskv_put_multi(provider_handle, db_id, 5*numThreadsLocal*m_counter_ksizes.size(), m_counter_kptrs.data(), m_counter_ksizes.data(), m_counter_vptrs.data(), m_counter_vsizes.data());
    assert(ret == 0);
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

    assert(argc == 3);
    svr_addr_file = strdup(argv[0]);
    mplex_id = atoi(argv[1]);
    char * db_name_ = strdup(argv[2]);
    removeSpaces(db_name_);
    db_name = db_name_;

    /* Register the callback object */
    TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(&cb, id);
    enabled = true;
    return 0;
}


#endif // TAU_MOCHI
