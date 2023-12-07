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
#include <sdskv-server.h>

/* So that we can keep track of how long this plugin takes */
#include "Tau_scoped_timer.h"

static bool enabled{false};
static bool initialized{false};
static bool opened{false};
static bool done{false};
static margo_instance_id mid;
static symbiomon_provider_t provider;
static sdskv_provider_t sdskv_provider;
static int reduction_frequency = 0;
static int my_rank = 0;
static int size = 0;
static int numSomaRanks = 0;

typedef enum {
    MODE_DATABASES = 0,
    MODE_PROVIDERS = 1
} kv_mplex_mode_t;

struct options
{
    char *listen_addr_str;
    unsigned num_db;
    char *db_name;
    sdskv_db_type_t db_type;
    char *host_file;
    kv_mplex_mode_t mplex_mode;
};

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
    int ret;
    /* initialize the local mochi provider */
    mid = margo_init("ofi+verbs://", MARGO_SERVER_MODE, 0, -1);
    if (mid == MARGO_INSTANCE_NULL)
    {
	std::cerr << "Error: margo_init()" << std::endl;
	ret = -1;
        return;
    }

    my_rank = RtsLayer::myNode();
    size = tau_totalnodes(0,1);
    assert (getenv("TAU_NUM_SOMA_RANKS") != NULL);
    numSomaRanks = atoi(getenv("TAU_NUM_SOMA_RANKS"));

    /* initialize the SDSKV server */
    struct options opts;
    opts.num_db = 1;
    char db_name_[] = "foo";
    char host_file_[] = "sdskv.add";
    opts.db_name = db_name_;
    opts.db_type = KVDB_MAP;
    opts.host_file = host_file_;

    int i;
    struct sdskv_provider_init_info sdskv_args
        = SDSKV_PROVIDER_INIT_INFO_INIT;
        sdskv_args.json_config = NULL;
        sdskv_args.rpc_pool = SDSKV_ABT_POOL_DEFAULT;

    if(my_rank >= (size - numSomaRanks)) {
	ret = sdskv_provider_register(mid, 1,
                &sdskv_args,
                &sdskv_provider);

        if(ret != 0)
        {
            fprintf(stderr, "Error: sdskv_provider_register()\n");
            margo_finalize(mid);
            return;
        }
    }

    sdskv_database_id_t db_id;
    char* path_ = opts.db_name;
    char* x = strrchr(path_, '/');
    char* db_name = path_;
    if(x != NULL) {
        db_name = x+1;
        *x = '\0';
    }

    char rank_str[20];
    sprintf(rank_str, "%d", my_rank);
    char * path = (char*)malloc((strlen(path_)+20)*sizeof(char));
    strcpy(path, path_);
    strcat(path, rank_str);

    sdskv_config_t db_config = {
        .db_name = db_name,
        .db_path = (x == NULL ? "" : path),
        .db_type = opts.db_type,
        .db_comp_fn_name = SDSKV_COMPARE_DEFAULT,
        .db_no_overwrite = 0
    };

    ret = 0;
    if(my_rank >= (size - numSomaRanks)) {
        ret = sdskv_provider_attach_database(sdskv_provider, &db_config, &db_id);
    }

    if(ret != 0)
    {
        fprintf(stderr, "Error: sdskv_provider_attach_database()\n");
        margo_finalize(mid);
        return;
    }


    if(opts.host_file)
    {
        /* write the server address to file if requested */
        FILE *fp;
        hg_addr_t self_addr;
        char self_addr_str[128];
        hg_size_t self_addr_str_sz = 128;
        hg_return_t hret;

        /* figure out what address this server is listening on */
        hret = margo_addr_self(mid, &self_addr);
        if(hret != HG_SUCCESS)
        {
            fprintf(stderr, "Error: margo_addr_self()\n");
            margo_finalize(mid);
            return;
        }
        hret = margo_addr_to_string(mid, self_addr_str, &self_addr_str_sz, self_addr);
        if(hret != HG_SUCCESS)
        {
            fprintf(stderr, "Error: margo_addr_to_string()\n");
            margo_addr_free(mid, self_addr);
            margo_finalize(mid);
            return;
        }

        // Write addresses to a file
        if(my_rank == size - numSomaRanks) {
            fp = fopen(opts.host_file, "w");
	    if(!fp)
            {
	    	perror("fopen");
	        margo_finalize(mid);
	        return;
	    }
	    fprintf(fp, "%d\n", numSomaRanks);
	    fclose(fp);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	int i = 0;
	for(i = size - numSomaRanks; i < size; i++) {
        	if(my_rank == i) {
		    fprintf(stderr, "Rank: %d is writing out its address..\n", i);
	            fp = fopen(opts.host_file, "a");
	            if(!fp)
        	    {
	                perror("fopen");
	                margo_finalize(mid);
	                return;
	            }

	            fprintf(fp, "%s %d %s\n", self_addr_str, 1, db_name);
		    fflush(fp);

	            fclose(fp);
 	       }
	       MPI_Barrier(MPI_COMM_WORLD);
	}

        margo_addr_free(mid, self_addr);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* register the COLLECTOR provider */
     struct symbiomon_provider_args args = SYMBIOMON_PROVIDER_ARGS_INIT;
    args.push_finalize_callback = 0;
    //args.pool = pool;

    if(my_rank < size - numSomaRanks) {
      ret = symbiomon_provider_register(mid, 42, &args, &provider);
      if (ret != 0)
        {
	    std::cerr << "Error: symbiomon_provider_register()" << std::endl;
	    margo_finalize(mid);
	    ret = -1;
            return;
        }
      assert(ret == 0);
    }

    if (ret == 0) { initialized = true; }

    /* If I am an AGGREGATOR, I enter the infinite progress loop */
    if(my_rank >= size - numSomaRanks) {
       margo_wait_for_finalize(mid);
    }
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

    stringstream rank_;
    rank_ << my_rank;
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
	//MPI_Barrier(MPI_COMM_WORLD);
	/* Perform global aggregation if rank 0*/
	/* if(!rank) {
            fprintf(stderr, "SYMBIOMON: Performing metric aggregation.\n");
	    symbiomon_metric_global_reduce_all(provider, size);
	}*/
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

    if (opened and (my_rank < (size - numSomaRanks))) {
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
