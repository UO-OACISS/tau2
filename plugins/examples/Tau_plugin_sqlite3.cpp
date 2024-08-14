/************************************************************************************************
 * *   Plugin Testing
 * *   Tests basic functionality of a plugin for function registration event
 * *
 * *********************************************************************************************/


#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <regex>

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>
#include <Profile/TauMetaData.h>
#include <pthread.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

#ifdef TAU_MPI
#include <mpi.h>
#endif

#include <sqlite3.h>
#include "Tau_plugin_sqlite3_schema.h"

bool done = false;
sqlite3 *db;
char *zErrMsg = 0;
int rc = SQLITE_OK;
int comm_rank = 0;
int comm_size = 1;
std::unordered_map<int,size_t> thread_map;
std::unordered_map<std::string,size_t> metric_map;
std::unordered_map<std::string,size_t> timer_map;
std::unordered_map<std::string,size_t> counter_map;

typedef struct context_event_index {
    size_t counter_id;
    bool has_context;
    size_t timer_id;
} context_event_index_t;

inline bool file_exists (const char * name) {
  struct stat buffer;
  return (stat (name, &buffer) == 0);
}

static int callback(void *NotUsed, int argc, char **argv, char **azColName) {
   int i;
   for(i = 0; i<argc; i++) {
      printf("%s = %s\n", azColName[i], argv[i] ? argv[i] : "NULL");
   }
   printf("\n");
   return 0;
}

bool open_database() {
    std::stringstream ss;
    ss << TauEnv_get_profiledir() << "/tauprofile.db";
    char * filename = strdup(ss.str().c_str());
    /* check if file exists */
    bool exists = file_exists(filename);
    /* open the database */
    rc = sqlite3_open(filename, &db);

    if( rc ) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        return false;
    } else {
        TAU_VERBOSE("Opened database %s successfully\n", filename);
    }
    free(filename);
    return (exists);
}

void create_database() {
    bool exists = open_database();
    if (exists) {
        TAU_VERBOSE("TAU database exists, adding trial to it\n");
    } else {
        rc = sqlite3_exec(db, database_schema, callback, 0, &zErrMsg);

        if( rc != SQLITE_OK ){
            fprintf(stderr, "SQL error creating schema!: %s\n", zErrMsg);
            sqlite3_free(zErrMsg);
            return;
        } else {
            TAU_VERBOSE("TAU database created successfully\n");
        }
    }
    return;
}

void begin_transaction() {
    std::stringstream sql;
    sql << "begin transaction;";
    rc = sqlite3_exec(db, sql.str().c_str(), callback, 0, &zErrMsg);

    if( rc != SQLITE_OK ){
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        fprintf(stderr, "failed query: %s\n", sql.str().c_str());
        sqlite3_free(zErrMsg);
    } else {
        //TAU_VERBOSE("Begin Transaction\n");
    }
}

void end_transaction() {
    std::stringstream sql;
    sql << "commit transaction;";
    rc = sqlite3_exec(db, sql.str().c_str(), callback, 0, &zErrMsg);

    if( rc != SQLITE_OK ){
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        fprintf(stderr, "failed query: %s\n", sql.str().c_str());
        sqlite3_free(zErrMsg);
    } else {
        //TAU_VERBOSE("Commit Transaction\n");
    }
}

size_t store_trial() {
    std::stringstream sql;
    size_t trial_id = 0;
    /* Get the executable name from the TAU metadata */
    Tau_metadata_key key;
    key.name = strdup("Executable");
    const char * execname = Tau_metadata_getMetaData(0).at(key)->data.cval;
    sql << "insert into trial (name) values ('" << execname << "');";
    rc = sqlite3_exec(db, sql.str().c_str(), callback, 0, &zErrMsg);

    if( rc != SQLITE_OK ){
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        fprintf(stderr, "failed query: %s\n", sql.str().c_str());
        sqlite3_free(zErrMsg);
    } else {
        trial_id = sqlite3_last_insert_rowid(db);
        TAU_VERBOSE("Trial %lu inserted successfully\n", trial_id);
    }
    return trial_id;
}

void store_threads(size_t trial_id) {
    int nodeid = TAU_PROFILE_GET_NODE();
    int numThreads = RtsLayer::getTotalThreads();
    for(int t = 0 ; t < numThreads ; t++) {
        std::stringstream sql;
        size_t thread_id;
        sql << "insert into thread (trial, node_rank, context_rank, thread_rank) values ("
        << trial_id << ","
        << nodeid << ","
        << 0 << ","
        << t << ");";
        rc = sqlite3_exec(db, sql.str().c_str(), callback, 0, &zErrMsg);

        if( rc != SQLITE_OK ){
            fprintf(stderr, "SQL error: %s\n", zErrMsg);
            fprintf(stderr, "failed query: %s\n", sql.str().c_str());
            sqlite3_free(zErrMsg);
        } else {
            thread_id = sqlite3_last_insert_rowid(db);
            //TAU_VERBOSE("Thread %lu inserted successfully\n", thread_id);
        }
        thread_map[t] = thread_id;
    }
}

void store_metadata(size_t trial_id) {
    int numThreads = RtsLayer::getTotalThreads();
    for(int t = 0 ; t < numThreads ; t++) {
        for (MetaDataRepo::iterator it = Tau_metadata_getMetaData(t).begin();
            it != Tau_metadata_getMetaData(t).end(); it++) {
            std::string value;
            switch(it->second->type) {
                case TAU_METADATA_TYPE_STRING:
                    value = std::string(it->second->data.cval);
                    break;
                case TAU_METADATA_TYPE_INTEGER:
                    value = std::to_string(it->second->data.ival);
                    break;
                case TAU_METADATA_TYPE_DOUBLE:
                    value = std::to_string(it->second->data.dval);
                    break;
                case TAU_METADATA_TYPE_TRUE:
                    value = std::string("true");
                    break;
                case TAU_METADATA_TYPE_FALSE:
                    value = std::string("false");
                    break;
                case TAU_METADATA_TYPE_NULL:
                    value = std::string("(null)");
                    break;
                default:
                    break;
            }
            std::stringstream sql;
            size_t thread_id = thread_map[t];
            sql << "insert into metadata (trial, thread, name, value) values ("
            << trial_id << ","
            << thread_id << ",'"
            << it->first.name << "','"
            << value << "');";
            rc = sqlite3_exec(db, sql.str().c_str(), callback, 0, &zErrMsg);

            if( rc != SQLITE_OK ){
                fprintf(stderr, "SQL error: %s\n", zErrMsg);
                fprintf(stderr, "failed query: %s\n", sql.str().c_str());
                sqlite3_free(zErrMsg);
            } else {
                //TAU_VERBOSE("Metadata %lu %s %s inserted successfully\n", thread_id, it->first.name, value.c_str());
            }
        }
    }
}

void store_metric(size_t trial_id, const char * name) {
    std::stringstream sql;
    sql << "insert into metric (trial, name) values ("
    << trial_id << ",'"
    << name << "');";
    rc = sqlite3_exec(db, sql.str().c_str(), callback, 0, &zErrMsg);

    size_t metric_id = 0;
    if( rc != SQLITE_OK ){
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        fprintf(stderr, "failed query: %s\n", sql.str().c_str());
        sqlite3_free(zErrMsg);
    } else {
        metric_id = sqlite3_last_insert_rowid(db);
        //TAU_VERBOSE("Metric %s %lu inserted successfully\n", name, metric_id);
    }
    metric_map[std::string(name)] = metric_id;
}

void store_metrics(size_t trial_id) {
    store_metric(trial_id, "Number of Calls");
    //store_metric(trial_id, "subroutines");
    int numMetrics = 0; // Tau_Global_numCounters();
    const char **counterNames;
    TauMetrics_getCounterList(&counterNames, &numMetrics);
    for(int m = 0 ; m < numMetrics ; m++) {
        std::stringstream name;
        name << "Inclusive " << counterNames[m];
        store_metric(trial_id, name.str().c_str());
        name.str(std::string());
        name << "Exclusive " << counterNames[m];
        store_metric(trial_id, name.str().c_str());
    }
}

void shorten_timer_name(std::string& name) {
    std::size_t index = name.find(" [{");
    if (index != std::string::npos) {
        name = name.substr(0,index);
    }
}

size_t store_timer(size_t trial_id, std::string longName, bool has_parent, size_t parent_timer) {
    std::string shortName(longName.c_str());
    shorten_timer_name(shortName);
    std::stringstream sql;
    sql << "insert into timer (trial, name, short_name, parent) values ("
    << trial_id << ",'"
    << longName.c_str() << "','"
    << shortName.c_str() << "',";
    if (has_parent) {
        sql << parent_timer;
    } else {
        sql << "NULL";
    }
    sql << ");";
    rc = sqlite3_exec(db, sql.str().c_str(), callback, 0, &zErrMsg);

    size_t timer_id = 0;
    if( rc != SQLITE_OK ){
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        fprintf(stderr, "failed query: %s\n", sql.str().c_str());
        sqlite3_free(zErrMsg);
    } else {
        timer_id = sqlite3_last_insert_rowid(db);
        //TAU_VERBOSE("%s %lu %lu inserted successfully\n", longName.c_str(), parent_timer, timer_id);
    }
    return timer_id;
}

size_t get_or_store_timer(size_t trial_id, std::string tree_name, std::string name, bool has_parent, size_t parent_id) {
    if (timer_map.count(tree_name) > 0) {
        return timer_map[tree_name];
    } else {
        size_t timer_id = store_timer(trial_id, name, has_parent, parent_id);
        timer_map[tree_name] = timer_id;
        return timer_id;
    }
}

size_t store_counter(size_t trial_id, std::string name) {
    std::stringstream sql;
    sql << "insert into counter (trial, name) values ("
    << trial_id << ",'"
    << name.c_str() << "');";
    rc = sqlite3_exec(db, sql.str().c_str(), callback, 0, &zErrMsg);

    size_t counter_id = 0;
    if( rc != SQLITE_OK ){
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        fprintf(stderr, "failed query: %s\n", sql.str().c_str());
        sqlite3_free(zErrMsg);
    } else {
        counter_id = sqlite3_last_insert_rowid(db);
        //TAU_VERBOSE("%s %lu %lu inserted successfully\n", name.c_str(), parent_timer, timer_id);
    }
    return counter_id;
}

size_t get_or_store_counter(size_t trial_id, std::string name) {
    if (counter_map.count(name) > 0) {
        return counter_map[name];
    } else {
        size_t counter_id = store_counter(trial_id, name);
        counter_map[name] = counter_id;
        return counter_id;
    }
}

/* If the timer is a callpath:
 *   - split the name into tokens
 *   - lookup and/or store token[0]
 *   - lookup and/or store token[0] => token[1]
 *   - etc.
 */
size_t decompose_and_store_timer(size_t trial_id, const char * name) {
    std::string longName(name);
    std::size_t index = longName.find(" => ");
    if (index == std::string::npos) {
        return get_or_store_timer(trial_id, longName, longName, false, 0);
    } else {
        std::regex separator(" => ");
        std::sregex_token_iterator token(longName.begin(), longName.end(),
            separator, -1);
        std::sregex_token_iterator end;
        std::string name = *token;
        size_t parent_id = get_or_store_timer(trial_id, name, name, false, 0);
        std::stringstream ss;
        ss << name;
        while(++token != end) {
            std::string current = *token;
            ss << " => " << current;
            parent_id = get_or_store_timer(trial_id, ss.str(), current, true, parent_id);
        }
        return parent_id;
    }
}

void store_timer_value(size_t timer_id, size_t metric_id, size_t thread_id,
    double value) {
    std::stringstream sql;
    sql << "insert into timer_value (timer, metric, thread, value) values ("
    << timer_id << ","
    << metric_id << ","
    << thread_id << ","
    << value << ");";
    rc = sqlite3_exec(db, sql.str().c_str(), callback, 0, &zErrMsg);
    if( rc != SQLITE_OK ){
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        fprintf(stderr, "failed query: %s\n", sql.str().c_str());
        sqlite3_free(zErrMsg);
    } else {
        //TAU_VERBOSE("timer_value inserted successfully\n");
    }
}

void store_timers(size_t trial_id) {
    // get the FunctionInfo database, and iterate over it
    std::vector<FunctionInfo*>::const_iterator it;
    RtsLayer::LockDB();
    /* Copy the function info database so we can release the lock */
    std::vector<FunctionInfo*> tmpTimers(TheFunctionDB());
    RtsLayer::UnLockDB();

    std::map<std::string, std::vector<double> >::iterator timer_map_it;

    int numMetrics = 0; // Tau_Global_numCounters();
    const char **counterNames;
    TauMetrics_getCounterList(&counterNames, &numMetrics);

    //foreach: TIMER
    for (it = tmpTimers.begin(); it != tmpTimers.end(); it++) {
        FunctionInfo *fi = *it;
        size_t timer_id = decompose_and_store_timer(trial_id, fi->GetName());
        for(int t = 0 ; t < RtsLayer::getTotalThreads() ; t++) {
            size_t thread_id = thread_map[t];
            size_t metric_id = metric_map["Number of Calls"];
            long value = fi->GetCalls(t);
            if (value == 0L) { continue; } // this thread didn't call it
            store_timer_value(timer_id, metric_id, thread_id, (double)value);
            double inclusive[TAU_MAX_COUNTERS] = {0.0};
            double exclusive[TAU_MAX_COUNTERS] = {0.0};
            fi->getInclusiveValues(t, inclusive);
            fi->getExclusiveValues(t, exclusive);
            for(int m = 0 ; m < numMetrics ; m++) {
                std::stringstream name;
                name << "Inclusive " << counterNames[m];
                metric_id = metric_map[name.str()];
                store_timer_value(timer_id, metric_id, thread_id, inclusive[m]);
                name.str(std::string());
                name << "Exclusive " << counterNames[m];
                metric_id = metric_map[name.str()];
                store_timer_value(timer_id, metric_id, thread_id, exclusive[m]);
            }
        }
    }
}

context_event_index_t * decompose_and_store_counter(size_t trial_id, tau::TauUserEvent *ue) {
    std::string longName(ue->GetName().c_str());
    std::size_t index = longName.find(" : ");
    context_event_index_t * tmp = (context_event_index_t*)calloc(1, sizeof(context_event_index_t));
    if (index == std::string::npos) {
        tmp->counter_id = get_or_store_counter(trial_id, longName);
        tmp->has_context = false;
    } else {
        std::regex separator(" : ");
        std::sregex_token_iterator token(longName.begin(), longName.end(),
            separator, -1);
        std::sregex_token_iterator end;
        std::string name = *token;
        tmp->counter_id = get_or_store_counter(trial_id, name);
        if(++token != end) {
            std::string context = *token;
            tmp->has_context = true;
            tmp->timer_id = decompose_and_store_timer(trial_id, context.c_str());
        }
    }
    return tmp;
}

void store_counter_value(context_event_index_t * context, size_t thread_id, tau::TauUserEvent * ue, int tid) {
    std::stringstream sql;
    sql << "insert into counter_value (counter, timer, thread, sample_count, maximum_value, minimum_value, mean_value, sum_of_squares) values ("
    << context->counter_id << ",";
    if (context->has_context) {
        sql << context->timer_id << ",";
    } else {
        sql << "NULL" << ",";
    }
    sql << thread_id << ","
    << ue->GetNumEvents(tid) << ","
    << ue->GetMax(tid) << ","
    << ue->GetMin(tid) << ","
    << ue->GetMean(tid) << ","
    << ue->GetSumSqr(tid) << ");";
    rc = sqlite3_exec(db, sql.str().c_str(), callback, 0, &zErrMsg);
    if( rc != SQLITE_OK ){
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        fprintf(stderr, "failed query: %s\n", sql.str().c_str());
        sqlite3_free(zErrMsg);
    } else {
        //TAU_VERBOSE("counter_value inserted successfully\n");
    }
}

void store_counters(size_t trial_id) {
    int numEvents = 0;
    double max, min, mean, sumsqr;

    RtsLayer::LockDB();
    tau::AtomicEventDB::const_iterator it2;

    //foreach: Counter
    for (it2 = tau::TheEventDB().begin(); it2 != tau::TheEventDB().end(); it2++) {
        tau::TauUserEvent *ue = (*it2);
        if (ue == NULL) continue;
        context_event_index_t * context = decompose_and_store_counter(trial_id, ue);
        for(int t = 0 ; t < RtsLayer::getTotalThreads() ; t++) {
            size_t thread_id = thread_map[t];
            long value = ue->GetNumEvents(t);
            if (value == 0L) { continue; } // this thread didn't call it
            store_counter_value(context, thread_id, ue, t);
        }
        free(context);
    }
    RtsLayer::UnLockDB();
}

size_t store_profile(size_t trial_id) {
    begin_transaction();
    if (trial_id == 0UL) {
        trial_id = store_trial();
    }
    store_threads(trial_id);
    store_metadata(trial_id);
    store_metrics(trial_id);
    store_timers(trial_id);
    store_counters(trial_id);
    end_transaction();
    return trial_id;
}

void close_database() {
    sqlite3_close(db);
}

void write_profile_to_database() {
    if (done) { return; }
    if (RtsLayer::myThread() != 0) { return; }
    TauProfiler_updateAllIntermediateStatistics();
    size_t trial_id = 0;
    if (comm_rank <= 0) {
        create_database();
        trial_id = store_profile(0UL);
        close_database();
#ifdef TAU_MPI
        if (comm_size > 1) {
            printf("Rank 0 sending to rank 1\n"); fflush(stdout);
            PMPI_Send(&trial_id, 1, MPI_UNSIGNED_LONG, comm_rank+1, 1, MPI_COMM_WORLD);
        }
#endif
    } else {
#ifdef TAU_MPI
        MPI_Status status;
        printf("Rank %d receiving from rank %d\n", comm_rank, comm_rank-1); fflush(stdout);
        PMPI_Recv(&trial_id, 1, MPI_UNSIGNED_LONG, comm_rank-1, 1, MPI_COMM_WORLD, &status);
#endif
        open_database();
        trial_id = store_profile(trial_id);
        close_database();
#ifdef TAU_MPI
        if (comm_rank+1 < comm_size) {
            printf("Rank %d sending to rank %d\n", comm_rank, comm_rank+1); fflush(stdout);
            PMPI_Send(&trial_id, 1, MPI_UNSIGNED_LONG, comm_rank+1, 1, MPI_COMM_WORLD);
        }
#endif
    }
    done = true;
}

int Tau_plugin_event_end_of_execution_sqlite3(Tau_plugin_event_end_of_execution_data_t *data) {
    //printf("NULL PLUGIN %s\n", __func__);
    write_profile_to_database();
    return 0;
}

int Tau_plugin_event_pre_end_of_execution_sqlite3(Tau_plugin_event_pre_end_of_execution_data_t* data) {
    //printf("NULL PLUGIN %s\n", __func__);
#ifdef TAU_MPI
    write_profile_to_database();
#endif
    return 0;
}

int Tau_plugin_metadata_registration_complete_sqlite3(Tau_plugin_event_metadata_registration_data_t* data) {
    //printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_post_init_sqlite3(Tau_plugin_event_post_init_data_t* data) {
    //printf("NULL PLUGIN %s\n", __func__);
    comm_rank = RtsLayer::myNode();
    comm_size = tau_totalnodes(0,1);
    return 0;
}

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
    Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
    TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

    /* Required event support */
    /*
    cb->Trigger = Tau_plugin_event_trigger_sqlite3;
    cb->Dump = Tau_plugin_event_dump_sqlite3;
    cb->MetadataRegistrationComplete = Tau_plugin_metadata_registration_complete_sqlite3;
    */
    cb->PostInit = Tau_plugin_event_post_init_sqlite3;
    cb->PreEndOfExecution = Tau_plugin_event_pre_end_of_execution_sqlite3;
    cb->EndOfExecution = Tau_plugin_event_end_of_execution_sqlite3;

    /* Trace events */
    /*
    cb->Send = Tau_plugin_event_send_sqlite3;
    cb->Recv = Tau_plugin_event_recv_sqlite3;
    cb->FunctionEntry = Tau_plugin_event_function_entry_sqlite3;
    cb->FunctionExit = Tau_plugin_event_function_exit_sqlite3;
    cb->AtomicEventTrigger = Tau_plugin_event_atomic_trigger_sqlite3;
    */

    TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);

    return 0;
}

