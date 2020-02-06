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

#include <Profile/Profiler.h>
#include <Profile/TauSampling.h>
#include <Profile/TauMetrics.h>
#include <Profile/TauAPI.h>
#include <Profile/TauPlugin.h>
#include <pthread.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

#ifdef TAU_MPI
#include <mpi.h>
#endif

#include <sqlite3.h>
#include "Tau_plugin_sqlite_schema.h"

bool done = false;;
sqlite3 *db;
char *zErrMsg = 0;
int rc = SQLITE_OK;
int comm_rank = 0;
int comm_size = 1;
std::unordered_map<int,size_t> thread_map;
std::unordered_map<std::string,size_t> metric_map;

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
    const char * filename = "tauprofile.db";
    /* check if file exists */
    bool exists = file_exists(filename);
    /* open the database */
    rc = sqlite3_open(filename, &db);

    if( rc ) {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        return false;
    } else {
        fprintf(stderr, "Opened database successfully\n");
    }
    return (exists);
}

void create_database() {
    bool exists = open_database();
    if (exists) {
        fprintf(stdout, "Database exists\n");
    } else {
        rc = sqlite3_exec(db, database_schema, callback, 0, &zErrMsg);
   
        if( rc != SQLITE_OK ){
            fprintf(stderr, "SQL error: %s\n", zErrMsg);
            sqlite3_free(zErrMsg);
            return;
        } else {
            fprintf(stdout, "Table created successfully\n");
        }
    }
    return;
}

size_t store_trial() {
    std::stringstream sql;
    size_t trial_id = 0;
    sql << "insert into trial (name) values ('foo');";
    rc = sqlite3_exec(db, sql.str().c_str(), callback, 0, &zErrMsg);
 
    if( rc != SQLITE_OK ){
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        sqlite3_free(zErrMsg);
    } else {
        trial_id = sqlite3_last_insert_rowid(db);
        fprintf(stdout, "Trial %lu inserted successfully\n", trial_id);
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
            sqlite3_free(zErrMsg);
        } else {
            thread_id = sqlite3_last_insert_rowid(db);
            fprintf(stdout, "Thread %lu inserted successfully\n", thread_id);
        }
        thread_map[t] = thread_id;
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
        sqlite3_free(zErrMsg);
    } else {
        metric_id = sqlite3_last_insert_rowid(db);
        fprintf(stdout, "%s %lu inserted successfully\n", name, metric_id);
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

size_t store_timer(size_t trial_id, FunctionInfo *fi) {
    std::string longName(fi->GetName());
    std::string shortName(fi->GetName());
    shorten_timer_name(shortName);
    std::stringstream sql;
    sql << "insert into timer (trial, name, short_name) values ("
    << trial_id << ",'"
    << longName.c_str() << "','"
    << shortName.c_str() << "');";
    rc = sqlite3_exec(db, sql.str().c_str(), callback, 0, &zErrMsg);

    size_t timer_id = 0;
    if( rc != SQLITE_OK ){
        fprintf(stderr, "SQL error: %s\n", zErrMsg);
        sqlite3_free(zErrMsg);
    } else {
        timer_id = sqlite3_last_insert_rowid(db);
        fprintf(stdout, "%s %lu inserted successfully\n", shortName.c_str(), timer_id);
    }
    //metric_map[std::string(name)] = metric_id;
    return timer_id;
}

void store_timers(size_t trial_id) {
    // get the FunctionInfo database, and iterate over it
    std::vector<FunctionInfo*>::const_iterator it;
    RtsLayer::LockDB();
    /* Copy the function info database so we can release the lock */
    std::vector<FunctionInfo*> tmpTimers(TheFunctionDB());
    RtsLayer::UnLockDB();

    std::map<std::string, std::vector<double> >::iterator timer_map_it;

    //foreach: TIMER
    for (it = tmpTimers.begin(); it != tmpTimers.end(); it++) {
        FunctionInfo *fi = *it;
        store_timer(trial_id, fi);
    }
 }

size_t store_profile(size_t trial_id) {
    if (trial_id == 0UL) {
        trial_id = store_trial();
    }
    store_threads(trial_id);
    store_metrics(trial_id);
    store_timers(trial_id);
    return trial_id;
}

void close_database() {
    sqlite3_close(db);
}

void write_profile_to_database() {
    if (done) { return; }
    size_t trial_id = 0;
    if (comm_rank == 0) {
        trial_id = store_profile(0UL);
#ifdef TAU_MPI
        PMPI_Send(&trial_id, 1, MPI_UNSIGNED_LONG, 1, comm_rank+1, MPI_COMM_WORLD);
#endif
    } else {
#ifdef TAU_MPI
        MPI_Status status;
        PMPI_Recv(&trial_id, 1, MPI_UNSIGNED_LONG, comm_rank-1, 0, MPI_COMM_WORLD, &status);
#endif
        trial_id = store_profile(trial_id);
#ifdef TAU_MPI
        if (comm_rank+1 < comm_size) {
            PMPI_Send(&trial_id, 1, MPI_UNSIGNED_LONG, 1, comm_rank+1, MPI_COMM_WORLD);
        }
#endif
    }
    close_database();
    done = true;
}

int Tau_plugin_event_end_of_execution_null(Tau_plugin_event_end_of_execution_data_t *data) {
    printf("NULL PLUGIN %s\n", __func__);
    write_profile_to_database();
    return 0;
}

int Tau_plugin_event_pre_end_of_execution(Tau_plugin_event_pre_end_of_execution_data_t* data) {
    printf("NULL PLUGIN %s\n", __func__);
    write_profile_to_database();
    return 0;
}

int Tau_plugin_metadata_registration_complete_null(Tau_plugin_event_metadata_registration_data_t* data) {
    printf("NULL PLUGIN %s\n", __func__);
    return 0;
}

int Tau_plugin_event_post_init_null(Tau_plugin_event_post_init_data_t* data) {
    printf("NULL PLUGIN %s\n", __func__);
#ifdef TAU_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
#endif
    if (comm_rank == 0) {
        create_database();
    }
    return 0;
}  

/*This is the init function that gets invoked by the plugin mechanism inside TAU.
 * Every plugin MUST implement this function to register callbacks for various events 
 * that the plugin is interested in listening to*/
extern "C" int Tau_plugin_init_func(int argc, char **argv, int id) {
    Tau_plugin_callbacks * cb = (Tau_plugin_callbacks*)malloc(sizeof(Tau_plugin_callbacks));
    TAU_UTIL_INIT_TAU_PLUGIN_CALLBACKS(cb);

    done = false;

    /* Required event support */
    /*
    cb->Trigger = Tau_plugin_event_trigger_null;
    cb->Dump = Tau_plugin_event_dump_null;
    cb->MetadataRegistrationComplete = Tau_plugin_metadata_registration_complete_null;
    */
    cb->PostInit = Tau_plugin_event_post_init_null;
    cb->PreEndOfExecution = Tau_plugin_event_pre_end_of_execution;
    cb->EndOfExecution = Tau_plugin_event_end_of_execution_null;

    /* Trace events */
    /*
    cb->Send = Tau_plugin_event_send_null;
    cb->Recv = Tau_plugin_event_recv_null;
    cb->FunctionEntry = Tau_plugin_event_function_entry_null;
    cb->FunctionExit = Tau_plugin_event_function_exit_null;
    cb->AtomicEventTrigger = Tau_plugin_event_atomic_trigger_null;
    */

    TAU_UTIL_PLUGIN_REGISTER_CALLBACKS(cb, id);

    return 0;
}

