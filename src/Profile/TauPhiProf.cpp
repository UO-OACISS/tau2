#include <TAU.h>
#include <Profile/TauBfd.h>

#include <sstream>
#include <iostream>
#include <dlfcn.h> // link with -ldl -rdynamic a

#include <stdio.h>

#warning "Compiling PhiProf support"

#include <phiprof.hpp>

/* The phiprof interface makes an extensive use of ids. 
   Keep the correspondance between ids and labels here. */
std::map<std::string, int> timers_labels;
std::map<int, std::string> timers_ids;

int insertOrGet( const std::string &label ){
    int id;
    RtsLayer::LockEnv();
    auto it = timers_labels.find( label );
    if( it == timers_labels.end() ){
        id = timers_labels.size();
        timers_labels[ label ] = id;
        timers_ids[ id ] = label;
    } else {
        id =  it->second;
    }    
    RtsLayer::UnLockEnv();
    return id;
}
/* Simple initialization. Returns true if started succesfully */

bool phiprof::initialize(){
    Tau_create_top_level_timer_if_necessary(); // might not be necessary
    return true;
}

/* Initialize a timer, with a particular label   
 *
 * Initialize a timer. This enables one to define groups, and to use
 * the return id value for more efficient starts/stops in tight
 * loops. If this function is called for an existing timer it will
 * simply just return the id.
 */

int phiprof::initializeTimer(const std::string &label, const std::vector<std::string> &groups){
    return insertOrGet( label );
}

int phiprof::initializeTimer(const std::string &label){
    return insertOrGet( label );
}

int phiprof::initializeTimer(const std::string &label,const std::string &group1) {
    return insertOrGet( label );
}

int phiprof::initializeTimer(const std::string &label,const std::string &group1,const std::string &group2){
    return insertOrGet( label );
}

int phiprof::initializeTimer(const std::string &label,const std::string &group1,const std::string &group2,const std::string &group3){
    return insertOrGet( label );
}

/* Get id number of an existing timer that is a child of the currently
 * active one */

int phiprof::getChildId(const std::string &label){
    /* TODO */
    return -1;
}

bool phiprof::start(const std::string &label){
    TAU_START( label.c_str() ); // crashing here when MPI is enabled 
    return true;
}

bool phiprof::start(int id){
    auto label = timers_ids[ id ];
    TAU_START( label.c_str() );
    return true;
}

bool phiprof::stop (const std::string &label, double workUnits, const std::string &workUnitLabel){
    //TAU_STOP( label.c_str() );
    Tau_global_stop();
    return true;
}

bool phiprof::stop (int id, double workUnits, const std::string &workUnitLabel){
    auto label = timers_ids[ id ];
    TAU_STOP( label.c_str() );
    return true;
}

bool phiprof::stop (int id){
    auto label = timers_ids[ id ];
    TAU_STOP( label.c_str() );
    return true;
}

bool phiprof::print(MPI_Comm comm, std::string fileNamePrefix){
    return true;
}

