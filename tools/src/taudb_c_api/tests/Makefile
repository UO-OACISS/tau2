#****************************************************************************
#*                      TAU Performance System (R)                          *
#*                      http://tau.uoregon.edu                              *
#****************************************************************************
#*    Copyright 2012                                                       **
#*    Department of Computer and Information Science, University of Oregon **
#*    Juelich Supercomputing Center, NIC, Germany                          ** 
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************

.SUFFIXES : .c

include ../Makefile.in

BUILD_TARGETS=activeharmony \
save_trial \
parse_tau_profile_test \
dump_functions \
parse_config_test \
perfdmf_query_trial_test \
perfdmf_query_timers_test \
perfdmf_query_metrics_test \
perfdmf_query_threads_test \
perfdmf_query_timer_callpaths_test \
taudb_main \
taudb_query_trial_test \
taudb_query_timers_test \
taudb_query_counters_test \
taudb_query_metrics_test \
taudb_query_threads_test \
taudb_query_timer_values_test \
taudb_query_stats_test \
taudb_query_timer_callpaths_test \
taudb_query_timer_call_data_test \
perfdmf_test \
query_all

TEST_TARGETS=parse_tau_profile_test \
#save_trial \
activeharmony \
taudb_main\
parse_config_test \
taudb_query_trial_test \
taudb_query_timers_test \
taudb_query_metrics_test \
taudb_query_threads_test \
taudb_query_stats_test \
taudb_query_timer_callpaths_test \
taudb_query_timer_call_data_test \
taudb_query_timer_values_test \

#perfdmf_query_trial_test \
#perfdmf_query_metrics_test \
#perfdmf_query_threads_test \
#perfdmf_query_timers_test \
#perfdmf_query_timer_callpaths_test \

# rebuild everything if the makefile changes
all: Makefile ../lib/libtaudb.a
	@for i in dump_functions $(BUILD_TARGETS) ; do (cd $$i; $(MAKE) "MAKE=$(MAKE)" || exit 0); done

test: all
	@for i in dump_functions $(TEST_TARGETS) ; do (cd $$i; $(MAKE) "MAKE=$(MAKE)" test || exit 0); done

clean:
	@for i in dump_functions $(BUILD_TARGETS) ; do (cd $$i; $(MAKE) "MAKE=$(MAKE)" clean || exit 0); done

remake: clean all
