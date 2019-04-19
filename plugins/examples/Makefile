###############################################################################
## Top level Makefile for Plugin system #
###############################################################################

include ../../include/Makefile
include ../../utils/include/Makefile

PLUGININSTALLDIR = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/shared$(TAU_CONFIG)

MAKE = make

CC = $(TAU_CC)

CXX = $(TAU_CXX)

CFLAGS = $(TAU_MPI_COMPILE_INCLUDE) $(TAU_INCLUDE) $(TAU_DEFS) $(USER_OPT) $(TAU_INTERNAL_FLAGS) -fPIC -I. -g #-DTAU_ANALYTICS_INSTRUMENTATION_TOGGLE

LDFLAGS = $(TAU_MPI_LIB)

OBJS = libTAU-atomic-trigger.so libTAU-memory-usage-async.so libTAU-emulate-profiles-analysis-async.so libTAU-load-balance-async.so ibTAU-analyze-node-profiles.so libTAU-analyze-node-profiles-async.so libTAU-filter-plugin.so libTAU-mpit-recommend-sharp-usage-plugin.so libTAU-mpit-mvapich-free_unused_vbufs.so libTAU-selective-tracing.so libTAU-snapshot-trigger.so libTAU-mpit-exchange-data.so libTAU-analyze-profiles.so libTAU-load-balance.so tau_plugin_example_disable_instrumentation_runtime.o tau_plugin_example_mpit_recommend_sharp_usage.o tau_plugin_example_free_unused_vbufs.o tau_plugin_sos.o tau_plugin_thread_summary.o tau_plugin_example_selective_tracing.o tau_plugin_example_profile_snapshot_trigger.o tau_plugin_example_mpit_exchange_pvar_data.o tau_plugin_example_analyze_profile_snapshots.o tau_plugin_example_load_balance_async.o tau_plugin_example_load_balance.o tau_plugin_example_analyze_node_profile_snapshots.o tau_plugin_example_analyze_node_profile_snapshots_async.o tau_plugin_example_emulate_profile_snapshots_analysis_async.o tau_plugin_example_memory_usage_async.o tau_plugin_example_atomic_trigger.o

SHAREDLIBS = libTAU-memory-usage-async.so libTAU-emulate-profiles-analysis-async.so libTAU-load-balance-async.so libTAU-analyze-node-profiles.so libTAU-analyze-node-profiles-async.so libTAU-filter-plugin.so libTAU-mpit-recommend-sharp-usage-plugin.so libTAU-mpit-mvapich-free_unused_vbufs.so libTAU-sos-plugin.so libTAU-thread-summary-plugin.so libTAU-selective-tracing.so libTAU-snapshot-trigger.so libTAU-mpit-exchange-data.so libTAU-analyze-profiles.so libTAU-load-balance.so libTAU-memory-usage-async.so libTAU-adios2-plugin.so libTAU-adios2-trace-plugin.so libTAU-atomic-trigger.so libTAU-adios2-trace-plugin.so

all: install
	@echo

install: $(SHAREDLIBS)
	@mkdir -p $(PLUGININSTALLDIR)
	$(TAU_INSTALL) $^ $(PLUGININSTALLDIR)

libTAU-atomic-trigger.so: tau_plugin_example_atomic_trigger.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-atomic-trigger.so tau_plugin_example_atomic_trigger.o

libTAU-sos-plugin.so: tau_plugin_sos.o TauSOS.o
	$(CXX) tau_plugin_sos.o TauSOS.o $(TAU_SOS_LINKER_OPTS) $(TAU_SHFLAGS) libTAU-sos-plugin.so

libTAU-adios2-plugin.so: tau_plugin_adios2.o
	$(CXX) tau_plugin_adios2.o $(TAU_ADIOS2_LINKER_OPTS) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-adios2-plugin.so

libTAU-adios2-trace-plugin.so: tau_plugin_adios2_trace.o
	$(CXX) $(TAU_STDCXX11_FLAG) tau_plugin_adios2_trace.o $(TAU_ADIOS2_LINKER_OPTS) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-adios2-trace-plugin.so
libTAU-thread-summary-plugin.so: tau_plugin_thread_summary.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-thread-summary-plugin.so tau_plugin_thread_summary.o

libTAU-memory-usage-async.so: tau_plugin_example_memory_usage_async.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS)  libTAU-memory-usage-async.so tau_plugin_example_memory_usage_async.o

libTAU-emulate-profiles-analysis-async.so: tau_plugin_example_emulate_profile_snapshots_analysis_async.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-emulate-profiles-analysis-async.so tau_plugin_example_emulate_profile_snapshots_analysis_async.o

libTAU-analyze-node-profiles-async.so: tau_plugin_example_analyze_node_profile_snapshots_async.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-analyze-node-profiles-async.so tau_plugin_example_analyze_node_profile_snapshots_async.o

libTAU-analyze-node-profiles.so: tau_plugin_example_analyze_node_profile_snapshots.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-analyze-node-profiles.so tau_plugin_example_analyze_node_profile_snapshots.o

libTAU-filter-plugin.so: tau_plugin_example_disable_instrumentation_runtime.o 
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-filter-plugin.so tau_plugin_example_disable_instrumentation_runtime.o 

libTAU-selective-tracing.so: tau_plugin_example_selective_tracing.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-selective-tracing.so tau_plugin_example_selective_tracing.o

libTAU-snapshot-trigger.so: tau_plugin_example_profile_snapshot_trigger.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-snapshot-trigger.so tau_plugin_example_profile_snapshot_trigger.o

libTAU-mpit-exchange-data.so: tau_plugin_example_mpit_exchange_pvar_data.o 
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-mpit-exchange-data.so tau_plugin_example_mpit_exchange_pvar_data.o

libTAU-analyze-profiles.so: tau_plugin_example_analyze_profile_snapshots.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-analyze-profiles.so tau_plugin_example_analyze_profile_snapshots.o 

libTAU-load-balance-async.so: tau_plugin_example_load_balance_async.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-load-balance-async.so tau_plugin_example_load_balance_async.o

libTAU-load-balance.so: tau_plugin_example_load_balance.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-load-balance.so tau_plugin_example_load_balance.o

tau_plugin_example_atomic_trigger.o: Tau_plugin_example_atomic_trigger.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_atomic_trigger.cpp -o tau_plugin_example_atomic_trigger.o

tau_plugin_example_memory_usage_async.o: Tau_plugin_example_memory_usage_async.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_memory_usage_async.cpp -o tau_plugin_example_memory_usage_async.o

tau_plugin_example_emulate_profile_snapshots_analysis_async.o: Tau_plugin_example_emulate_profile_snapshots_analysis_async.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_emulate_profile_snapshots_analysis_async.cpp -o tau_plugin_example_emulate_profile_snapshots_analysis_async.o

tau_plugin_example_analyze_node_profile_snapshots_async.o: Tau_plugin_example_analyze_node_profile_snapshots_async.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_analyze_node_profile_snapshots_async.cpp -o tau_plugin_example_analyze_node_profile_snapshots_async.o

tau_plugin_example_analyze_node_profile_snapshots.o: Tau_plugin_example_analyze_node_profile_snapshots.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_analyze_node_profile_snapshots.cpp -o tau_plugin_example_analyze_node_profile_snapshots.o
 
tau_plugin_example_mpit_exchange_pvar_data.o: Tau_plugin_example_mpit_exchange_pvar_data.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_mpit_exchange_pvar_data.cpp -o tau_plugin_example_mpit_exchange_pvar_data.o

tau_plugin_example_profile_snapshot_trigger.o: Tau_plugin_example_profile_snapshot_trigger.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_profile_snapshot_trigger.cpp -o tau_plugin_example_profile_snapshot_trigger.o 

tau_plugin_example_selective_tracing.o: Tau_plugin_example_selective_tracing.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_selective_tracing.cpp -o tau_plugin_example_selective_tracing.o

tau_plugin_example_analyze_profile_snapshots.o: Tau_plugin_example_analyze_profile_snapshots.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_analyze_profile_snapshots.cpp -o tau_plugin_example_analyze_profile_snapshots.o

tau_plugin_example_load_balance_async.o: Tau_plugin_example_load_balance_async.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_load_balance_async.cpp -o tau_plugin_example_load_balance_async.o

tau_plugin_example_load_balance.o: Tau_plugin_example_load_balance.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_load_balance.cpp -o tau_plugin_example_load_balance.o
  
tau_plugin_sos.o: Tau_plugin_sos.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_sos.cpp -o tau_plugin_sos.o

tau_plugin_thread_summary.o: Tau_plugin_thread_summary.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_thread_summary.cpp -o tau_plugin_thread_summary.o

TauSOS.o: TauSOS.cpp
	$(CXX) $(CFLAGS) -c TauSOS.cpp -o TauSOS.o

tau_plugin_adios2.o: Tau_plugin_adios2.cpp
	$(CXX) $(CFLAGS) $(TAU_ADIOS2_INCLUDE_OPTS) -c Tau_plugin_adios2.cpp -o tau_plugin_adios2.o

tau_plugin_adios2_trace.o: Tau_plugin_adios2_trace.cpp
	$(CXX) $(TAU_STDCXX11_FLAG) $(CFLAGS) $(TAU_ADIOS2_INCLUDE_OPTS) -c Tau_plugin_adios2_trace.cpp -o tau_plugin_adios2_trace.o

tau_plugin_example_disable_instrumentation_runtime.o: Tau_plugin_example_disable_instrumentation_runtime.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_disable_instrumentation_runtime.cpp -o tau_plugin_example_disable_instrumentation_runtime.o

libTAU-mpit-recommend-sharp-usage-plugin.so : tau_plugin_example_mpit_recommend_sharp_usage.o
	 $(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-mpit-recommend-sharp-usage-plugin.so  tau_plugin_example_mpit_recommend_sharp_usage.o

tau_plugin_example_mpit_recommend_sharp_usage.o: Tau_plugin_example_mpit_recommend_sharp_usage.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_mpit_recommend_sharp_usage.cpp -o tau_plugin_example_mpit_recommend_sharp_usage.o

libTAU-mpit-mvapich-free_unused_vbufs.so: tau_plugin_example_free_unused_vbufs.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-mpit-mvapich-free_unused_vbufs.so tau_plugin_example_free_unused_vbufs.o

tau_plugin_example_free_unused_vbufs.o: Tau_plugin_example_mpit_mvapich_free_unused_vbufs.cpp
	$(CXX) $(CFLAGS) -c Tau_plugin_example_mpit_mvapich_free_unused_vbufs.cpp -o tau_plugin_example_free_unused_vbufs.o
clean:
	/bin/rm -rf *.o $(SHAREDLIBS)
