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

OBJS = Tau_plugin_example_disable_instrumentation_runtime.o Tau_plugin_example_mpit_recommend_sharp_usage.o Tau_plugin_example_free_unused_vbufs.o Tau_plugin_pooky2.o Tau_plugin_sos.o Tau_plugin_thread_summary.o Tau_plugin_example_selective_tracing.o Tau_plugin_example_profile_snapshot_trigger.o Tau_plugin_example_mpit_exchange_pvar_data.o Tau_plugin_example_analyze_profile_snapshots.o Tau_plugin_example_load_balance_async.o Tau_plugin_example_load_balance.o Tau_plugin_example_analyze_node_profile_snapshots.o Tau_plugin_example_analyze_node_profile_snapshots_async.o Tau_plugin_example_emulate_profile_snapshots_analysis_async.o Tau_plugin_example_memory_usage_async.o Tau_plugin_example_atomic_trigger.o

SHAREDLIBS = libTAU-memory-usage-async$(TAU_SHLIBX) libTAU-emulate-profiles-analysis-async$(TAU_SHLIBX) libTAU-load-balance-async$(TAU_SHLIBX) libTAU-analyze-node-profiles$(TAU_SHLIBX) libTAU-analyze-node-profiles-async$(TAU_SHLIBX) libTAU-filter-plugin$(TAU_SHLIBX) libTAU-mpit-recommend-sharp-usage-plugin$(TAU_SHLIBX) libTAU-mpit-mvapich-free_unused_vbufs$(TAU_SHLIBX) libTAU-pooky2-plugin$(TAU_SHLIBX) libTAU-sos-plugin$(TAU_SHLIBX) libTAU-thread-summary-plugin$(TAU_SHLIBX) libTAU-selective-tracing$(TAU_SHLIBX) libTAU-snapshot-trigger$(TAU_SHLIBX) libTAU-mpit-exchange-data$(TAU_SHLIBX) libTAU-analyze-profiles$(TAU_SHLIBX) libTAU-load-balance$(TAU_SHLIBX) libTAU-memory-usage-async$(TAU_SHLIBX) libTAU-adios2-plugin$(TAU_SHLIBX) libTAU-adios2-trace-plugin$(TAU_SHLIBX) libTAU-atomic-trigger$(TAU_SHLIBX) libTAU-adios2-trace-plugin$(TAU_SHLIBX) libTAU-null-plugin$(TAU_SHLIBX) libTAU-monitoring-plugin$(TAU_SHLIBX) $(TAU_SQLITE3_PLUGIN_LIB)

all: install
	@echo

install: $(SHAREDLIBS)
	@mkdir -p $(PLUGININSTALLDIR)
	$(TAU_INSTALL) $^ $(PLUGININSTALLDIR)

libTAU-atomic-trigger$(TAU_SHLIBX): Tau_plugin_example_atomic_trigger.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-atomic-trigger$(TAU_SHLIBX) Tau_plugin_example_atomic_trigger.o

libTAU-sos-plugin$(TAU_SHLIBX): Tau_plugin_sos.o TauSOS.o
	$(CXX) Tau_plugin_sos.o TauSOS.o $(TAU_SOS_LINKER_OPTS) $(TAU_SHFLAGS) libTAU-sos-plugin$(TAU_SHLIBX)

libTAU-pooky2-plugin$(TAU_SHLIBX): Tau_plugin_pooky2.o
	$(CXX) $(TAU_STDCXX11_FLAG) Tau_plugin_pooky2.o $(LDFLAGS) $(TAU_SHFLAGS) libTAU-pooky2-plugin$(TAU_SHLIBX)

libTAU-adios2-plugin$(TAU_SHLIBX): Tau_plugin_adios2.o
	$(CXX) Tau_plugin_adios2.o $(TAU_ADIOS2_LINKER_OPTS) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-adios2-plugin$(TAU_SHLIBX) $(TAU_SHLIBS)

libTAU-adios2-trace-plugin$(TAU_SHLIBX): Tau_plugin_adios2_trace.o
	$(CXX) $(TAU_STDCXX11_FLAG) Tau_plugin_adios2_trace.o $(TAU_ADIOS2_LINKER_OPTS) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-adios2-trace-plugin$(TAU_SHLIBX)

libTAU-null-plugin$(TAU_SHLIBX): Tau_plugin_example_null.o
	$(CXX) $(TAU_STDCXX11_FLAG) Tau_plugin_example_null.o $(LDFLAGS) $(TAU_SHFLAGS) libTAU-null-plugin$(TAU_SHLIBX)

libTAU-sqlite3-plugin$(TAU_SHLIBX): Tau_plugin_sqlite3.o
	$(CXX) $(TAU_STDCXX11_FLAG) Tau_plugin_sqlite3.o $(TAU_SQLITE3_LINKER_OPTS) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-sqlite3-plugin$(TAU_SHLIBX)

libTAU-monitoring-plugin$(TAU_SHLIBX): Tau_plugin_monitoring.o
	$(CXX) $(TAU_STDCXX11_FLAG) Tau_plugin_monitoring.o $(TAU_PAPI_EXTRA_FLAGS) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-monitoring-plugin$(TAU_SHLIBX) $(TAU_SHLIBS)

libTAU-thread-summary-plugin$(TAU_SHLIBX): Tau_plugin_thread_summary.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-thread-summary-plugin$(TAU_SHLIBX) Tau_plugin_thread_summary.o

libTAU-memory-usage-async$(TAU_SHLIBX): Tau_plugin_example_memory_usage_async.o
	$(CXX) $(TAU_STDCXX11_FLAG) $(LDFLAGS) $(TAU_SHFLAGS)  libTAU-memory-usage-async$(TAU_SHLIBX) Tau_plugin_example_memory_usage_async.o

libTAU-emulate-profiles-analysis-async$(TAU_SHLIBX): Tau_plugin_example_emulate_profile_snapshots_analysis_async.o
	$(CXX) $(TAU_STDCXX11_FLAG) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-emulate-profiles-analysis-async$(TAU_SHLIBX) Tau_plugin_example_emulate_profile_snapshots_analysis_async.o

libTAU-analyze-node-profiles-async$(TAU_SHLIBX): Tau_plugin_example_analyze_node_profile_snapshots_async.o
	$(CXX) $(TAU_STDCXX11_FLAG) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-analyze-node-profiles-async$(TAU_SHLIBX) Tau_plugin_example_analyze_node_profile_snapshots_async.o

libTAU-analyze-node-profiles$(TAU_SHLIBX): Tau_plugin_example_analyze_node_profile_snapshots.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-analyze-node-profiles$(TAU_SHLIBX) Tau_plugin_example_analyze_node_profile_snapshots.o

libTAU-filter-plugin$(TAU_SHLIBX): Tau_plugin_example_disable_instrumentation_runtime.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-filter-plugin$(TAU_SHLIBX) Tau_plugin_example_disable_instrumentation_runtime.o

libTAU-selective-tracing$(TAU_SHLIBX): Tau_plugin_example_selective_tracing.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-selective-tracing$(TAU_SHLIBX) Tau_plugin_example_selective_tracing.o

libTAU-snapshot-trigger$(TAU_SHLIBX): Tau_plugin_example_profile_snapshot_trigger.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-snapshot-trigger$(TAU_SHLIBX) Tau_plugin_example_profile_snapshot_trigger.o

libTAU-mpit-exchange-data$(TAU_SHLIBX): Tau_plugin_example_mpit_exchange_pvar_data.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-mpit-exchange-data$(TAU_SHLIBX) Tau_plugin_example_mpit_exchange_pvar_data.o

libTAU-analyze-profiles$(TAU_SHLIBX): Tau_plugin_example_analyze_profile_snapshots.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-analyze-profiles$(TAU_SHLIBX) Tau_plugin_example_analyze_profile_snapshots.o

libTAU-load-balance-async$(TAU_SHLIBX): Tau_plugin_example_load_balance_async.o
	$(CXX) $(TAU_STDCXX11_FLAG) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-load-balance-async$(TAU_SHLIBX) Tau_plugin_example_load_balance_async.o

libTAU-load-balance$(TAU_SHLIBX): Tau_plugin_example_load_balance.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-load-balance$(TAU_SHLIBX) Tau_plugin_example_load_balance.o

Tau_plugin_example_atomic_trigger.o: Tau_plugin_example_atomic_trigger.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

Tau_plugin_example_memory_usage_async.o: Tau_plugin_example_memory_usage_async.cpp
	$(CXX) $(TAU_STDCXX11_FLAG) $(CFLAGS) -c $< -o $@

Tau_plugin_example_emulate_profile_snapshots_analysis_async.o: Tau_plugin_example_emulate_profile_snapshots_analysis_async.cpp
	$(CXX) $(TAU_STDCXX11_FLAG) $(CFLAGS) -c $< -o $@

Tau_plugin_example_analyze_node_profile_snapshots_async.o: Tau_plugin_example_analyze_node_profile_snapshots_async.cpp
	$(CXX) $(TAU_STDCXX11_FLAG) $(CFLAGS) -c $< -o $@

Tau_plugin_example_analyze_node_profile_snapshots.o: Tau_plugin_example_analyze_node_profile_snapshots.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

Tau_plugin_example_mpit_exchange_pvar_data.o: Tau_plugin_example_mpit_exchange_pvar_data.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

Tau_plugin_example_profile_snapshot_trigger.o: Tau_plugin_example_profile_snapshot_trigger.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

Tau_plugin_example_selective_tracing.o: Tau_plugin_example_selective_tracing.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

Tau_plugin_example_analyze_profile_snapshots.o: Tau_plugin_example_analyze_profile_snapshots.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

Tau_plugin_example_load_balance_async.o: Tau_plugin_example_load_balance_async.cpp
	$(CXX) $(TAU_STDCXX11_FLAG) $(CFLAGS) -c $< -o $@

Tau_plugin_example_load_balance.o: Tau_plugin_example_load_balance.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

Tau_plugin_pooky2.o: Tau_plugin_pooky2.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

Tau_plugin_sos.o: Tau_plugin_sos.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

Tau_plugin_thread_summary.o: Tau_plugin_thread_summary.cpp
	$(CXX) $(CFLAGS) $(TAU_STDCXX11_FLAG) -c $< -o $@

TauSOS.o: TauSOS.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

Tau_plugin_adios2.o: Tau_plugin_adios2.cpp
	$(CXX) $(TAU_STDCXX11_FLAG) $(CFLAGS) $(TAU_ADIOS2_INCLUDE_OPTS) -c $< -o $@

Tau_plugin_adios2_trace.o: Tau_plugin_adios2_trace.cpp
	$(CXX) $(TAU_STDCXX11_FLAG) $(CFLAGS) $(TAU_ADIOS2_INCLUDE_OPTS) -c $< -o $@

Tau_plugin_example_null.o: Tau_plugin_example_null.cpp
	$(CXX) $(TAU_STDCXX11_FLAG) $(CFLAGS) -c $< -o $@

Tau_plugin_sqlite3.o: Tau_plugin_sqlite3.cpp Tau_plugin_sqlite3_schema.h
	$(CXX) $(TAU_STDCXX11_FLAG) $(CFLAGS) $(TAU_SQLITE3_INCLUDE_OPTS) -c $< -o $@

Tau_plugin_monitoring.o: Tau_plugin_monitoring.cpp
	$(CXX) $(TAU_STDCXX11_FLAG) $(CFLAGS) -c $< -o $@

Tau_sockets.o: Tau_sockets.cpp Tau_sockets.h
	$(CXX) $(TAU_STDCXX11_FLAG) $(CFLAGS) -c $< -o $@

Tau_plugin_example_disable_instrumentation_runtime.o: Tau_plugin_example_disable_instrumentation_runtime.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

libTAU-mpit-recommend-sharp-usage-plugin$(TAU_SHLIBX) : Tau_plugin_example_mpit_recommend_sharp_usage.o
	 $(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-mpit-recommend-sharp-usage-plugin$(TAU_SHLIBX)  Tau_plugin_example_mpit_recommend_sharp_usage.o

Tau_plugin_example_mpit_recommend_sharp_usage.o: Tau_plugin_example_mpit_recommend_sharp_usage.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

libTAU-mpit-mvapich-free_unused_vbufs$(TAU_SHLIBX): Tau_plugin_example_free_unused_vbufs.o
	$(CXX) $(LDFLAGS) $(TAU_SHFLAGS) libTAU-mpit-mvapich-free_unused_vbufs$(TAU_SHLIBX) Tau_plugin_example_free_unused_vbufs.o

Tau_plugin_example_free_unused_vbufs.o: Tau_plugin_example_mpit_mvapich_free_unused_vbufs.cpp
	$(CXX) $(CFLAGS) -c $< -o $@
clean:
	/bin/rm -rf *.o $(SHAREDLIBS)
