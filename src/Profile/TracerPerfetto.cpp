/****************************************************************************
**                      TAU Portable Profiling Package                     **
**                      http://www.cs.uoregon.edu/research/tau             **
*****************************************************************************
**    Copyright 2025                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**      File            : TracerPerfetto.cpp                                **
**      Description     : TAU Perfetto Trace Writer                         **
**      Contact         : tau-bugs@cs.uoregon.edu                           **
**      Documentation   : See http://www.cs.uoregon.edu/research/tau        **
****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <string>
#include <atomic>
#include <algorithm>
#include <functional>
#include <dirent.h>
#include <time.h>
#include <zlib.h>
#include <sched.h>

#ifdef TAU_MPI
#include <mpi.h>
extern "C" int Tau_get_usesMPI(void);
#endif

extern "C" int Tau_get_node_confirmed(void);

#include <tau_internal.h>
#include <Profile/Profiler.h>
#include <Profile/TauEnv.h>
#include <Profile/TauTrace.h>
#include <Profile/TauTracePerfetto.h>
#include <Profile/TauMetrics.h>
#include <Profile/UserEvent.h>
#include <Profile/TauMetaData.h>

#ifdef TAU_PERFETTO
#ifndef PERFETTO_ENABLE_LEGACY_TRACE_EVENTS
#define PERFETTO_ENABLE_LEGACY_TRACE_EVENTS 1
#endif
#include <perfetto.h>
#endif

extern "C" x_uint64 TauTraceGetTimeStamp(int tid);
extern "C" int Tau_is_thread_fake(int t);

using namespace tau;

#ifdef TAU_PERFETTO

// Default buffer size (KB) and flush period (ms) — overridden by
// TAU_PERFETTO_BUFFER_SIZE and TAU_PERFETTO_FLUSH_PERIOD env vars.

// Rank 0 is defined as node 0 or node -1 (for serial execution with MPI configured)
static inline bool is_rank0() {
  int n = RtsLayer::myNode();
  return (n == 0 || n == -1);
}

/* ------------------------------------------------------------------------- *
 * Perfetto categories
 * ------------------------------------------------------------------------- */
PERFETTO_DEFINE_CATEGORIES(
  perfetto::Category("tau"),
  perfetto::Category("tau_meta"),
  perfetto::Category("tau_counter"),
  perfetto::Category("tau_mpi")
);
PERFETTO_TRACK_EVENT_STATIC_STORAGE();

/* ------------------------------------------------------------------------- *
 * Platform thread id
 * ------------------------------------------------------------------------- */
#if defined(__linux__)
#include <sys/syscall.h>
static inline int64_t get_os_tid_now() { return (int64_t)syscall(SYS_gettid); }
#else
#include <thread>
static inline int64_t get_os_tid_now() {
  auto id = std::this_thread::get_id();
  int64_t out = 0;
  memcpy(&out, &id, std::min(sizeof(out), sizeof(id)));
  return out;
}
#endif

/* ------------------------------------------------------------------------- *
 * Temporary buffered event prior to init
 * ------------------------------------------------------------------------- */
struct temp_buffer_entry {
    long int ev;
    x_uint64 ts_us;
    x_int64 par;
    int kind;
    temp_buffer_entry(long int e, x_uint64 t, x_int64 p, int k)
      : ev(e), ts_us(t), par(p), kind(k) {}
};

/* ------------------------------------------------------------------------- *
 * Thread data
 * ------------------------------------------------------------------------- */
struct PerfettoThreadData {
    bool track_defined = false;
    bool buffers_written = false;
    bool thread_closed = false;
    std::vector<temp_buffer_entry>* temp_buffers = nullptr;
    int64_t os_tid = 0;
    uint64_t last_ts_ns = 0;
};

/* ------------------------------------------------------------------------- *
 * User Event (Counter) metadata
 * ------------------------------------------------------------------------- */
struct PerfettoCounterInfo {
    bool monotonic = false;
    bool defined = false;
    std::string name;
};

struct CachedFuncInfo {
    const char* name = "";
    const char* type = "";
    bool is_tau_internal = false;
};

/* ------------------------------------------------------------------------- *
 * Global Perfetto state
 * ------------------------------------------------------------------------- */
struct PerfettoGlobal {
    std::atomic<bool> initialized{false};
    std::atomic<bool> finished{false};
    std::atomic<bool> disabled{false};
	std::atomic<bool> initializing{false};
    std::atomic<bool> backend_initialized{false};

    int file_fd = -1;
    int init_rank = -2;   // Rank used when perfetto_do_init was called
    std::string per_rank_path;
    std::unique_ptr<perfetto::TracingSession> session;
    std::atomic<uint64_t> events_emitted{0}; // Counter for data-loss detection

    std::mutex global_state_mutex;
    std::vector<PerfettoThreadData*> thread_data;
    std::unordered_map<uint64_t, PerfettoCounterInfo> counter_map;
    std::unordered_map<long, CachedFuncInfo> function_cache;

    // Aggregated metadata
    std::vector<std::pair<std::string, std::string>> metadata_kv;
};

static PerfettoGlobal g_perfetto;
static std::atomic<uint64_t> g_fake_os_tid_counter{10000000};
// This flag prevents re-entrancy during initialization.
// If a thread is already in TauTracePerfettoInit, it should not
// try to record trace events that would lead back to initialization.
static std::atomic<bool> g_perfetto_initializing{false};

/* ---- Diagnostic helper: print Perfetto SDK TraceStats to stderr ---- */
static void dump_perfetto_trace_stats(int rank) {
    if (!g_perfetto.session) return;
    auto cb_args = g_perfetto.session->GetTraceStatsBlocking();
    if (cb_args.success) {
        perfetto::protos::gen::TraceStats stats;
        if (stats.ParseFromArray(cb_args.trace_stats_data.data(),
                                 cb_args.trace_stats_data.size())) {
            fprintf(stderr,
                "TAU_PERFETTO_DIAG[rank=%d]: SDK TraceStats:\n"
                "  producers_connected=%u producers_seen=%lu\n"
                "  data_sources_registered=%u data_sources_seen=%lu\n"
                "  tracing_sessions=%u total_buffers=%u\n"
                "  chunks_discarded=%lu patches_discarded=%lu\n"
                "  invalid_packets=%lu\n"
                "  flushes_requested=%lu flushes_succeeded=%lu flushes_failed=%lu\n"
                "  final_flush_outcome=%d\n",
                rank,
                stats.producers_connected(), (unsigned long)stats.producers_seen(),
                stats.data_sources_registered(), (unsigned long)stats.data_sources_seen(),
                stats.tracing_sessions(), stats.total_buffers(),
                (unsigned long)stats.chunks_discarded(), (unsigned long)stats.patches_discarded(),
                (unsigned long)stats.invalid_packets(),
                (unsigned long)stats.flushes_requested(), (unsigned long)stats.flushes_succeeded(),
                (unsigned long)stats.flushes_failed(),
                (int)stats.final_flush_outcome());
            for (int i = 0; i < stats.buffer_stats_size(); ++i) {
                const auto& bs = stats.buffer_stats()[i];
                fprintf(stderr,
                    "  buffer[%d]: size=%luKB bytes_written=%lu bytes_overwritten=%lu\n"
                    "    chunks_written=%lu chunks_overwritten=%lu chunks_discarded=%lu\n"
                    "    write_wrap_count=%lu patches_succeeded=%lu patches_failed=%lu\n"
                    "    abi_violations=%lu trace_writer_packet_loss=%lu\n",
                    i, (unsigned long)(bs.buffer_size()/1024),
                    (unsigned long)bs.bytes_written(), (unsigned long)bs.bytes_overwritten(),
                    (unsigned long)bs.chunks_written(), (unsigned long)bs.chunks_overwritten(),
                    (unsigned long)bs.chunks_discarded(),
                    (unsigned long)bs.write_wrap_count(),
                    (unsigned long)bs.patches_succeeded(), (unsigned long)bs.patches_failed(),
                    (unsigned long)bs.abi_violations(),
                    (unsigned long)bs.trace_writer_packet_loss());
            }
            fflush(stderr);
        }
    }
}

/* ------------------------------------------------------------------------- *
 * Helpers
 * ------------------------------------------------------------------------- */
// Returns the rank, treating -1 as 0 for serial execution with MPI configured
static inline uint64_t get_rank() {
    int r = RtsLayer::myNode();
    if (r < 0) r = 0;
    return (uint64_t)r;
}

static void ensure_thread_vector(int tid) {
    if ((int)g_perfetto.thread_data.size() <= tid) {
        std::lock_guard<std::mutex> lk(g_perfetto.global_state_mutex);
        while ((int)g_perfetto.thread_data.size() <= tid) {
            g_perfetto.thread_data.push_back(new PerfettoThreadData());
        }
    }
}

static inline uint64_t us_to_ns(x_uint64 us) { return (uint64_t)us * 1000ULL; }

static inline uint64_t current_ts_ns_for_thread(int tid) {
    return us_to_ns(TauTraceGetTimeStamp(tid));
}

static bool is_monotonic_user_event(uint64_t id) {
    TauInternalFunctionGuard guard;
    for (auto it = TheEventDB().begin(); it != TheEventDB().end(); ++it)
        if ((*it)->GetId() == id) return (*it)->IsMonotonicallyIncreasing();
    return false;
}

// Compute unique flow ID from MPI message parameters
static uint64_t compute_flow_id(int src, int dst, int tag, int comm) {
    uint64_t s = (uint64_t)(src & 0xFFFF);
    uint64_t d = (uint64_t)(dst & 0xFFFF);
    uint64_t t = (uint64_t)(tag & 0xFFFF);
    uint64_t c = (uint64_t)(comm & 0xFFFF);
    return (s << 48) | (d << 32) | (t << 16) | c;
}

static inline bool contains_str(const char* hay, const char* needle) {
    if (!hay || !needle) return false;
    return strstr(hay, needle) != nullptr;
}

static const CachedFuncInfo& get_cached_function_info(long func_id) {
	// Fast
    auto it = g_perfetto.function_cache.find(func_id);
    if (it != g_perfetto.function_cache.end()) {
        return it->second;
    }

    // Slow: The function is not in the cache. We must acquire a lock to write.
    std::lock_guard<std::mutex> lk(g_perfetto.global_state_mutex);

    // Double-check: Another thread might have added the entry while we waited for the lock.
    it = g_perfetto.function_cache.find(func_id);
    if (it != g_perfetto.function_cache.end()) {
        return it->second;
    }

    // This is the only thread that will compute and insert this func_id.
    TauInternalFunctionGuard guard;
    FunctionInfo* fi = nullptr;
    for (auto fit = TheFunctionDB().begin(); fit != TheFunctionDB().end(); ++fit) {
        if ((*fit)->GetFunctionId() == func_id) {
            fi = *fit;
            break;
        }
    }

    CachedFuncInfo new_info;
    if (fi) {
        new_info.name = fi->GetName() ? fi->GetName() : "";
        new_info.type = fi->GetType() ? fi->GetType() : "";
        // Intentionally filter internal TAU events to keep the trace clean.
        new_info.is_tau_internal = contains_str(new_info.name, "TauTraceClockOffset");
    } else {
        // Use a thread_local buffer for the fallback name to ensure thread safety.
        static thread_local char fallback_buf[256];
        snprintf(fallback_buf, sizeof(fallback_buf), "Function_%ld", func_id);
        new_info.name = fallback_buf;
    }

    // Insert the newly computed info into the cache and return a reference to it.
    auto result = g_perfetto.function_cache.emplace(func_id, new_info);
    return result.first->second;
}

/* ------------------------------------------------------------------------- *
 * Initialization
 * ------------------------------------------------------------------------- */
static void emit_process_descriptor(int rank) {
    auto track = perfetto::ProcessTrack::Current();
    auto desc = track.Serialize();
    std::string pname = std::string("Rank ") + std::to_string(rank);
    desc.set_name(pname);
    desc.mutable_process()->set_process_name(pname);
    desc.mutable_process()->set_pid(getpid());
    perfetto::TrackEvent::SetTrackDescriptor(track, desc);
    if (is_rank0()) {
        TAU_VERBOSE("TAU: Perfetto: process descriptor set: name='%s', rank=%d, pid=%d\n",
                    pname.c_str(), rank, (int)getpid());
    }
}

// Generate thread name with GPU context detection for CUDA, OpenCL, ROCm
static std::string get_thread_name(int rank, int tid) {
    TauInternalFunctionGuard guard;
    if (tid == 0) {
        return "main thread";
    }
    
    // Check if this is a fake thread (GPU context)
    if (Tau_is_thread_fake(tid)) {
        // CUDA thread detection
        const char* cuda_dev = Tau_metadata_get("CUDA Device", tid);
        if (cuda_dev && strcmp(cuda_dev, "") != 0) {
            const char* cuda_ctx = Tau_metadata_get("CUDA Context", tid);
            const char* cuda_stream = Tau_metadata_get("CUDA Stream", tid);
            char buf[256];
            if (cuda_stream && strcmp(cuda_stream, "0") == 0) {
                snprintf(buf, sizeof(buf), "CUDA [%s:%s:0]", cuda_dev, cuda_ctx);
            } else {
                snprintf(buf, sizeof(buf), "CUDA [%s:%s:%s]", cuda_dev, cuda_ctx, 
                         cuda_stream ? cuda_stream : "?");
            }
            return std::string(buf);
        }
        
        // OpenCL thread detection
        const char* opencl_dev = Tau_metadata_get("OpenCL Device", tid);
        if (opencl_dev && strcmp(opencl_dev, "") != 0) {
            const char* opencl_queue = Tau_metadata_get("OpenCL Command Queue", tid);
            char buf[256];
            snprintf(buf, sizeof(buf), "GPU dev%s:que%s", opencl_dev, 
                     opencl_queue ? opencl_queue : "?");
            return std::string(buf);
        }
        
        // ROCm thread detection
        const char* rocm_gpu = Tau_metadata_get("ROCM_GPU_ID", tid);
        if (rocm_gpu && strcmp(rocm_gpu, "") != 0) {
            const char* rocm_queue = Tau_metadata_get("ROCM_QUEUE_ID", tid);
            char buf[256];
            snprintf(buf, sizeof(buf), "GPU%s Queue%s", rocm_gpu, 
                     rocm_queue ? rocm_queue : "?");
            return std::string(buf);
        }
        
        // Generic GPU thread fallback
        return std::string("GPU thread ") + std::to_string(tid);
    }
    
    // Regular CPU thread
    return std::string("Rank ") + std::to_string(rank) + 
           ", CPU Thread " + std::to_string(tid);
}

static void emit_thread_descriptor(PerfettoThreadData* td, int tid) {
    auto track = perfetto::ThreadTrack::ForThread((uint64_t)td->os_tid);
    auto desc = track.Serialize();
    std::string tname = get_thread_name((int)get_rank(), tid);
    desc.mutable_thread()->set_thread_name(tname.c_str());
    perfetto::TrackEvent::SetTrackDescriptor(track, desc);
    if (is_rank0()) {
        TAU_VERBOSE("TAU: Perfetto: thread descriptor set: name='%s', tau_tid=%d, os_tid=%" PRId64 "\n",
                    tname.c_str(), tid, td->os_tid);
    }
}

static void perfetto_do_init() {
    if (g_perfetto.initialized) return;

    // Initialize the Perfetto backend only once; safe across reinit cycles.
    if (!g_perfetto.backend_initialized.load()) {
        perfetto::TracingInitArgs args;
        args.backends = perfetto::kInProcessBackend;
        perfetto::Tracing::Initialize(args);
        perfetto::TrackEvent::Register();
        g_perfetto.backend_initialized.store(true);
    }

    // Per-rank file: ${TRACEDIR}/tau.rank_<rank>.perfetto
    char path_buf[1024];
    snprintf(path_buf, sizeof(path_buf), "%s/tau.rank_%d.perfetto",
             TauEnv_get_tracedir() ? TauEnv_get_tracedir() : ".",
             (int)get_rank());
    g_perfetto.per_rank_path = path_buf;

    g_perfetto.file_fd = open(path_buf, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (g_perfetto.file_fd < 0) {
        
        TAU_VERBOSE("TAU: Perfetto: Failed to open %s\n", path_buf);
        
        g_perfetto.disabled = true;
        g_perfetto.initialized = true;
        g_perfetto.finished = true;
        return;
    }

    uint32_t buffer_kb = (uint32_t)TauEnv_get_perfetto_buffer_size();
    uint32_t flush_ms = (uint32_t)TauEnv_get_perfetto_flush_period();

    perfetto::TraceConfig cfg;
    auto* buf = cfg.add_buffers();
    buf->set_size_kb(buffer_kb);
    cfg.set_flush_period_ms(flush_ms);
    // Note: Setup(cfg, fd) automatically enables write_into_file when
    // an fd is passed (see perfetto SDK TracingSessionImpl::Setup).
    // We set file_write_period_ms to control how often data is drained
    // from the in-memory ring buffer to the file.  The SDK default is
    // 5000ms which is too slow for high event rates (e.g. 50K MatMult
    // iterations generating millions of MPI trace events).  Use the
    // same period as flush_ms (default 100ms) so the ring buffer never
    // accumulates more than ~100ms of data.
    cfg.set_file_write_period_ms(flush_ms);
    g_perfetto.events_emitted.store(0);

    // Allow the SDK to emit clock snapshots.  The trace processor
    // needs these to establish clock domain mappings.  Each per-rank
    // file gets its own rewritten sequence IDs during merge, so clock
    // snapshots from different ranks don't conflict.
    // cfg.mutable_builtin_data_sources()->set_disable_clock_snapshotting(true);

    auto* ds = cfg.add_data_sources();
    ds->mutable_config()->set_name("track_event");

    {
        protozero::HeapBuffered<perfetto::protos::pbzero::TrackEventConfig> te_cfg;
        auto* m = te_cfg.get();
        m->add_enabled_categories("tau");
        m->add_enabled_categories("tau_meta");
        m->add_enabled_categories("tau_counter");
        m->add_enabled_categories("tau_mpi");
        // Use explicit timestamps rather than incremental encoding
        m->set_disable_incremental_timestamps(true);
        ds->mutable_config()->set_track_event_config_raw(te_cfg.SerializeAsString());
    }

    g_perfetto.session = perfetto::Tracing::NewTrace();
    g_perfetto.session->Setup(cfg, g_perfetto.file_fd);
    g_perfetto.session->StartBlocking();

    // Emit process descriptor immediately so UI always has a name
    int rank = (int)get_rank();
    emit_process_descriptor(rank);
    g_perfetto.init_rank = rank;

    // Always print init diagnostic to stderr for every rank
    fprintf(stderr,
        "TAU_PERFETTO_DIAG[rank=%d]: initialized: pid=%d file=%s buffer=%uKB flush=%ums file_write_period=%ums\n",
        rank, (int)getpid(), path_buf, buffer_kb, flush_ms, flush_ms);
    fflush(stderr);

    if (is_rank0()) {
        TAU_VERBOSE("TAU: Perfetto started (rank=%d, buffer=%uKB, flush=%ums, file=%s, pid=%d)\n",
                    rank, buffer_kb, flush_ms, path_buf, (int)getpid());
    }

    g_perfetto.initialized = true;
    g_perfetto.finished = false;
    g_perfetto.disabled = false;
}

static void perfetto_finalize_locked() {
    if (!g_perfetto.initialized || g_perfetto.finished) return;

    int rank = (int)get_rank();
    int debug = TauEnv_get_perfetto_debug();

    // Always dump SDK stats before stopping the session (when debug is on, 
    // or unconditionally print a summary line)
    if (debug) {
        dump_perfetto_trace_stats(rank);
    }

    if (g_perfetto.session) {
        for (int i = 0; i < 2; ++i) {
            g_perfetto.session->FlushBlocking();
            usleep(20000);
        }
        g_perfetto.session->StopBlocking();
        g_perfetto.session.reset();
    }
    if (g_perfetto.file_fd >= 0) {
        fsync(g_perfetto.file_fd);
        close(g_perfetto.file_fd);
        g_perfetto.file_fd = -1;
    }
	
	char done_path[1024];
    snprintf(done_path, sizeof(done_path), "%s.done", g_perfetto.per_rank_path.c_str());
    int fd = open(done_path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd >= 0) {
        write(fd, "done\n", 5);
        close(fd);
    }
	
    // Check for possible data loss: if the file is unexpectedly small
    // relative to the number of events emitted, warn the user.
    uint64_t total_events = g_perfetto.events_emitted.load();
    struct stat st;
    off_t file_bytes = 0;
    if (!g_perfetto.per_rank_path.empty() &&
        stat(g_perfetto.per_rank_path.c_str(), &st) == 0) {
        file_bytes = st.st_size;
    }
    // Heuristic: each trace event typically produces >=10 bytes on disk.
    // If we emitted events but the file is nearly empty, data was lost.
    if (total_events > 100 && file_bytes < (off_t)(total_events * 2)) {
        fprintf(stderr, "TAU: Perfetto WARNING (rank %d): possible data loss detected.\n"
                        "     Events emitted: %lu, output file size: %ld bytes.\n"
                        "     Try increasing TAU_PERFETTO_BUFFER_SIZE (current: %dKB)\n"
                        "     or decreasing TAU_PERFETTO_FLUSH_PERIOD (current: %dms).\n",
                rank, (unsigned long)total_events, (long)file_bytes,
                TauEnv_get_perfetto_buffer_size(), TauEnv_get_perfetto_flush_period());
    }

    // Always print a per-rank summary to stderr for diagnostics
    fprintf(stderr,
        "TAU_PERFETTO_DIAG[rank=%d]: finalize: pid=%d init_rank=%d events=%lu "
        "file_bytes=%ld file=%s buffer=%dKB flush=%dms\n",
        rank, (int)getpid(), g_perfetto.init_rank,
        (unsigned long)total_events, (long)file_bytes,
        g_perfetto.per_rank_path.c_str(),
        TauEnv_get_perfetto_buffer_size(), TauEnv_get_perfetto_flush_period());
    fflush(stderr);

    g_perfetto.finished = true;
    TAU_VERBOSE("TAU: Perfetto finalized (rank=%d, events=%lu, file_bytes=%ld)\n",
                rank, (unsigned long)total_events, (long)file_bytes);
}

/* ------------------------------------------------------------------------- *
 * Thread metadata
 * ------------------------------------------------------------------------- */
static void ensure_thread_metadata(int rank, int tid) {
    TauInternalFunctionGuard guard;
    (void)rank;
    ensure_thread_vector(tid);
    PerfettoThreadData* td = g_perfetto.thread_data[tid];
    if (td->track_defined) return;

    if (Tau_is_thread_fake(tid)) {
        // This is a GPU stream or other logical thread. It doesn't have a real
        // OS TID. Assign a unique ID from our counter to guarantee a unique
        // Perfetto track.
        td->os_tid = g_fake_os_tid_counter++;
    } else {
        // This is a real CPU thread. Use the actual OS thread ID.
        td->os_tid = get_os_tid_now();
    }

    emit_thread_descriptor(td, tid);
    td->track_defined = true;
}

/* ------------------------------------------------------------------------- *
 * Replay buffered events
 * ------------------------------------------------------------------------- */
static void write_temp_buffer(int tid, int node_id) {
    ensure_thread_vector(tid);
    PerfettoThreadData* td = g_perfetto.thread_data[tid];
    td->buffers_written = true;
    if (!td->temp_buffers) return;

    if (is_rank0()) {
        TAU_VERBOSE("TAU: Perfetto: replaying %zu buffered events on tid=%d (rank=%d)\n",
                    td->temp_buffers->size(), tid, node_id);
    }
    for (const auto& e : *td->temp_buffers) {
        TauTracePerfettoEventWithNodeId(e.ev, e.par, tid, e.ts_us, 1, node_id, e.kind);
    }
    // Keep temp_buffers alive so they can be replayed again after a rank
    // reinitialize (e.g., mpirun_rsh). Cleanup happens in TauTracePerfettoClose.
}

/* ------------------------------------------------------------------------- *
 * Aggregated metadata emission on main thread at end-of-run
 * ------------------------------------------------------------------------- */
static void emit_aggregated_metadata_on_main_thread() {
    std::vector<std::pair<std::string, std::string>> local_kv;
    {
        std::lock_guard<std::mutex> lk(g_perfetto.global_state_mutex);
        if (g_perfetto.metadata_kv.empty()) return;
        local_kv.swap(g_perfetto.metadata_kv);
    }
    // Build a single string: key=value per line
    std::string joined;
    size_t total = 0;
    for (const auto& kv : local_kv) {
        total += kv.first.size() + kv.second.size() + 2;
    }
    joined.reserve(total + local_kv.size());
    for (size_t i = 0; i < local_kv.size(); ++i) {
        joined.append(local_kv[i].first);
        joined.push_back('=');
        joined.append(local_kv[i].second);
        if (i + 1 < local_kv.size()) joined.push_back('\n');
    }

    // Ensure main thread track exists
    int tid = 0;
    ensure_thread_metadata((int)get_rank(), tid);
    PerfettoThreadData* td = g_perfetto.thread_data[tid];
    auto track = perfetto::ThreadTrack::ForThread((uint64_t)td->os_tid);

    // Emit at end-of-run (current timestamp) for easy discovery at timeline end
    uint64_t ts_ns = current_ts_ns_for_thread(tid);
    TRACE_EVENT_INSTANT("tau_meta", "TAU Metadata", track, ts_ns,
                        "data", joined.c_str());

    if (is_rank0()) {
        TAU_VERBOSE("TAU: Perfetto: emitted metadata event 'TAU Metadata' at ts_ns=%" PRIu64 " (rank=%d, pairs=%zu)\n",
                    ts_ns, (int)get_rank(), local_kv.size());
    }
}

/* ------------------------------------------------------------------------- *
 * Public init
 * ------------------------------------------------------------------------- */
int TauTracePerfettoInit(int tid) {
    TauInternalFunctionGuard guard;
    return TauTracePerfettoInitTS(tid, TauTraceGetTimeStamp(tid));
}

int TauTracePerfettoInitTS(int tid, x_uint64 /*ts*/) {
    TauInternalFunctionGuard guard;
    if (g_perfetto.initialized.load() || g_perfetto.finished.load()) {
        return 0;
    }

#if defined(TAU_MPI)
    // Do not start Perfetto before the rank is reliably known.
    // Tau_get_node_confirmed() is set by setMyNode(), which is called
    // from the MPI_Init wrapper after PMPI_Comm_rank returns the real
    // rank.  Env-var detection (e.g. PMI_RANK) can give a wrong rank
    // (all ranks appear as 0 with mpirun_rsh), so we do NOT rely on
    // Tau_get_usesMPI() here — that flag may be set too early.
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if ((!mpi_initialized || !Tau_get_node_confirmed()) && TauEnv_get_set_node() <= -1) {
        if (TauEnv_get_perfetto_debug()) {
            static thread_local int defer_log_count = 0;
            if (defer_log_count++ < 3) {
                fprintf(stderr,
                    "TAU_PERFETTO_DIAG[pid=%d]: init deferred: mpi_initialized=%d node_confirmed=%d node=%d\n",
                    (int)getpid(), mpi_initialized, Tau_get_node_confirmed(), RtsLayer::myNode());
                fflush(stderr);
            }
        }
        return 1; // Defer init; caller will buffer the event.
    }
#endif

    // Use compare_exchange to safely elect one thread as the initializer.
    bool expected = false;
    if (g_perfetto.initializing.compare_exchange_strong(expected, true)) {
        // This thread is the initializer.
        perfetto_do_init(); // This call no longer happens under a lock.

        // After init is attempted, mark initialization as complete.
        // Other threads can now proceed.
        if (!g_perfetto.disabled.load()) {
            g_perfetto.initialized.store(true);
        }
        g_perfetto.initializing.store(false);

    } else {
        // Another thread is already initializing. Wait for it to finish.
        int spins = 0;
        while (g_perfetto.initializing.load()) {
            if ((++spins & 0xFF) == 0) sched_yield();
            if ((spins & 0xFFF) == 0) usleep(100); // Small backoff
        }
    }

    (void)tid;
    return g_perfetto.initialized.load() ? 0 : 1;
}

void TauTracePerfettoUnInitialize(int tid){(void)tid;}

/* Handle a rank change after Perfetto was already initialized.
 * This is called from TauTraceReinitialize (via setMyNode) when the
 * TAU node id changes, e.g. mpirun_rsh initially reports all ranks as 0
 * and later corrects them.  We close the current session/file (which
 * is tagged with the wrong rank) and reset state so that the
 * immediately-following TauTraceInit → perfetto_do_init() re-opens
 * with the correct rank.  Temp-buffers are preserved so the early
 * events can be replayed into the new session.
 */
void TauTracePerfettoReinitialize(int oldid, int newid, int tid) {
    (void)tid;
    if (!g_perfetto.initialized.load()) {
        // Log that reinitialize was called but skipped (init was deferred)
        fprintf(stderr,
            "TAU_PERFETTO_DIAG[pid=%d]: reinitialize skipped (not initialized): old=%d new=%d\n",
            (int)getpid(), oldid, newid);
        fflush(stderr);
        return;
    }

    TauInternalFunctionGuard guard;
    std::lock_guard<std::mutex> lk(g_perfetto.global_state_mutex);

    fprintf(stderr,
        "TAU_PERFETTO_DIAG[pid=%d]: reinitialize: rank %d -> %d (init_rank was %d)\n",
        (int)getpid(), oldid, newid, g_perfetto.init_rank);
    fflush(stderr);

    TAU_VERBOSE("TAU: Perfetto: Reinitializing: rank %d -> %d\n", oldid, newid);

    // 1. Stop the current tracing session (data in old file is discarded)
    if (g_perfetto.session) {
        g_perfetto.session->StopBlocking();
        g_perfetto.session.reset();
    }

    // 2. Close and remove the old per-rank file (tagged with wrong rank)
    if (g_perfetto.file_fd >= 0) {
        close(g_perfetto.file_fd);
        g_perfetto.file_fd = -1;
    }
    if (!g_perfetto.per_rank_path.empty()) {
        unlink(g_perfetto.per_rank_path.c_str());
        std::string done = g_perfetto.per_rank_path + ".done";
        unlink(done.c_str());
        g_perfetto.per_rank_path.clear();
    }

    // 3. Reset per-thread state so descriptors are re-emitted with the
    //    correct rank and buffered events are replayed into the new session.
    for (auto* td : g_perfetto.thread_data) {
        if (td) {
            td->track_defined   = false;
            td->buffers_written = false;
            td->thread_closed   = false;
        }
    }

    // 4. Mark as un-initialized; the next TauTraceInit (called right
    //    after setMyNode updates TheNode) triggers perfetto_do_init()
    //    with the correct rank.
    g_perfetto.init_rank = -2;
    g_perfetto.initialized.store(false);
    g_perfetto.finished.store(false);
    g_perfetto.disabled.store(false);
}

/* ------------------------------------------------------------------------- *
 * Kind classification
 * ------------------------------------------------------------------------- */
static inline bool is_func(int k){
    return (k==TAU_TRACE_EVENT_KIND_FUNC||
            k==TAU_TRACE_EVENT_KIND_CALLSITE||
            k==TAU_TRACE_EVENT_KIND_TEMP_FUNC);
}
static inline bool is_user(int k){
    return (k==TAU_TRACE_EVENT_KIND_USEREVENT||
            k==TAU_TRACE_EVENT_KIND_TEMP_USEREVENT);
}

/* ------------------------------------------------------------------------- *
 * Metadata capture (aggregate)
 * ------------------------------------------------------------------------- */
void TauTracePerfettoMetadata(const char* name, const char* value, int tid) {
    if (g_perfetto.finished || g_perfetto.disabled) return;

    TauInternalFunctionGuard guard;

    if (!g_perfetto.initialized) {
        TauTracePerfettoInit(tid);
        if (!g_perfetto.initialized) return;
    }

    std::lock_guard<std::mutex> lk(g_perfetto.global_state_mutex);
    g_perfetto.metadata_kv.emplace_back(
        std::string(name ? name : ""),
        std::string(value ? value : ""));
}

/* ------------------------------------------------------------------------- *
 * Function slice emission
 * ------------------------------------------------------------------------- */
static void emit_function(long func_id, bool is_enter, int rank, int tid,
                          uint64_t ts_ns) {
    ensure_thread_metadata(rank, tid);
    ensure_thread_vector(tid);
	
	const CachedFuncInfo& info = get_cached_function_info(func_id);
	if (info.is_tau_internal) {
        return;
    }
	PerfettoThreadData* td = g_perfetto.thread_data[tid];
	auto track = perfetto::ThreadTrack::ForThread((uint64_t)td->os_tid);
    if (info.type[0] != '\0') {
        if (is_enter) {
            TRACE_EVENT_BEGIN("tau", perfetto::DynamicString{info.name}, track, ts_ns, "type", info.type);
        } else {
            TRACE_EVENT_END("tau", track, ts_ns);
        }
    } else {
        if (is_enter) {
            TRACE_EVENT_BEGIN("tau", perfetto::DynamicString{info.name}, track, ts_ns);
        } else {
            TRACE_EVENT_END("tau", track, ts_ns);
        }
    }
}

/* ------------------------------------------------------------------------- *
 * User events (counters)
 * ------------------------------------------------------------------------- */
static void emit_user_event(uint64_t event_id, x_int64 raw_value,
                            int rank, int tid, uint64_t ts_ns) {
    ensure_thread_metadata(rank, tid);

    PerfettoCounterInfo ci;
    {
        // Thread-safe counter map access
        std::lock_guard<std::mutex> lk(g_perfetto.global_state_mutex);
        auto it = g_perfetto.counter_map.find(event_id);
        if (it == g_perfetto.counter_map.end()) {
            std::string ev_name("UserEvent");
            bool mono = is_monotonic_user_event(event_id);

            for (auto uit = TheEventDB().begin(); uit != TheEventDB().end(); ++uit) {
                if ((*uit)->GetId() == event_id) {
                    ev_name = (*uit)->GetName();
                    break;
                }
            }
            ci.monotonic = mono;
            ci.defined = true;
            ci.name = ev_name;
            g_perfetto.counter_map[event_id] = ci;
        } else {
            ci = it->second;
        }
    }

    // Emit monotonic counters as global tracks, others as instant events
    if (ci.monotonic) {
        TRACE_COUNTER("tau_counter", perfetto::DynamicString{ci.name.c_str()}, ts_ns, (int64_t)raw_value);
    } else {
        ensure_thread_vector(tid);
        PerfettoThreadData* td = g_perfetto.thread_data[tid];
        auto track = perfetto::ThreadTrack::ForThread((uint64_t)td->os_tid);
        TRACE_EVENT_INSTANT("tau", perfetto::DynamicString{ci.name.c_str()}, track, ts_ns,
                            "value", (int64_t)raw_value);
    }
}

/* ------------------------------------------------------------------------- *
 * MPI messages (flows)
 * ------------------------------------------------------------------------- */
static void emit_mpi_message(bool is_send, uint64_t flow_id,
                             int src_rank, int dst_rank,
                             int tag, int length,
                             int rank, int tid,
                             uint64_t ts_ns) {
    ensure_thread_metadata(rank, tid);
    ensure_thread_vector(tid);
    PerfettoThreadData* td = g_perfetto.thread_data[tid];
    auto track = perfetto::ThreadTrack::ForThread((uint64_t)td->os_tid);

    const char* name = is_send ? "MPI_Send" : "MPI_Recv";

    if (is_send) {
        TRACE_EVENT_INSTANT("tau_mpi", perfetto::DynamicString{name}, track, ts_ns,
                            perfetto::Flow::Global(flow_id),
                            "dst", dst_rank,
                            "tag", tag,
                            "bytes", length);
    } else {
        TRACE_EVENT_INSTANT("tau_mpi", perfetto::DynamicString{name}, track, ts_ns,
                            perfetto::TerminatingFlow::Global(flow_id),
                            "src", src_rank,
                            "tag", tag,
                            "bytes", length);
    }
}

static const char* get_kind_string(int kind) {
    switch(kind) {
        case TAU_TRACE_EVENT_KIND_FUNC: return "FUNC";
        case TAU_TRACE_EVENT_KIND_CALLSITE: return "CALLSITE";
        case TAU_TRACE_EVENT_KIND_TEMP_FUNC: return "TEMP_FUNC";
        case TAU_TRACE_EVENT_KIND_USEREVENT: return "USEREVENT";
        case TAU_TRACE_EVENT_KIND_TEMP_USEREVENT: return "TEMP_USEREVENT";
        default: return "UNKNOWN";
    }
}


/* ------------------------------------------------------------------------- *
 * Main event dispatcher
 * ------------------------------------------------------------------------- */
void TauTracePerfettoEventWithNodeId(long int ev, x_int64 par, int tid,
                                     x_uint64 ts_us, int use_ts, int node_id, int kind) {
    if (g_perfetto.finished.load() || g_perfetto.disabled.load()) return;

    TauInternalFunctionGuard guard;
	
    if (kind == TAU_TRACE_EVENT_KIND_TEMP_FUNC) kind = TAU_TRACE_EVENT_KIND_FUNC;
    else if (kind == TAU_TRACE_EVENT_KIND_TEMP_USEREVENT) kind = TAU_TRACE_EVENT_KIND_USEREVENT;

    if (RtsLayer::myNode() <= -1 || !g_perfetto.initialized.load()) {
        // If another thread is already initializing, buffer this event and
        // return immediately. This avoids blocking and preserves the event.
        if (RtsLayer::myNode() <= -1 || g_perfetto.initializing.load()) {
            ensure_thread_vector(tid);
            PerfettoThreadData* td = g_perfetto.thread_data[tid];
            if (!td->temp_buffers) td->temp_buffers = new std::vector<temp_buffer_entry>();
            x_uint64 t_us = use_ts ? ts_us : TauTraceGetTimeStamp(tid);
            td->temp_buffers->emplace_back(ev, t_us, par, kind);
            return;
        }

        // Otherwise, this thread is responsible for kicking off initialization.
        if (use_ts) {
            TauTracePerfettoInitTS(tid, ts_us);
        } else {
            TauTracePerfettoInit(tid);
        }

        // If initialization is still pending (e.g., MPI not yet initialized),
        // buffer the event instead of dropping it.
        if (!g_perfetto.initialized.load()) {
            ensure_thread_vector(tid);
            PerfettoThreadData* td = g_perfetto.thread_data[tid];
            if (!td->temp_buffers) td->temp_buffers = new std::vector<temp_buffer_entry>();
            x_uint64 t_us = use_ts ? ts_us : TauTraceGetTimeStamp(tid);
            td->temp_buffers->emplace_back(ev, t_us, par, kind);
            return;
        }
    }

    // ---- Buffer Replay Logic ----
    // This ensures buffered startup events from all threads are replayed.
    ensure_thread_vector(tid);
    if (!g_perfetto.thread_data[tid]->buffers_written) {
        write_temp_buffer(tid, node_id);
    }

    // ---- Normal Event Emission Logic ----
    int rank = node_id;
    uint64_t actual_ts_us = use_ts ? ts_us : TauTraceGetTimeStamp(tid);
    uint64_t ts_ns = us_to_ns(actual_ts_us);

    PerfettoThreadData* td = g_perfetto.thread_data[tid];
    if (ts_ns <= td->last_ts_ns) {
        ts_ns = td->last_ts_ns+1;
    }

    if (is_func(kind)) {
        bool is_enter = (par == 1);
        emit_function(ev, is_enter, rank, tid, ts_ns);
    } else if (is_user(kind)) {
        emit_user_event((uint64_t)ev, par, rank, tid, ts_ns);
    }

    g_perfetto.events_emitted.fetch_add(1, std::memory_order_relaxed);
    // Periodic progress logging (every 100K events per rank)
    {
        uint64_t count = g_perfetto.events_emitted.load(std::memory_order_relaxed);
        if (TauEnv_get_perfetto_debug() && (count % 100000) == 0 && count > 0) {
            fprintf(stderr, "TAU_PERFETTO_DIAG[rank=%d]: progress: %lu events emitted (pid=%d)\n",
                    rank, (unsigned long)count, (int)getpid());
            fflush(stderr);
        }
    }
    td->last_ts_ns = ts_ns;
}

void TauTracePerfettoEvent(long int ev, x_int64 par, int tid,
                           x_uint64 ts, int use_ts, int kind) {
    TauTracePerfettoEventWithNodeId(ev, par, tid, ts, use_ts, RtsLayer::myNode(), kind);
}

/* ------------------------------------------------------------------------- *
 * Direct MPI message
 * ------------------------------------------------------------------------- */
void TauTracePerfettoMsg(int send_or_recv, int type, int other_id, int length,
                         x_uint64 ts_us, int use_ts, int node_id) {
    if (g_perfetto.finished || g_perfetto.disabled) return;

    TauInternalFunctionGuard guard;

    if (!g_perfetto.initialized) {
        TauTracePerfettoInit(0);
        if (!g_perfetto.initialized) return;
    }

    int my_rank = (int)get_rank();
    int src_rank = (send_or_recv == TAU_MESSAGE_SEND) ? my_rank : other_id;
    int dst_rank = (send_or_recv == TAU_MESSAGE_SEND) ? other_id : my_rank;
    
    // Use node_id as communicator parameter for proper flow ID generation
    uint64_t flow_id = compute_flow_id(src_rank, dst_rank, type, node_id);
    int t = RtsLayer::myThread();

    uint64_t actual_ts_us = use_ts ? ts_us : TauTraceGetTimeStamp(t);
    uint64_t ts_ns = us_to_ns(actual_ts_us);

    emit_mpi_message(send_or_recv == TAU_MESSAGE_SEND, flow_id,
                     src_rank, dst_rank, type, length, my_rank, t, ts_ns);
}

/* ------------------------------------------------------------------------- *
 * Collectives
 * ------------------------------------------------------------------------- */
void TauTracePerfettoBarrierAllStart(int tag) {
    if (g_perfetto.finished || g_perfetto.disabled) return;
    TauInternalFunctionGuard guard; if (!g_perfetto.initialized) return;

    int tid = RtsLayer::myThread();
    ensure_thread_metadata((int)get_rank(), tid);
    uint64_t ts_ns = current_ts_ns_for_thread(tid);

    PerfettoThreadData* td = g_perfetto.thread_data[tid];
    auto track = perfetto::ThreadTrack::ForThread((uint64_t)td->os_tid);
    TRACE_EVENT_BEGIN("tau_mpi", "MPI_Barrier", track, ts_ns, "tag", tag);
}

void TauTracePerfettoBarrierAllEnd(int tag) {
    if (g_perfetto.finished || g_perfetto.disabled) return;
    TauInternalFunctionGuard guard; if (!g_perfetto.initialized) return;

    int tid = RtsLayer::myThread();
    uint64_t ts_ns = current_ts_ns_for_thread(tid);

    PerfettoThreadData* td = g_perfetto.thread_data[tid];
    auto track = perfetto::ThreadTrack::ForThread((uint64_t)td->os_tid);
    TRACE_EVENT_END("tau_mpi", track, ts_ns);
    (void)tag;
}

static const char* rma_collective_name(int type) {
    switch(type) {
        case TAU_TRACE_COLLECTIVE_TYPE_BARRIER:   return "RMA_Barrier";
        case TAU_TRACE_COLLECTIVE_TYPE_BROADCAST: return "RMA_Broadcast";
        case TAU_TRACE_COLLECTIVE_TYPE_ALLGATHER: return "RMA_Allgather";
        case TAU_TRACE_COLLECTIVE_TYPE_ALLREDUCE: return "RMA_Allreduce";
        default: return "RMA_Collective";
    }
}

void TauTracePerfettoRMACollectiveBegin(int tag, int type, int start, int stride,
                                        int size, int data_in, int data_out, int root) {
    if (g_perfetto.finished || g_perfetto.disabled) return;
    TauInternalFunctionGuard guard; if (!g_perfetto.initialized) return;
    int tid = RtsLayer::myThread();
    ensure_thread_metadata((int)get_rank(), tid);
    uint64_t ts_ns = current_ts_ns_for_thread(tid);
    const char* op = rma_collective_name(type);

    PerfettoThreadData* td = g_perfetto.thread_data[tid];
    auto track = perfetto::ThreadTrack::ForThread((uint64_t)td->os_tid);
    TRACE_EVENT_BEGIN("tau_mpi", perfetto::DynamicString{op}, track, ts_ns, "size", size);
    (void)tag;(void)start;(void)stride;(void)data_in;(void)data_out;(void)root;
}

void TauTracePerfettoRMACollectiveEnd(int tag, int type, int start, int stride,
                                      int size, int data_in, int data_out, int root) {
    if (g_perfetto.finished || g_perfetto.disabled) return;
    TauInternalFunctionGuard guard; if (!g_perfetto.initialized) return;
    int tid = RtsLayer::myThread();
    uint64_t ts_ns = current_ts_ns_for_thread(tid);

    PerfettoThreadData* td = g_perfetto.thread_data[tid];
    auto track = perfetto::ThreadTrack::ForThread((uint64_t)td->os_tid);
    TRACE_EVENT_END("tau_mpi", track, ts_ns);
    (void)tag;(void)type;(void)start;(void)stride;(void)size;(void)data_in;(void)data_out;(void)root;
}

/* ------------------------------------------------------------------------- *
 * Flush
 * ------------------------------------------------------------------------- */
void TauTracePerfettoFlushBuffer(int tid){
    TauInternalFunctionGuard guard;
    if (g_perfetto.session && g_perfetto.initialized && !g_perfetto.finished) {
        g_perfetto.session->FlushBlocking();
    }
    (void)tid;
}

/* ------------------------------------------------------------------------- *
 * Shutdown
 * ------------------------------------------------------------------------- */
void TauTracePerfettoShutdownComms(int tid){
    TauInternalFunctionGuard guard;
    if (!g_perfetto.initialized || g_perfetto.finished) return;

    int threadCount = (int)g_perfetto.thread_data.size();
    for (int t = 0; t < threadCount; t++) {
        if (g_perfetto.thread_data[t] && !g_perfetto.thread_data[t]->buffers_written) {
            write_temp_buffer(t, RtsLayer::myNode());
        }
    }
    TauTracePerfettoFlushBuffer(tid);
}


/* ------------------------------------------------------------------------- *
 * Perfetto merge
 * ------------------------------------------------------------------------- */

static inline bool file_exists(const std::string& name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

/* Scan the trace directory for tau.rank_*.perfetto files and return the
 * count.  Used as a fallback when tau_totalnodes() is unreliable (e.g.
 * MPI_Init wrapper was bypassed because MPI was already initialized). */
static int TauPerfettoCountRankFiles(const char* dir) {
    int count = 0;
    DIR* d = opendir(dir);
    if (!d) return 0;
    struct dirent* entry;
    while ((entry = readdir(d)) != nullptr) {
        const char* name = entry->d_name;
        // Match pattern: tau.rank_<N>.perfetto  (not .done)
        if (strncmp(name, "tau.rank_", 9) == 0) {
            const char* dot = strrchr(name, '.');
            if (dot && strcmp(dot, ".perfetto") == 0) {
                count++;
            }
        }
    }
    closedir(d);
    return count;
}

/* ---- Protobuf varint helpers for merge-time sequence ID rewriting ---- */

// Encode a uint64 as a protobuf varint into buf. Returns bytes written.
static int encode_varint(uint64_t value, uint8_t* buf) {
    int n = 0;
    while (value > 0x7F) {
        buf[n++] = (uint8_t)((value & 0x7F) | 0x80);
        value >>= 7;
    }
    buf[n++] = (uint8_t)value;
    return n;
}

// Decode a protobuf varint from data at *pos. Returns value, advances *pos.
static uint64_t decode_varint(const uint8_t* data, size_t len, size_t* pos) {
    uint64_t result = 0;
    int shift = 0;
    while (*pos < len) {
        uint8_t b = data[*pos];
        (*pos)++;
        result |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) return result;
        shift += 7;
        if (shift >= 64) break;
    }
    return result; // truncated
}

// Skip a protobuf field value based on wire type. Returns false if unparseable.
static bool skip_field(const uint8_t* data, size_t len, size_t* pos, int wire_type) {
    switch (wire_type) {
        case 0: // varint
            while (*pos < len && (data[*pos] & 0x80)) (*pos)++;
            if (*pos < len) (*pos)++;
            return true;
        case 1: // 64-bit
            *pos += 8;
            return *pos <= len;
        case 2: { // length-delimited
            uint64_t slen = decode_varint(data, len, pos);
            *pos += slen;
            return *pos <= len;
        }
        case 5: // 32-bit
            *pos += 4;
            return *pos <= len;
        default:
            return false;
    }
}

/* Rewrite field 1 (pid) inside a ProcessDescriptor or ThreadDescriptor
 * submessage to use the rank index instead of the OS PID.
 * When strip_process_name is true, also strips field 6 (process_name)
 * from ProcessDescriptor to avoid doubled labels in the Perfetto UI.
 * Returns a new submessage with the pid replaced. */
static std::vector<uint8_t> rewrite_pid_in_submessage(
        const uint8_t* data, size_t len, uint32_t new_pid,
        bool strip_process_name = false) {
    std::vector<uint8_t> out;
    out.reserve(len + 10);
    size_t pos = 0;
    while (pos < len) {
        size_t field_start = pos;
        uint64_t tag = decode_varint(data, len, &pos);
        uint32_t fn = (uint32_t)(tag >> 3);
        int wt = (int)(tag & 0x07);
        if (fn == 1 && wt == 0) {
            // Replace pid field
            decode_varint(data, len, &pos); // skip old value
            uint8_t tag_buf[10];
            int tag_len = encode_varint((1 << 3) | 0, tag_buf); // field 1 varint
            out.insert(out.end(), tag_buf, tag_buf + tag_len);
            uint8_t val_buf[10];
            int val_len = encode_varint(new_pid, val_buf);
            out.insert(out.end(), val_buf, val_buf + val_len);
        } else if (fn == 6 && strip_process_name) {
            // Strip process_name from ProcessDescriptor to prevent
            // doubled labels like "Rank 10 10" in the Perfetto UI.
            // The TrackDescriptor.name field provides the display label.
            size_t field_end = pos;
            if (!skip_field(data, len, &field_end, wt)) break;
            pos = field_end;
            // Don't emit this field
        } else {
            size_t field_end = pos;
            if (!skip_field(data, len, &field_end, wt)) break;
            out.insert(out.end(), data + field_start, data + field_end);
            pos = field_end;
        }
    }
    return out;
}

/* Rewrite a TrackDescriptor submessage for merge:
 *  - Rewrite PIDs in ProcessDescriptor (field 3) and ThreadDescriptor
 *    (field 4) to use rank_index+1 (avoiding PID 0 which is the Linux
 *    idle/swapper process and gets special treatment).
 *  - Strip process_name from ProcessDescriptor to prevent doubled labels
 *    like "Rank 10 10" (name + PID concatenated by the UI).
 *  - Rewrite name (field 2) to a zero-padded "Rank 00" format so the
 *    Perfetto UI shows ranks in correct numerical order. */
static std::vector<uint8_t> rewrite_track_descriptor(
        const uint8_t* data, size_t len,
        int rank_index, int num_ranks) {
    uint32_t new_pid = (uint32_t)(rank_index + 1);
    // Compute zero-pad width for rank names
    int pad_width = 1;
    { int m = num_ranks - 1; while (m >= 10) { m /= 10; pad_width++; } }
    char name_buf[64];
    snprintf(name_buf, sizeof(name_buf), "Rank %0*d", pad_width, rank_index);
    size_t name_slen = strlen(name_buf);

    std::vector<uint8_t> out;
    out.reserve(len + 40);
    bool has_process_desc = false;
    // Quick scan to see if this is a process descriptor
    { size_t s = 0;
      while (s < len) {
        uint64_t t = decode_varint(data, len, &s);
        uint32_t f = (uint32_t)(t >> 3); int w = (int)(t & 0x07);
        if (f == 3 && w == 2) { has_process_desc = true; break; }
        if (!skip_field(data, len, &s, w)) break;
      }
    }
    size_t pos = 0;
    while (pos < len) {
        size_t field_start = pos;
        uint64_t tag = decode_varint(data, len, &pos);
        uint32_t fn = (uint32_t)(tag >> 3);
        int wt = (int)(tag & 0x07);
        if (fn == 2 && wt == 2 && has_process_desc) {
            // Field 2: name — strip old on process tracks only,
            // will inject new zero-padded name at end
            uint64_t slen = decode_varint(data, len, &pos);
            pos += slen;
        } else if (fn == 3 && wt == 2) {
            // ProcessDescriptor — rewrite pid, strip process_name
            uint64_t sub_len = decode_varint(data, len, &pos);
            size_t sub_end = pos + sub_len;
            if (sub_end > len) break;
            std::vector<uint8_t> rewritten =
                rewrite_pid_in_submessage(data + pos, (size_t)sub_len,
                                          new_pid, /*strip_process_name=*/true);
            uint8_t tag_buf[10];
            int tag_len = encode_varint((3 << 3) | 2, tag_buf);
            out.insert(out.end(), tag_buf, tag_buf + tag_len);
            uint8_t len_buf[10];
            int len_n = encode_varint(rewritten.size(), len_buf);
            out.insert(out.end(), len_buf, len_buf + len_n);
            out.insert(out.end(), rewritten.begin(), rewritten.end());
            pos = sub_end;
        } else if (fn == 4 && wt == 2) {
            // ThreadDescriptor — rewrite pid only
            uint64_t sub_len = decode_varint(data, len, &pos);
            size_t sub_end = pos + sub_len;
            if (sub_end > len) break;
            std::vector<uint8_t> rewritten =
                rewrite_pid_in_submessage(data + pos, (size_t)sub_len,
                                          new_pid, /*strip_process_name=*/false);
            uint8_t tag_buf[10];
            int tag_len = encode_varint((4 << 3) | 2, tag_buf);
            out.insert(out.end(), tag_buf, tag_buf + tag_len);
            uint8_t len_buf[10];
            int len_n = encode_varint(rewritten.size(), len_buf);
            out.insert(out.end(), len_buf, len_buf + len_n);
            out.insert(out.end(), rewritten.begin(), rewritten.end());
            pos = sub_end;
        } else {
            size_t field_end = pos;
            if (!skip_field(data, len, &field_end, wt)) break;
            out.insert(out.end(), data + field_start, data + field_end);
            pos = field_end;
        }
    }
    // Inject new zero-padded name as field 2 (only for process tracks)
    if (has_process_desc) {
        uint8_t name_tag_buf[10];
        int name_tag_len = encode_varint((2 << 3) | 2, name_tag_buf);
        out.insert(out.end(), name_tag_buf, name_tag_buf + name_tag_len);
        uint8_t name_len_buf[10];
        int name_len_n = encode_varint(name_slen, name_len_buf);
        out.insert(out.end(), name_len_buf, name_len_buf + name_len_n);
        out.insert(out.end(), (uint8_t*)name_buf, (uint8_t*)name_buf + name_slen);
    }
    return out;
}

/* Rewrite a TracePacket during merge.  Performs these operations:
 *
 *   1. Rewrite trusted_packet_sequence_id (field 10): offset by
 *      (rank_index + 1) * SEQ_ID_STRIDE.  If no seq_id exists in
 *      the original packet, inject one.
 *
 *   2. Strip sequence_flags (field 13) from TrackDescriptor packets
 *      (field 60): our timestamp-sorted merge puts descriptors (ts=0)
 *      before the STATE_CLEARED packets on the same sequence.  Without
 *      stripping flags=SEQ_NEEDS_INCREMENTAL_STATE from descriptors,
 *      the trace processor would drop them because state hasn't been
 *      cleared yet at that point in the stream.
 *
 *   3. Strip ALL clock_snapshot packets (field 6).  TAU provides
 *      wall-clock nanosecond timestamps directly (CLOCK_BOOTTIME)
 *      and does not need Perfetto's clock-domain calibration.
 *      Clock snapshots from multiple ranks conflict and cause
 *      invalid_clock_snapshot errors in the trace processor.
 *
 *   4. Rewrite TrackDescriptor (field 60):
 *      - Set PID to rank_index+1 (avoiding PID 0 = Linux idle process)
 *      - Strip ProcessDescriptor.process_name (prevents doubled labels)
 *      - Rewrite name to zero-padded "Rank 00" format for proper ordering
 *
 * The packet_data/packet_len is the payload of the TracePacket
 * (inside the outer field-1 length-delimited wrapper).
 *
 * Returns a vector with the rewritten TracePacket payload wrapped
 * in the field-1 tag + length prefix, ready to write to the output.
 * Returns empty vector to signal the packet should be dropped.
 */
static const uint32_t SEQ_ID_STRIDE = 10000;

static std::vector<uint8_t> rewrite_packet(
        const uint8_t* packet_data, size_t packet_len,
        int rank_index, int num_ranks) {

    // ---- First pass: scan for key fields ----
    bool has_clock_snapshot = false;
    bool has_track_descriptor = false;

    size_t pos = 0;
    while (pos < packet_len) {
        uint64_t tag = decode_varint(packet_data, packet_len, &pos);
        uint32_t field_number = (uint32_t)(tag >> 3);
        int wire_type = (int)(tag & 0x07);

        if (field_number == 6 && wire_type == 2) {
            has_clock_snapshot = true;
        } else if (field_number == 60) {
            has_track_descriptor = true;
        }
        if (!skip_field(packet_data, packet_len, &pos, wire_type)) break;
    }

    // Drop ALL clock_snapshot packets.  TAU provides wall-clock
    // nanosecond timestamps directly (CLOCK_BOOTTIME) and doesn't
    // need Perfetto's clock-domain calibration.  Merging snapshots
    // from multiple ranks causes invalid_clock_snapshot errors.
    if (has_clock_snapshot) {
        return std::vector<uint8_t>();
    }

    // ---- Second pass: rebuild payload field-by-field ----
    std::vector<uint8_t> new_payload;
    new_payload.reserve(packet_len + 30);  // room for injected fields

    bool seq_written = false;

    pos = 0;
    while (pos < packet_len) {
        size_t field_start = pos;
        uint64_t tag = decode_varint(packet_data, packet_len, &pos);
        uint32_t field_number = (uint32_t)(tag >> 3);
        int wire_type = (int)(tag & 0x07);

        if (field_number == 10 && wire_type == 0) {
            // Field 10: trusted_packet_sequence_id — rewrite
            uint64_t old_seq = decode_varint(packet_data, packet_len, &pos);
            uint64_t new_seq = old_seq + (uint64_t)(rank_index + 1) * SEQ_ID_STRIDE;
            // Emit new field 10
            new_payload.push_back(0x50); // tag: field 10, wire type 0
            uint8_t vbuf[10];
            int vlen = encode_varint(new_seq, vbuf);
            new_payload.insert(new_payload.end(), vbuf, vbuf + vlen);
            seq_written = true;
        } else if (field_number == 13 && has_track_descriptor) {
            // Field 13: sequence_flags — strip from descriptor packets
            // so the trace processor doesn't require incremental state
            if (!skip_field(packet_data, packet_len, &pos, wire_type)) break;
            // Don't emit this field
        } else if (field_number == 60 && wire_type == 2) {
            // Field 60: TrackDescriptor — rewrite PIDs to use rank index
            uint64_t td_len = decode_varint(packet_data, packet_len, &pos);
            size_t td_end = pos + td_len;
            if (td_end > packet_len) break;
            std::vector<uint8_t> rewritten_td =
                rewrite_track_descriptor(
                    packet_data + pos, (size_t)td_len,
                    rank_index, num_ranks);
            // Emit field 60. Tag = 60<<3|2 = 482.
            uint8_t tag_buf[10];
            int tag_len = encode_varint((60 << 3) | 2, tag_buf);
            new_payload.insert(new_payload.end(), tag_buf, tag_buf + tag_len);
            uint8_t len_buf[10];
            int len_n = encode_varint(rewritten_td.size(), len_buf);
            new_payload.insert(new_payload.end(), len_buf, len_buf + len_n);
            new_payload.insert(new_payload.end(),
                               rewritten_td.begin(), rewritten_td.end());
            pos = td_end;
        } else {
            // Copy field as-is
            size_t field_end = pos;
            if (!skip_field(packet_data, packet_len, &field_end, wire_type)) break;
            new_payload.insert(new_payload.end(),
                               packet_data + field_start,
                               packet_data + field_end);
            pos = field_end;
        }
    }

    // Inject seq_id if not present in original packet
    if (!seq_written) {
        uint64_t injected_seq = (uint64_t)(rank_index + 1) * SEQ_ID_STRIDE;
        new_payload.push_back(0x50); // field 10 tag
        uint8_t vbuf[10];
        int vlen = encode_varint(injected_seq, vbuf);
        new_payload.insert(new_payload.end(), vbuf, vbuf + vlen);
    }

    // ---- Wrap in outer field-1 tag + length ----
    uint8_t outer_tag = (1 << 3) | 2; // 0x0A
    uint8_t outer_len_buf[10];
    int outer_len_n = encode_varint(new_payload.size(), outer_len_buf);

    std::vector<uint8_t> out;
    out.reserve(1 + outer_len_n + new_payload.size());
    out.push_back(outer_tag);
    out.insert(out.end(), outer_len_buf, outer_len_buf + outer_len_n);
    out.insert(out.end(), new_payload.begin(), new_payload.end());
    return out;
}

/* Extract the timestamp (field 8) from a TracePacket payload.
 * Returns 0 if no timestamp is found (descriptors, config packets). */
static uint64_t extract_packet_timestamp(const uint8_t* pkt_data, size_t pkt_len) {
    size_t pos = 0;
    while (pos < pkt_len) {
        uint64_t tag = decode_varint(pkt_data, pkt_len, &pos);
        uint32_t field_number = (uint32_t)(tag >> 3);
        int wire_type = (int)(tag & 0x07);
        if (field_number == 8 && wire_type == 0) {
            return decode_varint(pkt_data, pkt_len, &pos);
        }
        if (!skip_field(pkt_data, pkt_len, &pos, wire_type)) break;
    }
    return 0;
}

/* A parsed packet ready for sorted output. */
struct MergePacket {
    std::vector<uint8_t> data;  // Serialized outer wrapper (field 1 + payload)
    uint64_t timestamp;
    int rank;
    bool is_descriptor;  // Track/process descriptors go first
};

/* Read a rank file, parse all packets, rewrite them for merge
 * (seq IDs, sequence_flags on descriptors), and append to the
 * output vector. */
static bool parse_rank_file_for_merge(
        const char* rank_path, int rank_index, int num_ranks,
        std::vector<MergePacket>& out_packets) {

    int fd = open(rank_path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "TAU: Perfetto: WARNING: Could not open trace for rank %d at %s. Skipping.\n",
                rank_index, rank_path);
        return true;
    }
    struct stat st;
    fstat(fd, &st);
    size_t file_size = (size_t)st.st_size;
    if (file_size == 0) {
        close(fd);
        fprintf(stderr, "TAU: Perfetto: WARNING: Empty trace for rank %d at %s. Skipping.\n",
                rank_index, rank_path);
        return true;
    }

    uint8_t* file_data = (uint8_t*)malloc(file_size);
    if (!file_data) {
        close(fd);
        fprintf(stderr, "TAU: Perfetto: ERROR: Could not allocate %zu bytes for rank %d.\n",
                file_size, rank_index);
        return false;
    }

    size_t total_read = 0;
    while (total_read < file_size) {
        ssize_t n = read(fd, file_data + total_read, file_size - total_read);
        if (n <= 0) break;
        total_read += n;
    }
    close(fd);

    size_t pos = 0;
    int packets = 0;
    while (pos < total_read) {
        size_t pkt_start = pos;
        uint64_t tag = decode_varint(file_data, total_read, &pos);
        uint32_t field_number = (uint32_t)(tag >> 3);
        int wire_type = (int)(tag & 0x07);

        if (wire_type != 2) {
            if (!skip_field(file_data, total_read, &pos, wire_type)) break;
            // Non-length-delimited top-level data — include as-is
            MergePacket mp;
            mp.data.assign(file_data + pkt_start, file_data + pos);
            mp.timestamp = 0;
            mp.rank = rank_index;
            mp.is_descriptor = true;
            out_packets.push_back(std::move(mp));
            continue;
        }

        uint64_t pkt_len = decode_varint(file_data, total_read, &pos);
        if (pos + pkt_len > total_read) break;

        if (field_number == 1) {
            // TracePacket — rewrite with all merge-time transformations
            std::vector<uint8_t> rewritten = rewrite_packet(
                    file_data + pos, (size_t)pkt_len, rank_index, num_ranks);
            if (!rewritten.empty()) {
                uint64_t ts = extract_packet_timestamp(
                        file_data + pos, (size_t)pkt_len);
                // Detect descriptor packets (field 60 = track_descriptor,
                // field 36 = trace_config, field 33, field 45, etc.)
                bool is_desc = (ts == 0);
                if (!is_desc) {
                    size_t scan = 0;
                    while (scan < pkt_len) {
                        uint64_t t = decode_varint(file_data + pos, pkt_len, &scan);
                        uint32_t fn = (uint32_t)(t >> 3);
                        int wt = (int)(t & 0x07);
                        if (fn == 60 || fn == 36 || fn == 33 || fn == 45) {
                            is_desc = true;
                            break;
                        }
                        if (!skip_field(file_data + pos, pkt_len, &scan, wt)) break;
                    }
                }

                MergePacket mp;
                mp.data = std::move(rewritten);
                mp.timestamp = ts;
                mp.rank = rank_index;
                mp.is_descriptor = is_desc;
                out_packets.push_back(std::move(mp));
                packets++;
            }
        } else {
            // Non-packet top-level field — include as-is
            MergePacket mp;
            mp.data.assign(file_data + pkt_start, file_data + pos + pkt_len);
            mp.timestamp = 0;
            mp.rank = rank_index;
            mp.is_descriptor = true;
            out_packets.push_back(std::move(mp));
        }
        pos += pkt_len;
    }

    free(file_data);
    TAU_VERBOSE("TAU: Perfetto: rank %d: parsed %d packets for sorted merge\n",
                rank_index, packets);
    return true;
}

static void TauPerfettoMergeAllRanks(const char* dir, int num_ranks) {

    if (num_ranks <= 0) {
        printf("TAU: Perfetto merge skipped: number of ranks provided was %d.\n", num_ranks);
        return;
    }    

    bool do_gzip = TauEnv_get_perfetto_compress();
    std::string stem = std::string(dir) + "/tau.perfetto";
    std::string ext = do_gzip ? ".gz" : "";
    std::string out_path = stem + ext;

    int version = 1;
    while (file_exists(out_path)) {
        out_path = stem + "." + std::to_string(version++) + ext;
    }

    if (out_path != stem + ext) {
        printf("TAU: Perfetto: WARNING: Output file %s%s already exists. Saving to %s instead.\n",
               stem.c_str(), ext.c_str(), out_path.c_str());
    }

    printf("TAU: Perfetto: Merging traces for %d ranks into %s (timestamp-sorted interleaved merge)\n",
           num_ranks, out_path.c_str());
    fflush(stdout);

    // ---- Phase 1: Read and parse all per-rank files ----
    std::vector<MergePacket> all_packets;
    all_packets.reserve(2000000 * num_ranks);  // rough estimate
    bool success = true;
    int files_merged_count = 0;

    for (int i = 0; i < num_ranks; ++i) {
        char rank_path[1024];
        snprintf(rank_path, sizeof(rank_path), "%s/tau.rank_%d.perfetto", dir, i);
        struct stat st;
        if (stat(rank_path, &st) == 0) {
            TAU_VERBOSE("TAU: Perfetto: rank %d: %s (%ld bytes)\n", i, rank_path, (long)st.st_size);
        } else {
            fprintf(stderr, "TAU: Perfetto: rank %d: %s (MISSING)\n", i, rank_path);
            continue;
        }
        if (!parse_rank_file_for_merge(rank_path, i, num_ranks, all_packets)) {
            success = false;
            break;
        }
        files_merged_count++;
    }

    if (!success) {
        fprintf(stderr, "TAU: Perfetto merge failed during file parsing.\n");
        fflush(stdout);
        return;
    }

    printf("TAU: Perfetto: Parsed %zu total packets from %d ranks. Sorting by timestamp...\n",
           all_packets.size(), files_merged_count);
    fflush(stdout);

    // ---- Phase 2: Sort — descriptors first, then by timestamp ----
    std::sort(all_packets.begin(), all_packets.end(),
        [](const MergePacket& a, const MergePacket& b) {
            // Descriptors always come before events
            if (a.is_descriptor != b.is_descriptor)
                return a.is_descriptor;
            // Within descriptors, order by rank for consistency
            if (a.is_descriptor && b.is_descriptor)
                return a.rank < b.rank;
            // Within events, order by timestamp (stable within rank)
            if (a.timestamp != b.timestamp)
                return a.timestamp < b.timestamp;
            return a.rank < b.rank;
        });

    printf("TAU: Perfetto: Sort complete. Writing merged output...\n");
    fflush(stdout);

    // ---- Phase 3: Write sorted packets ----

    if (do_gzip) {
        gzFile out = gzopen(out_path.c_str(), "wb9");
        if (!out) {
            fprintf(stderr, "TAU: Perfetto: failed to open %s for gzip output\n", out_path.c_str());
            success = false;
        } else {
            for (size_t i = 0; i < all_packets.size() && success; ++i) {
                const auto& pkt = all_packets[i];
                size_t written = 0;
                while (written < pkt.data.size()) {
                    unsigned chunk = (unsigned)std::min(
                        pkt.data.size() - written, (size_t)(1 << 20));
                    int n = gzwrite(out, pkt.data.data() + written, chunk);
                    if (n <= 0) {
                        int errnum = 0;
                        const char* zerr_msg = gzerror(out, &errnum);
                        fprintf(stderr, "TAU: Perfetto: gzwrite error: %s\n", zerr_msg);
                        success = false;
                        break;
                    }
                    written += n;
                }
            }
            gzclose(out);
        }
    } else {
        int out_fd = open(out_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
        if (out_fd < 0) {
            fprintf(stderr, "TAU: Perfetto: failed to open %s for output\n", out_path.c_str());
            success = false;
        } else {
            for (size_t i = 0; i < all_packets.size() && success; ++i) {
                const auto& pkt = all_packets[i];
                size_t written = 0;
                while (written < pkt.data.size()) {
                    ssize_t n = write(out_fd, pkt.data.data() + written,
                                      pkt.data.size() - written);
                    if (n <= 0) {
                        perror("TAU: Perfetto: write error");
                        success = false;
                        break;
                    }
                    written += n;
                }
            }
            fsync(out_fd);
            close(out_fd);
        }
    }

	if (success) {
        printf("TAU: Perfetto merge complete (%d of %d expected files merged).\n", files_merged_count, num_ranks);
        if (TauEnv_get_perfetto_keep_files()) {
            printf("TAU: Perfetto: Keeping per-rank trace files (TAU_PERFETTO_KEEP_FILES=1).\n");
        } else {
            printf("TAU: Perfetto: Cleaning up per-rank trace files.\n");
            for (int i = 0; i < num_ranks; ++i) {
                char rank_path[1024];
                snprintf(rank_path, sizeof(rank_path), "%s/tau.rank_%d.perfetto", dir, i);
                unlink(rank_path);
	    		char done_path[1024];
                snprintf(done_path, sizeof(done_path), "%s/tau.rank_%d.perfetto.done", dir, i);
                unlink(done_path);
            }
        }
    } else {
        fprintf(stderr, "TAU: Perfetto merge failed.\n");
        fprintf(stderr, "TAU: Perfetto: Deleting incomplete output file %s.\n", out_path.c_str());
        unlink(out_path.c_str());
    }
    fflush(stdout);
}

static bool TauPerfettoWaitForSentinels(const char* dir, int total_ranks) {
    const int max_wait_sec = 120;
    const int check_interval_ms = 250;
	
	if(total_ranks <= 1) {
        return true;
    }

    printf("TAU: Perfetto: Rank 0 waiting for %d ranks to complete (timeout: %d seconds)...\n",
           total_ranks, max_wait_sec);
    fflush(stdout);

    auto start_time = time(nullptr);

    while (true) {
        std::vector<int> missing_ranks;
        for (int i = 0; i < total_ranks; ++i) {
            char sentinel_path[1024];
            snprintf(sentinel_path, sizeof(sentinel_path), "%s/tau.rank_%d.perfetto.done", dir, i);
            
            if (!file_exists(sentinel_path)) {
                missing_ranks.push_back(i);
            }
        }

        if (missing_ranks.empty()) {
            printf("TAU: Perfetto: All %d ranks complete.\n", total_ranks);
            fflush(stdout);
            return true;
        }

        if (difftime(time(nullptr), start_time) > max_wait_sec) {
            fprintf(stderr, "TAU: Perfetto: ERROR: Timed out after %d seconds waiting for sentinel files.\n", max_wait_sec);
            
            std::string missing_list_str;
            for(size_t i = 0; i < missing_ranks.size(); ++i) {
                if (i >= 5) { // Don't print a huge list
                    missing_list_str += "and " + std::to_string(missing_ranks.size() - i) + " more...";
                    break;
                }
                missing_list_str += std::to_string(missing_ranks[i]) + (i < missing_ranks.size() - 1 ? ", " : "");
            }
            fprintf(stderr, "TAU: Perfetto: Missing sentinel files from the following ranks: %s\n", missing_list_str.c_str());
            fprintf(stderr, "TAU: Perfetto: Aborting merge.\n");
            fflush(stderr);
            return false;
        }

        usleep(check_interval_ms * 1000);
    }
}

/* ------------------------------------------------------------------------- *
 * Close
 * ------------------------------------------------------------------------- */
void TauTracePerfettoClose(int tid){
    TauInternalFunctionGuard guard;
    if (!g_perfetto.initialized || g_perfetto.finished) return;

    ensure_thread_vector(tid);
    PerfettoThreadData* td = g_perfetto.thread_data[tid];

    int64_t current_os_tid = get_os_tid_now();
    if (td && td->os_tid != 0 && current_os_tid == td->os_tid && !td->thread_closed) {
        uint64_t ts_ns = us_to_ns(TauTraceGetTimeStamp(tid));
        auto track = perfetto::ThreadTrack::ForThread((uint64_t)td->os_tid);
        TRACE_EVENT_INSTANT("tau_meta", "ThreadExit", track, ts_ns,
                            "tau_tid", tid,
                            "os_tid", td->os_tid);
        td->thread_closed = true;
        TauTracePerfettoFlushBuffer(tid);
    }

    if (tid == 0) {
        // Emit aggregated metadata once on main thread
        emit_aggregated_metadata_on_main_thread();

        {
            std::lock_guard<std::mutex> lk(g_perfetto.global_state_mutex);
            // Clean up temp_buffers kept alive for potential rank-reinitialize.
            for (auto* td_iter : g_perfetto.thread_data) {
                if (td_iter && td_iter->temp_buffers) {
                    delete td_iter->temp_buffers;
                    td_iter->temp_buffers = nullptr;
                }
            }
            perfetto_finalize_locked();
        }

        // Rank-0: print user-facing info about per-rank files
        if (is_rank0()) {
			 if (TauEnv_get_perfetto_merge()) { 
				const char* outdir = TauEnv_get_tracedir();
				if (!outdir || !*outdir) outdir = ".";
				int nodes=tau_totalnodes(0,0);
				/* Fallback: if tau_totalnodes was never set (e.g. MPI_Init
				 * guard skipped the block because MPI was already initialized),
				 * scan the trace directory for per-rank files. */
				if (nodes <= 1) {
					int scanned = TauPerfettoCountRankFiles(outdir);
					if (scanned > nodes) {
						TAU_VERBOSE("TAU: Perfetto: tau_totalnodes=%d but found %d rank files; using scanned count.\n",
									nodes, scanned);
						nodes = scanned;
					}
				}
				TauPerfettoWaitForSentinels(outdir, nodes);
                TauPerfettoMergeAllRanks(outdir, nodes);
            } else {
            const char* outdir = TauEnv_get_tracedir();
            if (!outdir || !*outdir) outdir = ".";
            printf("TAU: Perfetto trace files written to %s\n", outdir);
            printf("TAU: To merge per-rank traces:\n");
            printf("TAU:   cat tau.rank_*.perfetto > tau.perfetto\n");
            printf("TAU: To compress the trace:\n");
            printf("TAU:   gzip -c tau.perfetto > tau.perfetto.gz\n");
            fflush(stdout);
			}
        }
    }
}

/* ------------------------------------------------------------------------- *
 * Stubs if TAU_PERFETTO disabled
 * ------------------------------------------------------------------------- */

#else /* !TAU_PERFETTO */
int  TauTracePerfettoInit(int){return 0;}
int  TauTracePerfettoInitTS(int,x_uint64){return 0;}
void TauTracePerfettoUnInitialize(int){}
void TauTracePerfettoReinitialize(int,int,int){}
void TauTracePerfettoEvent(long int,x_int64,int,x_uint64,int,int){}
void TauTracePerfettoEventWithNodeId(long int,x_int64,int,x_uint64,int,int,int){}
void TauTracePerfettoMsg(int,int,int,int,x_uint64,int,int){}
void TauTracePerfettoBarrierAllStart(int){}
void TauTracePerfettoBarrierAllEnd(int){}
void TauTracePerfettoRMACollectiveBegin(int,int,int,int,int,int,int,int){}
void TauTracePerfettoRMACollectiveEnd(int,int,int,int,int,int,int,int){}
void TauTracePerfettoFlushBuffer(int){}
void TauTracePerfettoShutdownComms(int){}
void TauTracePerfettoClose(int){}
void TauTracePerfettoMetadata(const char*, const char*, int){}
#endif /* TAU_PERFETTO */
