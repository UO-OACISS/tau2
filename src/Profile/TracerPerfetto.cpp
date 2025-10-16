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
#include <dirent.h>
#include <time.h>
#include <zlib.h>
#include <sched.h>

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

// Fixed default buffer size (KB)
static constexpr uint32_t kPerfettoBufferKB = 16384; // 16 MB

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

    int file_fd = -1;
    std::string per_rank_path;
    std::unique_ptr<perfetto::TracingSession> session;

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
    desc.mutable_process()->set_process_name(pname.c_str());
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

    perfetto::TracingInitArgs args;
    args.backends = perfetto::kInProcessBackend;
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();

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

    perfetto::TraceConfig cfg;
    auto* buf = cfg.add_buffers();
    buf->set_size_kb(kPerfettoBufferKB);
    cfg.set_flush_period_ms(1000);

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
    if (is_rank0()) {
        TAU_VERBOSE("TAU: Perfetto started (rank=%d, buffer=%uKB, file=%s, pid=%d)\n",
                    rank, kPerfettoBufferKB, path_buf, (int)getpid());
    }

    g_perfetto.initialized = true;
    g_perfetto.finished = false;
    g_perfetto.disabled = false;
}

static void perfetto_finalize_locked() {
    if (!g_perfetto.initialized || g_perfetto.finished) return;
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
	
    g_perfetto.finished = true;
    if (is_rank0()) {
        TAU_VERBOSE("TAU: Perfetto finalized (rank=%d)\n", (int)get_rank());
    }
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
    delete td->temp_buffers;
    td->temp_buffers = nullptr;
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
void TauTracePerfettoReinitialize(int oldid,int newid,int tid){(void)oldid;(void)newid;(void)tid;}

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
	
	    if (RtsLayer::myNode() <= -1) {
        TAU_VERBOSE("TAU: Perfetto: Dropping event (tid=%d, kind=%d) because rank is not yet set.\n", tid, kind);
        return;
		}

    if (kind == TAU_TRACE_EVENT_KIND_TEMP_FUNC) kind = TAU_TRACE_EVENT_KIND_FUNC;
    else if (kind == TAU_TRACE_EVENT_KIND_TEMP_USEREVENT) kind = TAU_TRACE_EVENT_KIND_USEREVENT;

    if (!g_perfetto.initialized.load()) {
        // If another thread is already initializing, buffer this event and
        // return immediately. This avoids blocking and preserves the event.
        if (g_perfetto.initializing.load()) {
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

        // If initialization failed, we must drop the event, but warn the user.
        if (!g_perfetto.initialized.load()) {
            fprintf(stderr,
                    "TAU: [PERFETTO_CRITICAL] Tracer initialization failed. "
                    "The current event will be dropped. "
                    "Check for permissions issues or other errors reported above.\n");
            fflush(stderr);
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
    if (ts_ns < td->last_ts_ns) {
        ts_ns = td->last_ts_ns;
    }

    if (is_func(kind)) {
        bool is_enter = (par == 1);
        emit_function(ev, is_enter, rank, tid, ts_ns);
    } else if (is_user(kind)) {
        emit_user_event((uint64_t)ev, par, rank, tid, ts_ns);
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

static void TauPerfettoMergeAllRanks(const char* dir, int num_ranks) {

    if (num_ranks <= 0) {
        printf("TAU: Perfetto merge skipped: number of ranks provided was %d.\n", num_ranks);
        return;
    }    

    bool do_gzip = TauEnv_get_perfetto_compress();
    std::string base_path = std::string(dir) + "/tau.perfetto";
    if (do_gzip) base_path += ".gz";
    std::string out_path = base_path;

    int version = 1;
    while (file_exists(out_path)) {
        out_path = base_path + "." + std::to_string(version++);
    }

    if (out_path != base_path) {
        printf("TAU: Perfetto: WARNING: Output file %s already exists. Saving to %s instead.\n",
               base_path.c_str(), out_path.c_str());
    }

    printf("TAU: Perfetto: Merging traces for %d ranks into %s\n", num_ranks, out_path.c_str());
    fflush(stdout);

    bool success = true;
    char* buf = new char[1 << 20]; // 1 MiB buffer
    int files_merged_count = 0;

    if (do_gzip) {
        gzFile out = gzopen(out_path.c_str(), "wb9");
        if (!out) {
            fprintf(stderr, "TAU: Perfetto: failed to open %s for gzip output\n", out_path.c_str());
            success = false;
        } else {
            for (int i = 0; i < num_ranks; ++i) {
                char rank_path[1024];
                snprintf(rank_path, sizeof(rank_path), "%s/tau.rank_%d.perfetto", dir, i);

                int in_fd = open(rank_path, O_RDONLY);
                if (in_fd < 0) {
                    fprintf(stderr, "TAU: Perfetto: WARNING: Could not open trace for rank %d at %s. Skipping.\n", i, rank_path);
                    continue;
                }
                files_merged_count++;

                ssize_t n_read;
                while ((n_read = read(in_fd, buf, sizeof(char) * (1 << 20))) > 0) {
                    if (gzwrite(out, buf, (unsigned)n_read) != n_read) {
                        int errnum = 0;
                        const char* zerr_msg = gzerror(out, &errnum);
                        fprintf(stderr, "TAU: Perfetto: write error: %s. Aborting merge.\n", zerr_msg);
                        success = false;
                        break;
                    }
                }
                close(in_fd);
                if (!success) break;
            }
            gzclose(out);
        }
    } else { // Not gzipped
        int out_fd = open(out_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
        if (out_fd < 0) {
            fprintf(stderr, "TAU: Perfetto: failed to open %s for output\n", out_path.c_str());
            success = false;
        } else {
            for (int i = 0; i < num_ranks; ++i) {
                char rank_path[1024];
                snprintf(rank_path, sizeof(rank_path), "%s/tau.rank_%d.perfetto", dir, i);

                int in_fd = open(rank_path, O_RDONLY);
                if (in_fd < 0) {
                    fprintf(stderr, "TAU: Perfetto: WARNING: Could not open trace for rank %d at %s. Skipping.\n", i, rank_path);
                    continue;
                }
                files_merged_count++;

                ssize_t n_read;
                while ((n_read = read(in_fd, buf, sizeof(char) * (1 << 20))) > 0) {
                    if (write(out_fd, buf, n_read) != n_read) {
                        perror("TAU: Perfetto: write error. Aborting merge");
                        success = false;
                        break;
                    }
                }
                close(in_fd);
                if (!success) break;
            }
            fsync(out_fd);
            close(out_fd);
        }
    }
    delete[] buf;

	if (success) {
        printf("TAU: Perfetto merge complete (%d of %d expected files merged).\n", files_merged_count, num_ranks);
        printf("TAU: Perfetto: Cleaning up per-rank trace files.\n");
        for (int i = 0; i < num_ranks; ++i) {
            char rank_path[1024];
            snprintf(rank_path, sizeof(rank_path), "%s/tau.rank_%d.perfetto", dir, i);
            unlink(rank_path);
			char done_path[1024];
            snprintf(done_path, sizeof(done_path), "%s/tau.rank_%d.perfetto.done", dir, i);
            unlink(done_path);
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
            perfetto_finalize_locked();
        }

        // Rank-0: print user-facing info about per-rank files
        if (is_rank0()) {
			 if (TauEnv_get_perfetto_merge()) { 
				const char* outdir = TauEnv_get_tracedir();
				if (!outdir || !*outdir) outdir = ".";
				int nodes=tau_totalnodes(0,0);
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
