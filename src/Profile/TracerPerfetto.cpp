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

#include <tau_internal.h>
#include <Profile/Profiler.h>
#include <Profile/TauEnv.h>
#include <Profile/TauTrace.h>
#include <Profile/TauTracePerfetto.h>
#include <Profile/TauMetrics.h>
#include <Profile/UserEvent.h>

#ifdef TAU_PERFETTO
#ifndef PERFETTO_ENABLE_LEGACY_TRACE_EVENTS
#define PERFETTO_ENABLE_LEGACY_TRACE_EVENTS 1
#endif
#include <perfetto.h>
#endif

extern "C" x_uint64 TauTraceGetTimeStamp(int tid);

using namespace tau;

#ifdef TAU_PERFETTO

// Disable automerge. Too expensive.
static constexpr bool kEnableAutoMerge = false;

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

static inline bool is_rank0() {
  int n = RtsLayer::myNode();
  return (n == 0 || n == -1); // treat -1 as single process before node set
}

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
};

/* ------------------------------------------------------------------------- *
 * User Event (Counter) metadata
 * ------------------------------------------------------------------------- */
struct PerfettoCounterInfo {
    bool monotonic = false;
    bool defined = false;
    std::string name;
};

/* ------------------------------------------------------------------------- *
 * Global Perfetto state
 * ------------------------------------------------------------------------- */
struct PerfettoGlobal {
    std::atomic<bool> initialized{false};
    std::atomic<bool> finished{false};
    std::atomic<bool> disabled{false};

    // Fixed default buffer size (KB);
    uint32_t buffer_kb = 16384; // 16 MB

    int file_fd = -1;
    std::string per_rank_path;
    std::unique_ptr<perfetto::TracingSession> session;

    std::mutex mutex;
    std::vector<PerfettoThreadData*> thread_data;
    std::unordered_map<uint64_t, PerfettoCounterInfo> counter_map;
    std::unordered_map<long, FunctionInfo*> function_cache;

    // Session id shared by all ranks via a filesystem token.
    long long session_id = -1;

    // Aggregated metadata
    std::vector<std::pair<std::string, std::string>> metadata_kv;
};

static PerfettoGlobal g_perfetto;

/* ------------------------------------------------------------------------- *
 * Helpers
 * ------------------------------------------------------------------------- */
static inline uint64_t get_rank() {
    int r = RtsLayer::myNode();
    if (r < 0) r = 0;
    return (uint64_t)r;
}

static void ensure_thread_vector(int tid) {
    if ((int)g_perfetto.thread_data.size() <= tid) {
        std::lock_guard<std::mutex> lk(g_perfetto.mutex);
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
    for (auto it = TheEventDB().begin(); it != TheEventDB().end(); ++it)
        if ((*it)->GetId() == id) return (*it)->IsMonotonicallyIncreasing();
    return false;
}

static uint64_t compute_flow_id(int src, int dst, int tag, int comm) {
    uint64_t s = (uint64_t)(src & 0xFFFF);
    uint64_t d = (uint64_t)(dst & 0xFFFF);
    uint64_t t = (uint64_t)(tag & 0xFFFF);
    uint64_t c = (uint64_t)(comm & 0xFFFF);
    return (s << 48) | (d << 32) | (t << 16) | c;
}

static FunctionInfo* get_function_info(long func_id) {
    auto it = g_perfetto.function_cache.find(func_id);
    if (it != g_perfetto.function_cache.end()) {
        return it->second;
    }
    FunctionInfo* fi = nullptr;
    for (auto fit = TheFunctionDB().begin(); fit != TheFunctionDB().end(); ++fit) {
        if ((*fit)->GetFunctionId() == func_id) {
            fi = *fit;
            break;
        }
    }
    g_perfetto.function_cache[func_id] = fi;
    return fi;
}

static const char* get_function_name(long func_id, char* fallback_buf, size_t buf_size) {
    FunctionInfo* fi = get_function_info(func_id);
    if (fi) {
        const char* name = fi->GetName();
        if (name && name[0] != '\0') return name;
    }
    snprintf(fallback_buf, buf_size, "Function_%ld", func_id);
    return fallback_buf;
}

static const char* get_function_type(long func_id) {
    FunctionInfo* fi = get_function_info(func_id);
    if (fi) {
        const char* type = fi->GetType();
        if (type && type[0] != '\0') return type;
    }
    return "";
}

static inline bool contains_str(const char* hay, const char* needle) {
    if (!hay || !needle) return false;
    return strstr(hay, needle) != nullptr;
}

/* ------------------------------------------------------------------------- *
 * Session token using filesystem
 * ------------------------------------------------------------------------- */
static long long now_ns_monotonicish() {
#if defined(CLOCK_REALTIME)
    struct timespec ts;
    if (clock_gettime(CLOCK_REALTIME, &ts) == 0) {
        return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
    }
#endif
    return (long long)time(nullptr) * 1000000000LL;
}

static long long read_session_file_blocking(const char* path) {
    for (int attempt = 0; attempt < 200; ++attempt) { // up to ~2s
        int fd = open(path, O_RDONLY);
        if (fd >= 0) {
            char buf[64] = {0};
            ssize_t rd = read(fd, buf, sizeof(buf)-1);
            close(fd);
            if (rd > 0) {
                char* endp = nullptr;
                long long sid = strtoll(buf, &endp, 10);
                if (sid > 0) return sid;
            }
        }
        usleep(10000); // 10ms
    }
    return -1;
}

static long long get_or_create_session_id() {
    const char* dir = TauEnv_get_tracedir();
    char path[1024];
    snprintf(path, sizeof(path), "%s/.tau.perfetto.session", dir ? dir : ".");

    int fd = open(path, O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (fd >= 0) {
        long long sid = now_ns_monotonicish();
        char buf[64];
        int len = snprintf(buf, sizeof(buf), "%lld\n", sid);
        if (len > 0) {
            (void)write(fd, buf, (size_t)len);
            fsync(fd);
        }
        close(fd);
        if (is_rank0()) {
            TAU_VERBOSE("TAU: Perfetto: created session file %s\n", path);
        }
        return sid;
    }

    if (errno == EEXIST) {
        long long sid = read_session_file_blocking(path);
        if (sid > 0) {
            if (is_rank0()) {
                TAU_VERBOSE("TAU: Perfetto: joined session id %lld via %s\n", sid, path);
            }
            return sid;
        }
        // Attempt recovery if the file was empty/unreadable
        fd = open(path, O_CREAT | O_EXCL | O_WRONLY, 0644);
        if (fd >= 0) {
            long long sid2 = now_ns_monotonicish();
            char buf[64];
            int len = snprintf(buf, sizeof(buf), "%lld\n", sid2);
            if (len > 0) {
                (void)write(fd, buf, (size_t)len);
                fsync(fd);
            }
            close(fd);
            if (is_rank0()) {
                TAU_VERBOSE("TAU: Perfetto: recovered session id %lld via %s\n", sid2, path);
            }
            return sid2;
        }
    }

    long long fallback = ((long long)getpid() << 32) ^ now_ns_monotonicish();
    if (is_rank0()) {
        TAU_VERBOSE("TAU: Perfetto: using fallback session id %lld (token unavailable)\n", fallback);
    }
    return fallback;
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

static void emit_thread_descriptor(PerfettoThreadData* td, int tid) {
    auto track = perfetto::ThreadTrack::ForThread((uint64_t)td->os_tid);
    auto desc = track.Serialize();
    std::string tname = (tid == 0) ? "main thread"
                                   : (std::string("TAU TID ") + std::to_string(tid));
    desc.mutable_thread()->set_thread_name(tname.c_str());
    perfetto::TrackEvent::SetTrackDescriptor(track, desc);
    if (is_rank0()) {
        TAU_VERBOSE("TAU: Perfetto: thread descriptor set: name='%s', tau_tid=%d, os_tid=%" PRId64 "\n",
                    tname.c_str(), tid, td->os_tid);
    }
}

static void perfetto_init_locked() {
    if (g_perfetto.initialized) return;

    // Establish a cross-rank session id without MPI via filesystem rendezvous.
    g_perfetto.session_id = get_or_create_session_id();

    perfetto::TracingInitArgs args;
    args.backends = perfetto::kInProcessBackend;
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();

    // Per-rank file: ${TRACEDIR}/tau.<session>.rank_<rank>.perfetto
    char path_buf[1024];
    snprintf(path_buf, sizeof(path_buf), "%s/tau.%lld.rank_%d.perfetto",
             TauEnv_get_tracedir(),
             g_perfetto.session_id,
             (int)get_rank());
    g_perfetto.per_rank_path = path_buf;

    g_perfetto.file_fd = open(path_buf, O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (g_perfetto.file_fd < 0) {
        if (is_rank0()) {
            TAU_VERBOSE("TAU: Perfetto: Failed to open %s\n", path_buf);
        }
        g_perfetto.disabled = true;
        g_perfetto.initialized = true;
        g_perfetto.finished = true;
        return;
    }

    perfetto::TraceConfig cfg;
    auto* buf = cfg.add_buffers();
    buf->set_size_kb(g_perfetto.buffer_kb);
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
        // Use explicit timestamps
        m->set_disable_incremental_timestamps(true);
        ds->mutable_config()->set_track_event_config_raw(te_cfg.SerializeAsString());
    }
    // Intentionally avoid periodic incremental state clears.

    g_perfetto.session = perfetto::Tracing::NewTrace();
    g_perfetto.session->Setup(cfg, g_perfetto.file_fd);
    g_perfetto.session->StartBlocking();

    // Emit the process descriptor immediately so the UI always has a name.
    int rank = (int)get_rank();
    emit_process_descriptor(rank);
    if (is_rank0()) {
        TAU_VERBOSE("TAU: Perfetto started (rank=%d, buffer=%uKB, file=%s, pid=%d)\n",
                    rank, g_perfetto.buffer_kb, path_buf, (int)getpid());
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
    g_perfetto.finished = true;
    if (is_rank0()) {
        TAU_VERBOSE("TAU: Perfetto finalized (rank=%d)\n", (int)get_rank());
    }
}

/* ------------------------------------------------------------------------- *
 * Thread metadata
 * ------------------------------------------------------------------------- */
static void ensure_thread_metadata(int rank, int tid) {
    (void)rank;
    ensure_thread_vector(tid);
    PerfettoThreadData* td = g_perfetto.thread_data[tid];
    if (td->track_defined) return;

    td->os_tid = get_os_tid_now();
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
        std::lock_guard<std::mutex> lk(g_perfetto.mutex);
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

    // Emit at end-of-run (current timestamp) so it's easy to find at the timeline end
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
    return TauTracePerfettoInitTS(tid, TauTraceGetTimeStamp(tid));
}

int TauTracePerfettoInitTS(int tid, x_uint64 /*ts*/) {
    TauInternalFunctionGuard guard;
    if (g_perfetto.initialized || g_perfetto.finished) return 0;

    // If node id is not set yet, defer initialization.
    if (RtsLayer::myNode() <= -1) {
        return 1;
    }

    std::lock_guard<std::mutex> lk(g_perfetto.mutex);
    if (!g_perfetto.initialized) {
        perfetto_init_locked();
    }
    (void)tid;
    return 0;
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
 * Metadata capture (aggregate; no per-item emission/logging)
 * ------------------------------------------------------------------------- */
void TauTracePerfettoMetadata(const char* name, const char* value, int tid) {
    if (g_perfetto.finished || g_perfetto.disabled) return;

    TauInternalFunctionGuard guard;

    if (!g_perfetto.initialized) {
        TauTracePerfettoInit(tid);
        if (!g_perfetto.initialized) return;
    }

    std::lock_guard<std::mutex> lk(g_perfetto.mutex);
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
    PerfettoThreadData* td = g_perfetto.thread_data[tid];
    auto track = perfetto::ThreadTrack::ForThread((uint64_t)td->os_tid);

    static thread_local char name_buf[256];
    const char* name = get_function_name(func_id, name_buf, sizeof(name_buf));
    const char* ftype = get_function_type(func_id);

    if (contains_str(name, "TauTraceClockOffset")) {
        return;
    }

    if (ftype && ftype[0] != '\0') {
        if (is_enter) {
            TRACE_EVENT_BEGIN("tau", perfetto::DynamicString{name}, track, ts_ns, "type", ftype);
        } else {
            TRACE_EVENT_END("tau", track, ts_ns);
        }
    } else {
        if (is_enter) {
            TRACE_EVENT_BEGIN("tau", perfetto::DynamicString{name}, track, ts_ns);
        } else {
            TRACE_EVENT_END("tau", track, ts_ns);
        }
    }
}

/* ------------------------------------------------------------------------- *
 * User events
 * ------------------------------------------------------------------------- */
static void emit_user_event(uint64_t event_id, x_int64 raw_value,
                            int rank, int tid, uint64_t ts_ns) {
    ensure_thread_metadata(rank, tid);

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
        PerfettoCounterInfo ci;
        ci.monotonic = mono;
        ci.defined = true;
        ci.name = ev_name;
        g_perfetto.counter_map[event_id] = ci;
        it = g_perfetto.counter_map.find(event_id);
    }

    const PerfettoCounterInfo& ci = it->second;

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

/* ------------------------------------------------------------------------- *
 * Main event dispatcher
 * ------------------------------------------------------------------------- */
void TauTracePerfettoEventWithNodeId(long int ev, x_int64 par, int tid,
                                     x_uint64 ts_us, int use_ts, int node_id, int kind) {
    if (g_perfetto.finished || g_perfetto.disabled) return;

    TauInternalFunctionGuard guard;

    if (kind == TAU_TRACE_EVENT_KIND_TEMP_FUNC) kind = TAU_TRACE_EVENT_KIND_FUNC;
    else if (kind == TAU_TRACE_EVENT_KIND_TEMP_USEREVENT) kind = TAU_TRACE_EVENT_KIND_USEREVENT;

    if (!g_perfetto.initialized) {
#if defined(TAU_MPI) || defined(TAU_SHMEM)
        ensure_thread_vector(tid);
        PerfettoThreadData* td = g_perfetto.thread_data[tid];
        if (!td->temp_buffers) td->temp_buffers = new std::vector<temp_buffer_entry>();
        x_uint64 t_us = use_ts ? ts_us : TauTraceGetTimeStamp(tid);
        td->temp_buffers->emplace_back(ev, t_us, par, kind);
        return;
#else
        if (use_ts)
            TauTracePerfettoInitTS(tid, ts_us);
        else
            TauTracePerfettoInit(tid);
        if (!g_perfetto.initialized) return;
#endif
    }

#if defined(TAU_MPI) || defined(TAU_SHMEM)
    ensure_thread_vector(tid);
    if (!g_perfetto.thread_data[tid]->buffers_written) {
        write_temp_buffer(tid, node_id);
    }
#endif

    int rank = node_id;
    uint64_t actual_ts_us = use_ts ? ts_us : TauTraceGetTimeStamp(tid);
    uint64_t ts_ns = us_to_ns(actual_ts_us);

    if (is_func(kind)) {
        bool is_enter = (par == 1);
        emit_function(ev, is_enter, rank, tid, ts_ns);

    } else if (is_user(kind)) {
        emit_user_event((uint64_t)ev, par, rank, tid, ts_ns);

    }
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
    uint64_t flow_id = compute_flow_id(src_rank, dst_rank, type, 0);
    int t = RtsLayer::myThread();

    uint64_t actual_ts_us = use_ts ? ts_us : TauTraceGetTimeStamp(t);
    uint64_t ts_ns = us_to_ns(actual_ts_us);

    emit_mpi_message(send_or_recv == TAU_MESSAGE_SEND, flow_id,
                     src_rank, dst_rank, type, length, my_rank, t, ts_ns);

    (void)node_id;
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
    if (g_perfetto.session && g_perfetto.initialized && !g_perfetto.finished) {
        g_perfetto.session->FlushBlocking();
    }
    (void)tid;
}

/* ------------------------------------------------------------------------- *
 * Merge traces (disabled by default; only used if kEnableAutoMerge == true)
 * ------------------------------------------------------------------------- */
static void merge_rank_traces_dirscan_and_cleanup() {
    if (!kEnableAutoMerge) return;
    if (RtsLayer::myNode() != 0) return;

    const char* outdir = TauEnv_get_tracedir();
    if (!outdir || !*outdir) outdir = ".";
    std::string out_path = std::string(outdir) + "/tau.perfetto";

    // Gather all rank files for this session id
    char prefix[256];
    snprintf(prefix, sizeof(prefix), "tau.%lld.rank_", g_perfetto.session_id);
    const char* suffix = ".perfetto";

    std::vector<std::pair<int,std::string>> rank_files;
    DIR* d = opendir(outdir);
    if (d) {
        struct dirent* ent;
        while ((ent = readdir(d)) != nullptr) {
            const char* nm = ent->d_name;
            if (!nm) continue;
            size_t ln = strlen(nm), lp = strlen(prefix), ls = strlen(suffix);
            if (ln > lp + ls && strncmp(nm, prefix, lp) == 0 &&
                strcmp(nm + (ln - ls), suffix) == 0) {

                std::string full = std::string(outdir) + "/" + nm;
                struct stat st{};
                if (stat(full.c_str(), &st) != 0) continue;
                if (st.st_size <= 0) continue;

                std::string mid(nm + lp, nm + ln - ls);
                char* endp = nullptr;
                long r = strtol(mid.c_str(), &endp, 10);
                if (endp && *endp == '\0' && r >= 0) {
                    rank_files.emplace_back((int)r, full);
                }
            }
        }
        closedir(d);
    }
    std::sort(rank_files.begin(), rank_files.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });

    int out_fd = open(out_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
    if (out_fd < 0) {
        if (is_rank0()) {
            TAU_VERBOSE("TAU: Perfetto merge: cannot open %s\n", out_path.c_str());
        }
        return;
    }

    bool wrote_any = false;
    std::vector<char> buffer(1<<22); // 4MB buffer to reduce syscalls if enabled
    for (const auto& rf : rank_files) {
        int in_fd = open(rf.second.c_str(), O_RDONLY);
        if (in_fd < 0) {
            if (is_rank0()) {
                TAU_VERBOSE("TAU: Perfetto merge: missing %s (skipping)\n", rf.second.c_str());
            }
            continue;
        }
        ssize_t rd;
        while ((rd = read(in_fd, buffer.data(), buffer.size())) > 0) {
            ssize_t off = 0;
            while (off < rd) {
                ssize_t wr = write(out_fd, buffer.data() + off, (size_t)(rd - off));
                if (wr < 0) { if (is_rank0()) TAU_VERBOSE("TAU: Perfetto merge: write error\n"); break; }
                off += wr;
            }
            wrote_any = true;
        }
        close(in_fd);
    }
    fsync(out_fd);
    close(out_fd);

    if (wrote_any) {
        if (is_rank0()) {
            TAU_VERBOSE("TAU: Perfetto: concatenated %zu rank traces into %s\n",
                        rank_files.size(), out_path.c_str());
        }
        // Delete original per-rank files
        for (const auto& rf : rank_files) {
            (void)unlink(rf.second.c_str());
        }
        // Delete session token file
        char sess_path[1024];
        snprintf(sess_path, sizeof(sess_path), "%s/.tau.perfetto.session", outdir);
        (void)unlink(sess_path);
    } else {
        if (is_rank0()) {
            TAU_VERBOSE("TAU: Perfetto: merge wrote no data; leaving per-rank files in place\n");
        }
    }
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
        // Emit aggregated metadata at the end-of-run on main thread.
        emit_aggregated_metadata_on_main_thread();

        {
            std::lock_guard<std::mutex> lk(g_perfetto.mutex);
            perfetto_finalize_locked();
        }

        // By default, do NOT merge automatically.
        merge_rank_traces_dirscan_and_cleanup();

        // Rank-0: print user-facing instructions (concatenate then compress).
        if (is_rank0()) {
            const char* outdir = TauEnv_get_tracedir();
            if (!outdir || !*outdir) outdir = ".";
            // Inform where per-rank files are and how to merge/compress later.
            printf("TAU: Perfetto trace files written to %s\n", outdir);
            printf("TAU: To merge per-rank traces for this run:\n");
            printf("TAU:   cd %s && cat tau.%lld.rank_*.perfetto > tau.perfetto\n", outdir, g_perfetto.session_id);
            printf("TAU: To compress the merged trace:\n");
            printf("TAU:   cd %s && gzip -c tau.perfetto > tau.perfetto.gz\n", outdir);
            fflush(stdout);
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