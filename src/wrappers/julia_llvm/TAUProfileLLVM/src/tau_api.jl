# ============================================================================
# tau_api.jl — direct bindings to the TAU C API
#
# Part of the TAUProfile module (LLVM backend).
# ============================================================================

# ============================================================================
# Handles
# ============================================================================

"""
    TauTimer(name; type="", group="TAU_USER")

Handle to a TAU timer, an immutable wrapper around the TAU `FunctionInfo`
for a timer called `name`. Equivalent to `TAU_PROFILE_TIMER(t, name, type, group)` in
C. The timer is registered once, with the given type string, and associated
with the profile group named by `group`.

# Example
```julia
t = TauTimer("solve"; type="Float64 (Int)", group="TAU_USER")
tau_start(t)
solve(n)
tau_stop(t)
```
"""
struct TauTimer
    ptr::Ptr{Cvoid}   # FunctionInfo*
end

function TauTimer(name::AbstractString; type::AbstractString="",
                  group::AbstractString="TAU_USER")
    isempty(strip(name)) && throw(ArgumentError("timer name must not be empty"))
    isempty(strip(group)) && throw(ArgumentError("group name must not be empty"))
    _tau_active() || return TauTimer(C_NULL)
    grp = ccall((:Tau_get_profile_group, _libTAU[]), Culong, (Cstring,), group)
    handle = Ref{Ptr{Cvoid}}(C_NULL)
    ccall((:Tau_profile_c_timer, _libTAU[]), Cvoid,
          (Ptr{Ptr{Cvoid}}, Cstring, Cstring, Culong, Cstring),
          handle, name, type, grp, group)
    return TauTimer(handle[])
end

# True when libTAU is loaded and `t` refers to a real FunctionInfo.
_live(t::TauTimer) = _tau_active() && t.ptr != C_NULL

# ============================================================================
# Task pinning and thread id
# ============================================================================

# Pin the current task to its OS thread. libTAU keeps a timer stack per OS
# thread, so a task must not migrate while one of its timers is running.
# TODO remove forcing task sticky when we can handle task migration ~nchaimov
_pin_task!() = (current_task().sticky = true; nothing)

# TAU's id for the calling OS thread.
_tau_thread() = ccall((:Tau_get_thread, _libTAU[]), Cint, ())

# ============================================================================
# Timers
# ============================================================================

"""
    tau_start(t::TauTimer)

Start the timer given by `t` on the current thread.
"""
function tau_start(t::TauTimer)
    _live(t) || return nothing
    _pin_task!()
    ccall((:Tau_start_timer, _libTAU[]), Cvoid, (Ptr{Cvoid}, Cint, Cint),
          t.ptr, 0, _tau_thread())
    nothing
end

"""
    tau_stop(t::TauTimer)

Stop the timer given by `t`.
"""
function tau_stop(t::TauTimer)
    _live(t) || return nothing
    _pin_task!()
    ccall((:Tau_stop_timer, _libTAU[]), Cvoid, (Ptr{Cvoid}, Cint),
          t.ptr, _tau_thread())
    nothing
end

# ============================================================================
# Readback
# ============================================================================

"""
    tau_get_calls(t::TauTimer) -> Int

Number of times `t` has been started on the current thread. Returns `nothing`
when libTAU is not loaded or `t` is null.
"""
function tau_get_calls(t::TauTimer)
    _live(t) || return nothing
    _pin_task!()
    calls = Ref{Clong}(0)
    ccall((:Tau_get_calls, _libTAU[]), Cvoid, (Ptr{Cvoid}, Ptr{Clong}, Cint),
          t.ptr, calls, _tau_thread())
    return Int(calls[])
end

"""
    tau_get_child_calls(t::TauTimer) -> Int

Number of timers started while `t` was the innermost open timer on the
current thread. Returns `nothing` when libTAU is not loaded or `t` is null.
"""
function tau_get_child_calls(t::TauTimer)
    _live(t) || return nothing
    _pin_task!()
    calls = Ref{Clong}(0)
    ccall((:Tau_get_child_calls, _libTAU[]), Cvoid, (Ptr{Cvoid}, Ptr{Clong}, Cint),
          t.ptr, calls, _tau_thread())
    return Int(calls[])
end

# Upper bound on the number of values TAU writes into a per-timer metric
# array (TAU_MAX_COUNTERS in include/Profile/Profiler.h).
const _TAU_MAX_COUNTERS = 25

"""
    tau_get_inclusive(t::TauTimer) -> Vector{Float64}

Inclusive value of `t` on the current thread for each counter, in the order
given by [`tau_counter_names`](@ref). Returns an empty vector when libTAU is
not loaded or `t` is null.
"""
function tau_get_inclusive(t::TauTimer)
    _live(t) || return Float64[]
    n = length(tau_counter_names())
    n == 0 && return Float64[]
    _pin_task!()
    vals = zeros(Float64, max(n, _TAU_MAX_COUNTERS))
    ccall((:Tau_get_inclusive_values, _libTAU[]), Cvoid,
          (Ptr{Cvoid}, Ptr{Cdouble}, Cint), t.ptr, vals, _tau_thread())
    return vals[1:n]
end

"""
    tau_get_exclusive(t::TauTimer) -> Vector{Float64}

Exclusive value of `t` on the current thread for each counter, in the order
given by [`tau_counter_names`](@ref). Returns an empty vector when libTAU is
not loaded or `t` is null.
"""
function tau_get_exclusive(t::TauTimer)
    _live(t) || return Float64[]
    n = length(tau_counter_names())
    n == 0 && return Float64[]
    _pin_task!()
    vals = zeros(Float64, max(n, _TAU_MAX_COUNTERS))
    ccall((:Tau_get_exclusive_values, _libTAU[]), Cvoid,
          (Ptr{Cvoid}, Ptr{Cdouble}, Cint), t.ptr, vals, _tau_thread())
    return vals[1:n]
end

# Copy `n` C strings out of a malloc'd array of pointers TAU handed us, then
# free what TAU allocated: the array always, the strings only when TAU
# strdup'd them rather than pointing into its own storage.
function _take_c_string_list(list::Ptr{Ptr{Cchar}}, n::Integer; free_strings::Bool)
    list == C_NULL && return String[]
    out = Vector{String}(undef, max(n, 0))
    for i in 1:n
        p = unsafe_load(list, i)
        out[i] = p == C_NULL ? "" : unsafe_string(p)
        free_strings && p != C_NULL && Libc.free(p)
    end
    Libc.free(list)
    return out
end

"""
    tau_counter_names() -> Vector{String}

Names of the counters TAU is collecting (`TAU_METRICS`). Returns an empty
vector when libTAU is not loaded.
"""
function tau_counter_names()
    _tau_active() || return String[]
    list = Ref{Ptr{Ptr{Cchar}}}(C_NULL)
    n = Ref{Cint}(0)
    ccall((:Tau_get_counter_info, _libTAU[]), Cvoid,
          (Ptr{Ptr{Ptr{Cchar}}}, Ptr{Cint}), list, n)
    return _take_c_string_list(list[], n[]; free_strings=true)
end

"""
    tau_function_names() -> Vector{String}

Names of every timer TAU has registered so far, in registration order.
Returns an empty vector when libTAU is not loaded.
"""
function tau_function_names()
    _tau_active() || return String[]
    list = Ref{Ptr{Ptr{Cchar}}}(C_NULL)
    n = Ref{Cint}(0)
    ccall((:Tau_the_function_list, _libTAU[]), Cvoid,
          (Ptr{Ptr{Ptr{Cchar}}}, Ptr{Cint}), list, n)
    return _take_c_string_list(list[], n[]; free_strings=false)
end

# ============================================================================
# Timer attributes
# ============================================================================

"""
    tau_set_name(t::TauTimer, name)

Change the name under which `t` is reported. No effect when libTAU is not loaded
or `t` is null.
"""
function tau_set_name(t::TauTimer, name::AbstractString)
    isempty(strip(name)) && throw(ArgumentError("timer name must not be empty"))
    _live(t) || return nothing
    ccall((:Tau_profile_set_name, _libTAU[]), Cvoid, (Ptr{Cvoid}, Cstring), t.ptr, name)
    nothing
end

"""
    tau_set_type(t::TauTimer, type)

Change the type string reported alongside the name of `t`. No effect when libTAU
is not loaded or `t` is null.
"""
function tau_set_type(t::TauTimer, type::AbstractString)
    _live(t) || return nothing
    ccall((:Tau_profile_set_type, _libTAU[]), Cvoid, (Ptr{Cvoid}, Cstring), t.ptr, type)
    nothing
end

"""
    tau_set_group(t::TauTimer, group)

Move `t` into the profile group named `group`. No effect when libTAU is not
loaded or `t` is null.
"""
function tau_set_group(t::TauTimer, group::AbstractString)
    isempty(strip(group)) && throw(ArgumentError("group name must not be empty"))
    _live(t) || return nothing
    ccall((:Tau_profile_set_group_name, _libTAU[]), Cvoid, (Ptr{Cvoid}, Cstring), t.ptr, group)
    nothing
end

# ============================================================================
# User events
# ============================================================================

"""
    TauEvent(name)

Handle to a TAU user event: an immutable wrapper around the `TauUserEvent`
that TAU keeps for `name`. Equivalent to `TAU_REGISTER_EVENT(e, name)` in C.
Use it with [`tau_event`](@ref) when the same event is triggered many times,
to skip the name lookup on each trigger.

TAU looks events up by name, so a second `TauEvent` with the same name refers
to the same event.
"""
struct TauEvent
    ptr::Ptr{Cvoid}   # TauUserEvent*
end

function TauEvent(name::AbstractString)
    isempty(strip(name)) && throw(ArgumentError("event name must not be empty"))
    _tau_active() || return TauEvent(C_NULL)
    ptr = ccall((:Tau_get_userevent, _libTAU[]), Ptr{Cvoid}, (Cstring,), name)
    return TauEvent(ptr)
end

_live(e::TauEvent) = _tau_active() && e.ptr != C_NULL

"""
    tau_event(name, value)
    tau_event(e::TauEvent, value)

Trigger the user event given by `name` or by handle `e` with `value`. TAU
records the count, minimum, maximum, mean and standard deviation of the
values an event is triggered with. Equivalent to `TAU_TRIGGER_EVENT` and
`TAU_EVENT` in C.
"""
function tau_event(name::AbstractString, value::Real)
    isempty(strip(name)) && throw(ArgumentError("event name must not be empty"))
    _tau_active() || return nothing
    _pin_task!()
    ccall((:Tau_trigger_userevent, _libTAU[]), Cvoid, (Cstring, Cdouble), name, value)
    nothing
end

function tau_event(e::TauEvent, value::Real)
    _live(e) || return nothing
    _pin_task!()
    ccall((:Tau_userevent, _libTAU[]), Cvoid, (Ptr{Cvoid}, Cdouble), e.ptr, value)
    nothing
end

"""
    tau_context_event(name, value)

Trigger the context event given by `name` with `value`. A context event is a
user event whose recorded name is suffixed with the call path of the timers
open when it is triggered, so the same event is reported separately for each
context. Equivalent to `TAU_TRIGGER_CONTEXT_EVENT` in C.
"""
function tau_context_event(name::AbstractString, value::Real)
    isempty(strip(name)) && throw(ArgumentError("event name must not be empty"))
    _tau_active() || return nothing
    _pin_task!()
    ccall((:Tau_trigger_context_event, _libTAU[]), Cvoid, (Cstring, Cdouble), name, value)
    nothing
end

# ============================================================================
# Metadata
# ============================================================================

"""
    tau_metadata(name, value)

Record `value` under `name` in the profile's metadata block. Any `value` is
accepted and stored as `string(value)`. Equivalent to `TAU_METADATA` in C.
"""
function tau_metadata(name::AbstractString, value)
    isempty(strip(name)) && throw(ArgumentError("metadata name must not be empty"))
    _tau_active() || return nothing
    ccall((:Tau_metadata, _libTAU[]), Cvoid, (Cstring, Cstring), name, string(value))
    nothing
end

"""
    tau_context_metadata(name, value)

Record `value` under `name`, attached to the timer that is open when the call
is made, so the profile reports which timer instance it belongs to. Any
`value` is accepted and stored as `string(value)`. Equivalent to
`TAU_CONTEXT_METADATA` in C.
"""
function tau_context_metadata(name::AbstractString, value)
    isempty(strip(name)) && throw(ArgumentError("metadata name must not be empty"))
    _tau_active() || return nothing
    _pin_task!()
    ccall((:Tau_context_metadata, _libTAU[]), Cvoid, (Cstring, Cstring), name, string(value))
    nothing
end

# ============================================================================
# Dynamic timers
# ============================================================================

"""
    tau_dynamic_start(name)

Start a new dynamic timer -- a timer whose name is `name` with the iteration
count appended, so that each start/stop pair is reported as its own entry
(`name[0]`, `name[1]`, ...). Equivalent to `TAU_DYNAMIC_TIMER_START` in C.

# Example
```julia
for step in 1:nsteps
    tau_dynamic_start("timestep")
    advance!(state)
    tau_dynamic_stop("timestep")
end
```
"""
function tau_dynamic_start(name::AbstractString)
    isempty(strip(name)) && throw(ArgumentError("timer name must not be empty"))
    _tau_active() || return nothing
    _pin_task!()
    ccall((:Tau_dynamic_start, _libTAU[]), Cvoid, (Cstring, Cint), name, 0)
    nothing
end

"""
    tau_dynamic_stop(name)

Stop the dynamic timer for the current iteration of `name` and advance the
iteration count. Equivalent to `TAU_DYNAMIC_TIMER_STOP` in C.
"""
function tau_dynamic_stop(name::AbstractString)
    isempty(strip(name)) && throw(ArgumentError("timer name must not be empty"))
    _tau_active() || return nothing
    _pin_task!()
    ccall((:Tau_dynamic_stop, _libTAU[]), Cvoid, (Cstring, Cint), name, 0)
    nothing
end

# ============================================================================
# Runtime control
# ============================================================================

"""
    tau_enable_instrumentation()
    tau_disable_instrumentation()

Turn all TAU timers on or off at runtime. While instrumentation is disabled,
timer starts and stops -- from `tau_start`, `TauTimer` handles, dynamic timers
and the hooks inserted by `tau_rewrite_and_call` alike -- are ignored, so a
region run in that window is not reflected in the profile. Equivalent to
`TAU_ENABLE_INSTRUMENTATION` and `TAU_DISABLE_INSTRUMENTATION` in C.

A timer started while instrumentation is enabled must also be stopped while
it is enabled, or TAU will report an overlapping timer error.

# Example
```julia
tau_disable_instrumentation()
tau_rewrite_and_call(solve, n)      # not measured
tau_enable_instrumentation()
tau_rewrite_and_call(solve, n)      # measured
```
"""
function tau_enable_instrumentation()
    _tau_active() || return nothing
    ccall((:Tau_enable_instrumentation, _libTAU[]), Cvoid, ())
    nothing
end

@doc (@doc tau_enable_instrumentation)
function tau_disable_instrumentation()
    _tau_active() || return nothing
    ccall((:Tau_disable_instrumentation, _libTAU[]), Cvoid, ())
    nothing
end

"""
    tau_enable_group(group)
    tau_disable_group(group)

Enable or disable every timer in the profile group named `group`.
Starts and stops of a timer in a disabled group are ignored. Equivalent to
`TAU_ENABLE_GROUP_NAME` and `TAU_DISABLE_GROUP_NAME` in C.
"""
function tau_enable_group(group::AbstractString)
    isempty(strip(group)) && throw(ArgumentError("group name must not be empty"))
    _tau_active() || return nothing
    ccall((:Tau_enable_group_name, _libTAU[]), Culong, (Cstring,), group)
    nothing
end

@doc (@doc tau_enable_group)
function tau_disable_group(group::AbstractString)
    isempty(strip(group)) && throw(ArgumentError("group name must not be empty"))
    _tau_active() || return nothing
    ccall((:Tau_disable_group_name, _libTAU[]), Culong, (Cstring,), group)
    nothing
end

"""
    tau_enable_all_groups()
    tau_disable_all_groups()

Enable or disable every profile group at once. Equivalent to
`TAU_ENABLE_ALL_GROUPS` and `TAU_DISABLE_ALL_GROUPS` in C.
"""
function tau_enable_all_groups()
    _tau_active() || return nothing
    ccall((:Tau_enable_all_groups, _libTAU[]), Culong, ())
    nothing
end

@doc (@doc tau_enable_all_groups)
function tau_disable_all_groups()
    _tau_active() || return nothing
    ccall((:Tau_disable_all_groups, _libTAU[]), Culong, ())
    nothing
end

# ============================================================================
# Output
# ============================================================================

"""
    tau_dump()
    tau_dump(prefix)

Write the profile collected so far without waiting for the program to end.
`tau_dump()` writes the current thread's data to `dump.<node>.<context>.<thread>`
in the profile directory; `tau_dump(prefix)` writes every thread's data to
`<prefix>.<node>.<context>.<thread>`. The files have the same format as the
`profile.*` files written at exit, which are still written. Equivalent to
`TAU_DB_DUMP` and `TAU_DB_DUMP_PREFIX` in C.
"""
function tau_dump()
    _tau_active() || return nothing
    _pin_task!()
    ccall((:Tau_dump, _libTAU[]), Cint, ())
    nothing
end

function tau_dump(prefix::AbstractString)
    isempty(strip(prefix)) && throw(ArgumentError("dump prefix must not be empty"))
    _tau_active() || return nothing
    ccall((:Tau_dump_prefix, _libTAU[]), Cint, (Cstring,), prefix)
    nothing
end

"""
    tau_snapshot(name)

Record a snapshot of the profile under `name`. Snapshots are written to
`snapshot.<node>.<context>.<thread>` in the profile directory and can be
compared against one another in ParaProf. Equivalent to
`TAU_PROFILE_SNAPSHOT` in C.
"""
function tau_snapshot(name::AbstractString)
    isempty(strip(name)) && throw(ArgumentError("snapshot name must not be empty"))
    _tau_active() || return nothing
    _pin_task!()
    ccall((:Tau_profile_snapshot, _libTAU[]), Cvoid, (Cstring,), name)
    nothing
end

"""
    tau_exit(msg="Julia exit")

Alert TAU to an exit call. Call it before a script ends with `exit()` or an
error exit, so that any profiles or event traces are written to disk before
quitting. Equivalent to `TAU_PROFILE_EXIT` in C.

# Example
```julia
if !isfile(input)
    tau_exit("input file missing")
    exit(1)
end
```
"""
function tau_exit(msg::AbstractString="Julia exit")
    _tau_active() || return nothing
    ccall((:Tau_exit, _libTAU[]), Cvoid, (Cstring,), msg)
    nothing
end

# ============================================================================
# Node and thread identity
# ============================================================================

"""
    tau_set_node(node)

Set the node id of the executing process for profiling and tracing. Tasks
are identified by node, context and thread ids, with profile files being
named `profile.<node>.<context>.<thread>` accordingly. TAU sets the node
itself under MPI. Equivalent to `TAU_PROFILE_SET_NODE` in C.
"""
function tau_set_node(node::Integer)
    node < 0 && throw(ArgumentError("node id must not be negative"))
    _tau_active() || return nothing
    ccall((:Tau_set_node, _libTAU[]), Cvoid, (Cint,), node)
    nothing
end

"""
    tau_get_node() -> Int

Node id of the executing process, as set by [`tau_set_node`](@ref) or by
TAU's MPI support; `-1` when no node has been set or libTAU is not loaded.
Equivalent to `TAU_PROFILE_GET_NODE` in C.
"""
function tau_get_node()
    _tau_active() || return -1
    return Int(ccall((:Tau_get_node, _libTAU[]), Cint, ()))
end

"""
    tau_get_thread() -> Int

TAU's id for the calling OS thread; `-1` when libTAU is not loaded.
Equivalent to `TAU_PROFILE_GET_THREAD` in C.
"""
function tau_get_thread()
    _tau_active() || return -1
    _pin_task!()
    return Int(_tau_thread())
end

# ============================================================================
# Memory tracking
# ============================================================================

"""
    tau_track_memory_here()

Record the heap memory in use at this point as the user event
`Heap Memory Used (KB)`. Equivalent to `TAU_TRACK_MEMORY_HERE` in C.
"""
function tau_track_memory_here()
    _tau_active() || return nothing
    _pin_task!()
    ccall((:Tau_track_memory_here, _libTAU[]), Cvoid, ())
    nothing
end

"""
    tau_track_memory_footprint_here()

Record the process's resident set size and its peak, together with its
thread count and context switches, as the context events
`Memory Footprint (VmRSS) (KB)`, `Peak Memory Usage Resident Set Size (VmHWM) (KB)`,
`Threads`, `Voluntary Context Switches` and `Non-voluntary Context Switches`.
Equivalent to `TAU_TRACK_MEMORY_FOOTPRINT_HERE` in C.
"""
function tau_track_memory_footprint_here()
    _tau_active() || return nothing
    _pin_task!()
    ccall((:Tau_track_memory_rss_and_hwm_here, _libTAU[]), Cvoid, ())
    nothing
end

"""
    tau_track_memory_headroom_here()

Record how much more memory the process could allocate at this point as the
context event `Memory Headroom Left (MB)`. Equivalent to
`TAU_TRACK_MEMORY_HEADROOM_HERE` in C.
"""
function tau_track_memory_headroom_here()
    _tau_active() || return nothing
    _pin_task!()
    ccall((:Tau_track_memory_headroom_here, _libTAU[]), Cvoid, ())
    nothing
end

"""
    tau_enable_tracking_memory()
    tau_disable_tracking_memory()

Turn heap memory tracking on or off. While it is off,
[`tau_track_memory_here`](@ref) records nothing. It is on by default.
Equivalent to `TAU_ENABLE_TRACKING_MEMORY` and `TAU_DISABLE_TRACKING_MEMORY`
in C.
"""
function tau_enable_tracking_memory()
    _tau_active() || return nothing
    ccall((:Tau_enable_tracking_memory, _libTAU[]), Cvoid, ())
    nothing
end

@doc (@doc tau_enable_tracking_memory)
function tau_disable_tracking_memory()
    _tau_active() || return nothing
    ccall((:Tau_disable_tracking_memory, _libTAU[]), Cvoid, ())
    nothing
end

# ============================================================================
# Timer stack queries
# ============================================================================

# Innermost open timer on the calling thread; C_NULL when no timer is open.
_current_event() = ccall((:Tau_query_current_event, _libTAU[]), Ptr{Cvoid}, ())

# Name of the timer behind an event handle; `nothing` for C_NULL.
function _event_name(event::Ptr{Cvoid})
    event == C_NULL && return nothing
    s = ccall((:Tau_query_event_name, _libTAU[]), Ptr{Cchar}, (Ptr{Cvoid},), event)
    return s == C_NULL ? nothing : unsafe_string(s)
end

"""
    tau_current_timer_name() -> Union{String, Nothing}

Name of the innermost timer open on the calling thread. At the top level of
a script this is TAU's `.TAU application` timer, or `taupreload_main` under
`tau_julia`. Returns `nothing` when no timer is open or libTAU is not loaded.
Equivalent to `TAU_QUERY_GET_CURRENT_EVENT` followed by `TAU_QUERY_GET_EVENT_NAME` in C.
"""
function tau_current_timer_name()
    _tau_active() || return nothing
    _pin_task!()
    return _event_name(_current_event())
end

"""
    tau_parent_timer_name() -> Union{String, Nothing}

Name of the timer enclosing the innermost one open on the calling thread.
Returns `nothing` when the current timer is the outermost one
(`.TAU application`), when no timer is open, or when libTAU is not loaded.
Equivalent to `TAU_QUERY_GET_PARENT_EVENT` followed by `TAU_QUERY_GET_EVENT_NAME` in C.
"""
function tau_parent_timer_name()
    _tau_active() || return nothing
    _pin_task!()
    cur = _current_event()
    cur == C_NULL && return nothing
    parent = ccall((:Tau_query_parent_event, _libTAU[]), Ptr{Cvoid}, (Ptr{Cvoid},), cur)
    return _event_name(parent)
end
