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
