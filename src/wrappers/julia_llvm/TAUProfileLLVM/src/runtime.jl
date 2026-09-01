# ============================================================================
# runtime.jl — libTAU binding, hook functions, task-layout offsets, utility timers
#
# Part of the TAUProfile module (LLVM backend). Included from TAUProfile.jl;
# see that file for the license and the include order.
# ============================================================================

# ============================================================================
# Hook functions — called from LLVM-generated code via embedded function pointers
# ============================================================================

"""
    _write_stderr(msg::String)

Write a message to stderr via raw ccall to avoid yielding to other tasks.
"""
function _write_stderr(msg::String)
    buf = Vector{UInt8}(msg)
    ccall(:write, Cssize_t, (Cint, Ptr{UInt8}, Csize_t), 2, buf, length(buf))
    nothing
end

"""
    _entry_hook(fname_ptr::Ptr{UInt8})::Cvoid

C-callable entry hook. Receives a pointer to a null-terminated function label string.
"""
function _entry_hook(fname_ptr::Ptr{UInt8})::Cvoid
    fname = unsafe_string(fname_ptr)
    _write_stderr(">>> Enter: $fname\n")
    return
end

"""
    _exit_hook(fname_ptr::Ptr{UInt8})::Cvoid

C-callable exit hook. Receives a pointer to a null-terminated function label string.
"""
function _exit_hook(fname_ptr::Ptr{UInt8})::Cvoid
    fname = unsafe_string(fname_ptr)
    _write_stderr("<<< Exit:  $fname\n")
    return
end

# C-callable function pointers, initialized in __init__
const _entry_hook_ptr = Ref{Ptr{Cvoid}}(C_NULL)
const _exit_hook_ptr = Ref{Ptr{Cvoid}}(C_NULL)

# TAU runtime state 

const _libTAU              = Ref{String}("")
const _JULIA_BLAS_LIB      = Ref{String}("")
const TAU_START_FPTR       = Ref{Ptr{Cvoid}}(C_NULL)
const TAU_STOP_FPTR        = Ref{Ptr{Cvoid}}(C_NULL)

const _TASK_OFFSET_FROM_PGCSTACK        = Ref{Int}(0)  # current_task = pgcstack + this
const _STICKY_BYTE_OFFSET_FROM_PGCSTACK = Ref{Int}(0)  # &task.sticky = pgcstack + this
const _offsets_ready = Ref(false)                      # true once __init__ computes offsets

const _force_text_hooks = Ref(false)

_tau_active() = !isempty(_libTAU[]) && TAU_START_FPTR[] != C_NULL && TAU_STOP_FPTR[] != C_NULL

function _active_hook_ptrs()
    if _tau_active() && !_force_text_hooks[]
        return (TAU_START_FPTR[], TAU_STOP_FPTR[], true)   # tau_mode = true
    else
        return (_entry_hook_ptr[], _exit_hook_ptr[], false)
    end
end

function _install_julia_blas_hook(tau_lib_path::String)
    if get(ENV, "TAU_BLAS_HOOK", "1") == "0"
        return
    end
    blas_lib = get(ENV, "TAU_JULIA_BLAS_LIB", "")
    if isempty(blas_lib)
        blas_lib = joinpath(dirname(tau_lib_path), "libTAU-julia-blas.so")
    end
    if !isfile(blas_lib)
        @debug "TAU julia_blas shim not found at $blas_lib; skipping BLAS hook."
        return
    end
    _JULIA_BLAS_LIB[] = blas_lib
    try
        installed = ccall((:tau_lbt_install, _JULIA_BLAS_LIB[]), Cint, ())
        if installed < 0
            @warn "tau_lbt_install returned $installed; BLAS timers will not be captured."
        else
            @debug "Installed $installed BLAS interception slots via libblastrampoline."
        end
    catch err
        @warn "Failed to install TAU julia_blas hook" exception=(err, catch_backtrace())
    end
end

# Runtime half of module initialization: hook pointers, pgcstack/sticky
# offsets, and libTAU loading. Called from __init__ in TAUProfile.jl.
function _init_runtime!()
    _entry_hook_ptr[] = @cfunction(_entry_hook, Cvoid, (Ptr{UInt8},))
    _exit_hook_ptr[] = @cfunction(_exit_hook, Cvoid, (Ptr{UInt8},))

    sticky_idx = findfirst(==(:sticky), fieldnames(Task))
    sticky_idx === nothing && error("Task has no :sticky field on Julia $(VERSION); " *
        "TAUProfile LLVM sticky-pin needs updating to the Task layout.")
    let task = ccall(:jl_get_current_task, Any, ())::Task
        task_addr = Int(UInt(pointer_from_objref(task)))
        pgc_addr  = Int(UInt(ccall(:jl_get_pgcstack, Ptr{UInt8}, ())))
        _TASK_OFFSET_FROM_PGCSTACK[] = task_addr - pgc_addr
        _STICKY_BYTE_OFFSET_FROM_PGCSTACK[] = (task_addr - pgc_addr) + Int(fieldoffset(Task, sticky_idx))
        _offsets_ready[] = true
    end

    tau_lib = get(ENV, "TAU_JULIA_LIB", "")
    if isempty(tau_lib)
        @warn "TAU_JULIA_LIB not set; TAUProfile LLVM tracing falls back to stderr logging."
    elseif !isfile(tau_lib)
        @warn "TAU library not found at $tau_lib; falling back to stderr logging."
    else
        _libTAU[] = tau_lib
        ccall((:Tau_init_initializeTAU, _libTAU[]), Cint, ())
        ccall((:Tau_create_top_level_timer_if_necessary, _libTAU[]), Cvoid, ())
        # The ccall/cglobal library position must reference a global
        # These function pointers will be inserted into and called from instrumented LLVM code
        TAU_START_FPTR[] = cglobal((:Tau_start, _libTAU[]))
        TAU_STOP_FPTR[]  = cglobal((:Tau_stop,  _libTAU[]))
        _install_julia_blas_hook(tau_lib)
    end
end

# Label for the rewrite timer.
const _PHASE1_TIMER_NAME = ".TAU Julia Rewrite"

# Profile group for the rewrite timer.
const _PHASE1_TIMER_GROUP = "TAU_UTILITY"

# Cached FunctionInfo* for the Phase 1 rewrite timer
const _PHASE1_TIMER_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

# Get-or-create a TAU_UTILITY timer FunctionInfo for `name`.
function _ensure_util_timer!(handleref::Ref{Ptr{Cvoid}}, name::String)
    if handleref[] == C_NULL
        grp = ccall((:Tau_get_profile_group, _libTAU[]), Culong, (Cstring,),
                    _PHASE1_TIMER_GROUP)
        ccall((:Tau_profile_c_timer, _libTAU[]), Cvoid,
              (Ptr{Ptr{Cvoid}}, Cstring, Cstring, Culong, Cstring),
              handleref, name, "", grp, _PHASE1_TIMER_GROUP)
    end
    return handleref[]
end

# Start a cached TAU_UTILITY timer on the (pinned) current thread.
function _tau_start_util_timer(handleref::Ref{Ptr{Cvoid}}, name::String)
    isempty(_libTAU[]) && return C_NULL
    current_task().sticky = true
    handle = _ensure_util_timer!(handleref, name)
    handle == C_NULL && return C_NULL
    tid = ccall((:Tau_get_thread, _libTAU[]), Cint, ())
    ccall((:Tau_start_timer, _libTAU[]), Cvoid, (Ptr{Cvoid}, Cint, Cint), handle, 0, tid)
    return handle
end

# Stop a TAU_UTILITY timer by handle on the current thread.
function _tau_stop_util_timer(handle::Ptr{Cvoid})
    (handle == C_NULL || isempty(_libTAU[])) && return
    tid = ccall((:Tau_get_thread, _libTAU[]), Cint, ())
    ccall((:Tau_stop_timer, _libTAU[]), Cvoid, (Ptr{Cvoid}, Cint), handle, tid)
    nothing
end

# --- Phase 1 rewrite timer ---
_tau_start_rewrite() = (_tau_start_util_timer(_PHASE1_TIMER_HANDLE, _PHASE1_TIMER_NAME); nothing)
_tau_stop_rewrite()  = _tau_stop_util_timer(_PHASE1_TIMER_HANDLE[])

# --- Phase 2 rewrite timer ---
const _PHASE2_TIMER_NAME = ".TAU Julia Phase 2 Rewrite"

# Cached FunctionInfo* for the :separate-mode Phase 2 timer.
const _PHASE2_TIMER_HANDLE = Ref{Ptr{Cvoid}}(C_NULL)

# Start the Phase 2 timer per _phase2_timing_mode.
function _tau_start_phase2()::Ptr{Cvoid}
    mode = _phase2_timing_mode[]
    mode === :off && return C_NULL
    return mode === :combined ?
        _tau_start_util_timer(_PHASE1_TIMER_HANDLE, _PHASE1_TIMER_NAME) :
        _tau_start_util_timer(_PHASE2_TIMER_HANDLE, _PHASE2_TIMER_NAME)
end

_tau_stop_phase2(handle::Ptr{Cvoid}) = _tau_stop_util_timer(handle)
