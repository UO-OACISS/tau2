# tau_timeroutputs.jl — TimerOutputs.jl sections as TAU timers.
#
# Loaded with `julia -L tau_timeroutputs.jl ...`, which tau_exec and tau_julia 
# do automatically for Julia targets.
# 
# Every TimerOutputs macro opens a section with `push!(to, label)` and closes
# it with `pop!(to)`. When TimerOutputs is loaded, we replace those methods
# with copies that also start and stop a TAU timer named after the label.
#
# Environment:
#   TAU_TIMEROUTPUTS_PREFIX    prepended to every TAU timer name
#   TAU_TIMEROUTPUTS_REPORT=1  at exit, print the merge of every TimerOutput seen to stderr
#   TAU_TIMEROUTPUTS_REPORT_EACH=1
#                              at exit, also print every TimerOutput seen, separately, in
#                              the order they were first used
#   TAU_TIMEROUTPUTS_ITERATE=a,b
#                              sections with these labels get one TAU timer per invocation
#   TAU_TIMEROUTPUTS_GCSTATS=1 at the end of each section, record Julia's allocation and GC counters
#                              for it as TAU context events
#   TAU_TIMEROUTPUTS_VERBOSE=1 verbose output; print whether TimerOutputs was hooked
#   TAU_JULIA_GC=1             time every garbage collection as the TAU timer "Julia GC"
#   TAU_JULIA_GC_LIB           libTAU-julia-gc to use instead
#   TAU_JULIA_LIB              libTAU to dlopen when none is preloaded
module TAUTimerOutputs

const TO_UUID = Base.UUID("a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f")
const VERBOSE = get(ENV, "TAU_TIMEROUTPUTS_VERBOSE", "0") != "0"
const REPORT_EACH = get(ENV, "TAU_TIMEROUTPUTS_REPORT_EACH", "0") != "0"
const REPORT  = REPORT_EACH || get(ENV, "TAU_TIMEROUTPUTS_REPORT", "0") != "0"
const PREFIX  = get(ENV, "TAU_TIMEROUTPUTS_PREFIX", "")
const ITERATE = Set{String}(filter(!isempty, String.(strip.(split(get(ENV, "TAU_TIMEROUTPUTS_ITERATE", ""), ',')))))
const GCSTATS = get(ENV, "TAU_TIMEROUTPUTS_GCSTATS", "0") != "0"
const JULIA_GC = get(ENV, "TAU_JULIA_GC", "0") != "0"
const GROUP   = "TimerOutputs"

_say(msg) = println(stderr, "tau_timeroutputs: ", msg)

const RTLD_DEFAULT = C_NULL
_dlsym(name) = ccall(:dlsym, Ptr{Cvoid}, (Ptr{Cvoid}, Cstring), RTLD_DEFAULT, name)

const opened = Ref(false)   # true when this file dlopen'd libTAU itself

function _resolve()
    if _dlsym("Tau_start_timer") == C_NULL
        lib = get(ENV, "TAU_JULIA_LIB", "")
        if !isempty(lib) && isfile(lib)
            Base.Libc.Libdl.dlopen(lib, Base.Libc.Libdl.RTLD_GLOBAL)
            opened[] = true
        end
    end
    names = ("Tau_profile_c_timer", "Tau_start_timer", "Tau_stop_timer",
             "Tau_get_thread", "Tau_get_profile_group",
             "Tau_init_initializeTAU", "Tau_create_top_level_timer_if_necessary",
             "Tau_get_node", "Tau_set_node")
    ptrs = map(_dlsym, names)
    all(!=(C_NULL), ptrs) || return nothing
    return ptrs
end

const _P = _resolve()
const active = _P !== nothing
const P_MK, P_START, P_STOP, P_TID, P_GRP, P_INIT, P_TOP, P_GETNODE, P_SETNODE =
    active ? _P : ntuple(_ -> C_NULL, 9)
const P_GETUE, P_UE = active ? (_dlsym("Tau_get_context_userevent"), _dlsym("Tau_context_userevent")) :
                               (C_NULL, C_NULL)

const grp     = Ref{Culong}(0)
const handles = Dict{String,Ptr{Cvoid}}()
const hlock   = ReentrantLock()
const stacks  = Vector{Vector{Ptr{Cvoid}}}()
const gcstacks = Vector{Vector{Base.GC_Num}}()   # Julia's GC counters at each open section's start

function _handle(label::String)
    @lock hlock get!(handles, label) do
        h = Ref{Ptr{Cvoid}}(C_NULL)
        ccall(P_MK, Cvoid, (Ptr{Ptr{Cvoid}}, Cstring, Cstring, Culong, Cstring),
              h, PREFIX * label, "", grp[], GROUP)
        h[]
    end
end

const iterations = Dict{String,Int}()

# "<label>[n]" for the n-th invocation of an ITERATE label.
@noinline function _iter_handle(label::String)
    n = @lock hlock (iterations[label] = get(iterations, label, 0) + 1)
    _handle(string(label, '[', n, ']'))
end

@noinline function _grow!(tid::Int)
    @lock hlock while length(stacks) < tid
        push!(stacks, Ptr{Cvoid}[])
        push!(gcstacks, Base.GC_Num[])
    end
    nothing
end

@inline function _stack()
    tid = Threads.threadid()
    tid > length(stacks) && _grow!(tid)
    @inbounds stacks[tid]
end

@inline _gcstack() = @inbounds gcstacks[Threads.threadid()]

# Julia's GC counters over one section.
# Base.gc_num() is process-wide and counts allocations across all threads.
# "malloc allocations" are the GC-managed buffers too large for Julia's pools.
const GCSTAT_NAMES = ("Julia allocated bytes", "Julia allocations", "Julia malloc allocations",
                      "Julia GC collections", "Julia GC full collections", "Julia GC time (s)")
_gcstat_values(d::Base.GC_Diff) =
    (Float64(d.allocd), Float64(Base.gc_alloc_count(d)), Float64(d.malloc + d.realloc),
     Float64(d.pause), Float64(d.full_sweep), 1.0e-9 * d.total_time)
const gcstat_events = fill(C_NULL, length(GCSTAT_NAMES))

@noinline function _gcstats_trigger()
    now = Base.gc_num()
    gst = _gcstack()
    isempty(gst) && return nothing
    vals = _gcstat_values(Base.GC_Diff(now, pop!(gst)))
    for i in 1:length(GCSTAT_NAMES)
        ccall(P_UE, Cvoid, (Ptr{Cvoid}, Cdouble), @inbounds(gcstat_events[i]), vals[i])
    end
    nothing
end

function on_push(label::String)
    active || return nothing
    current_task().sticky = true
    h = (isempty(ITERATE) || !(label in ITERATE)) ? _handle(label) : _iter_handle(label)
    push!(_stack(), h)
    ccall(P_START, Cvoid, (Ptr{Cvoid}, Cint, Cint), h, 0, ccall(P_TID, Cint, ()))
    GCSTATS && push!(_gcstack(), Base.gc_num())   # last, so the hook's own allocations are excluded
    nothing
end

function on_pop()
    active || return nothing
    st = _stack()
    isempty(st) && return nothing
    GCSTATS && _gcstats_trigger()   # before the stop, so the events' context is this section
    h = pop!(st)
    ccall(P_STOP, Cvoid, (Ptr{Cvoid}, Cint), h, ccall(P_TID, Cint, ()))
    nothing
end

const seen = WeakKeyDict{Any,Nothing}()
const seen_order = WeakRef[]     # first-use order, for REPORT_EACH
function _note(to)
    haskey(seen, to) && return nothing
    seen[to] = nothing
    REPORT_EACH && push!(seen_order, WeakRef(to))
    nothing
end

function _report(TO::Module)
    tos = collect(keys(seen))
    isempty(tos) && return
    try
        merged = copy(tos[1])
        for to in tos[2:end]
            TO.merge!(merged, to)
        end
        println(stderr, "tau_timeroutputs: merged TimerOutputs table ($(length(tos)) timer object(s)):")
        TO.print_timer(stderr, merged)
        println(stderr)
        if REPORT_EACH
            live = filter(!isnothing, [w.value for w in seen_order])
            for (k, to) in enumerate(live)
                println(stderr, "tau_timeroutputs: timer object $k of $(length(live)) (first-use order):")
                TO.print_timer(stderr, to)
                println(stderr)
            end
        end
    catch err
        _say("report failed: $err")
    end
end

const installed = Ref(false)

function install(TO::Module)
    installed[] && return
    v = pkgversion(TO)
    note = REPORT ? _note : Returns(nothing)
    if v === nothing
        _say("cannot determine the TimerOutputs version; sections will not be instrumented by TAU.")
        return
    elseif v.major == 1
        Core.eval(TO, quote
            function Base.push!(to::TimerOutput, label::String)
                $note(to)
                $on_push(label)
                section = child_section(current_section(to), label)
                push!(to.stack, section)
                return section
            end
            function Base.pop!(to::TimerOutput)
                $on_pop()
                return isempty(to.stack) ? nothing : pop!(to.stack)
            end
        end)
    elseif v.major == 0 && v.minor == 5
        Core.eval(TO, quote
            function Base.push!(to::TimerOutput, label::String)
                $note(to)
                $on_push(label)
                if length(to.timer_stack) == 0 # Root section
                    current_timer = to
                else # Not a root section
                    current_timer = to.timer_stack[end]
                end
                # Fast path
                if current_timer.prev_timer_label == label
                    timer = current_timer.prev_timer
                else
                    maybe_timer = get(current_timer.inner_timers, label, nothing)
                    if maybe_timer === nothing
                        timer = TimerOutput(label)
                        current_timer.inner_timers[label] = timer
                    else
                        timer = maybe_timer
                    end
                end
                timer = timer::TimerOutput
                current_timer.prev_timer_label = label
                current_timer.prev_timer = timer

                push!(to.timer_stack, timer)
                return timer.accumulated_data
            end
            function Base.pop!(to::TimerOutput)
                $on_pop()
                return pop!(to.timer_stack)
            end
        end)
    else
        _say("TimerOutputs $v is not a supported version (0.5.x or 1.x); sections will not be instrumented by TAU.")
        return
    end
    installed[] = true
    REPORT && atexit(() -> _report(TO))
    VERBOSE && _say("TimerOutputs $v: sections instrumented with TAU timers (group $GROUP" *
                    (isempty(PREFIX) ? "" : ", prefix \"$PREFIX\"") *
                    (isempty(ITERATE) ? "" : ", per-invocation timers for " * join(sort!(collect(ITERATE)), ", ")) *
                    (GCSTATS ? ", GC counters as context events" : "") * ")")
    nothing
end

function _on_package_loaded(id::Base.PkgId)
    id.uuid == TO_UUID || return nothing
    m = get(Base.loaded_modules, id, nothing)
    m === nothing || install(m)
    nothing
end

const gc_timed = Ref(false)
const P_GC_PRE = Ref(C_NULL)
const P_GC_POST = Ref(C_NULL)

struct DlInfo
    fname::Cstring
    fbase::Ptr{Cvoid}
    sname::Cstring
    saddr::Ptr{Cvoid}
end

function _gc_lib()
    lib = get(ENV, "TAU_JULIA_GC_LIB", "")
    isempty(lib) || return lib
    info = Ref{DlInfo}()
    ccall(:dladdr, Cint, (Ptr{Cvoid}, Ptr{DlInfo}), P_START, info) == 0 && return ""
    joinpath(dirname(unsafe_string(info[].fname)), "libTAU-julia-gc." * Base.Libc.Libdl.dlext)
end

# Registers Tau_julia_gc_pre/post from libTAU-julia-gc (src/wrappers/julia_gc) as Julia GC callbacks.
function _gc_install()
    lib = _gc_lib()
    h = isfile(lib) ? Base.Libc.Libdl.dlopen(lib; throw_error=false) : nothing
    if h === nothing
        _say("TAU_JULIA_GC=1, but libTAU-julia-gc was not found (\"$lib\"); collections will not be timed.")
        return nothing
    end
    init, pre, post = (Base.Libc.Libdl.dlsym(h, s) for s in (:Tau_julia_gc_init, :Tau_julia_gc_pre, :Tau_julia_gc_post))
    ccall(init, Cint, (Ptr{Cvoid},), cglobal(:jl_gc_live_bytes)) == 0 || return nothing
    P_GC_PRE[], P_GC_POST[] = pre, post
    ccall(:jl_gc_set_cb_pre_gc, Cvoid, (Ptr{Cvoid}, Cint), pre, 1)
    ccall(:jl_gc_set_cb_post_gc, Cvoid, (Ptr{Cvoid}, Cint), post, 1)
    gc_timed[] = true
    atexit(_gc_uninstall)
    VERBOSE && _say("garbage collections timed as the TAU timer \"Julia GC\"")
    nothing
end

function _gc_uninstall()
    gc_timed[] || return nothing
    ccall(:jl_gc_set_cb_pre_gc, Cvoid, (Ptr{Cvoid}, Cint), P_GC_PRE[], 0)
    ccall(:jl_gc_set_cb_post_gc, Cvoid, (Ptr{Cvoid}, Cint), P_GC_POST[], 0)
    gc_timed[] = false
    nothing
end

function __init__()
    if active
        ccall(P_INIT, Cint, ())
        ccall(P_TOP, Cvoid, ())
        grp[] = ccall(P_GRP, Culong, (Cstring,), GROUP)
        # TAU only writes profiles once a node id is set. A preloaded libTAU
        # gets it from tau_exec or the MPI wrapper; one we dlopen'd ourselves
        # has no wrapper layer and would never get one.
        if opened[] && ccall(P_GETNODE, Cint, ()) < 0
            ccall(P_SETNODE, Cvoid, (Cint,), 0)
        end
        if GCSTATS
            for (i, name) in enumerate(GCSTAT_NAMES)
                ue = Ref{Ptr{Cvoid}}(C_NULL)
                ccall(P_GETUE, Cvoid, (Ptr{Ptr{Cvoid}}, Cstring), ue, name)
                gcstat_events[i] = ue[]
            end
        end
        JULIA_GC && _gc_install()
    elseif VERBOSE
        _say("libTAU not found in this process; sections will not be instrumented by TAU..")
    end
    push!(Base.package_callbacks, _on_package_loaded)
    for (id, m) in Base.loaded_modules
        id.uuid == TO_UUID && install(m)
    end
    nothing
end

end # module
