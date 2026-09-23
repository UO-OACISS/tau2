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
#   TAU_TIMEROUTPUTS_VERBOSE=1 verbose output; print whether TimerOutputs was hooked
#   TAU_JULIA_LIB              libTAU to dlopen when none is preloaded
module TAUTimerOutputs

const TO_UUID = Base.UUID("a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f")
const VERBOSE = get(ENV, "TAU_TIMEROUTPUTS_VERBOSE", "0") != "0"
const REPORT  = get(ENV, "TAU_TIMEROUTPUTS_REPORT", "0") != "0"
const PREFIX  = get(ENV, "TAU_TIMEROUTPUTS_PREFIX", "")
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

const grp     = Ref{Culong}(0)
const handles = Dict{String,Ptr{Cvoid}}()
const hlock   = ReentrantLock()
const stacks  = Vector{Vector{Ptr{Cvoid}}}()

function _handle(label::String)
    @lock hlock get!(handles, label) do
        h = Ref{Ptr{Cvoid}}(C_NULL)
        ccall(P_MK, Cvoid, (Ptr{Ptr{Cvoid}}, Cstring, Cstring, Culong, Cstring),
              h, PREFIX * label, "", grp[], GROUP)
        h[]
    end
end

@noinline function _grow!(tid::Int)
    @lock hlock while length(stacks) < tid
        push!(stacks, Ptr{Cvoid}[])
    end
    nothing
end

@inline function _stack()
    tid = Threads.threadid()
    tid > length(stacks) && _grow!(tid)
    @inbounds stacks[tid]
end

function on_push(label::String)
    active || return nothing
    current_task().sticky = true
    h = _handle(label)
    push!(_stack(), h)
    ccall(P_START, Cvoid, (Ptr{Cvoid}, Cint, Cint), h, 0, ccall(P_TID, Cint, ()))
    nothing
end

function on_pop()
    active || return nothing
    st = _stack()
    isempty(st) && return nothing
    h = pop!(st)
    ccall(P_STOP, Cvoid, (Ptr{Cvoid}, Cint), h, ccall(P_TID, Cint, ()))
    nothing
end

const seen = WeakKeyDict{Any,Nothing}()
_note(to) = (haskey(seen, to) || (seen[to] = nothing); nothing)

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
                    (isempty(PREFIX) ? "" : ", prefix \"$PREFIX\"") * ")")
    nothing
end

function _on_package_loaded(id::Base.PkgId)
    id.uuid == TO_UUID || return nothing
    m = get(Base.loaded_modules, id, nothing)
    m === nothing || install(m)
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
