# ============================================================================
# api.jl — public API
#
# Part of the TAUProfile module (LLVM backend). 
# ============================================================================

# Manual instrumentation API

"""
    tau_start(name::String)

Start a TAU timer with the given name. No-op when libTAU is not loaded.
"""
function tau_start(name::String)
    if isempty(_libTAU[])
        return  # Silently skip if library not loaded
    end

    _pin_task!()
    ccall((:Tau_start, _libTAU[]), Cvoid, (Cstring,), name)
end

"""
    tau_stop(name::String)

Stop a TAU timer with the given name. No-op when libTAU is not loaded.
"""
function tau_stop(name::String)
    if isempty(_libTAU[])
        return  # Silently skip if library not loaded
    end

    ccall((:Tau_stop, _libTAU[]), Cvoid, (Cstring,), name)
end

"""
    @tau(name::String, expr)
    @tau(t::TauTimer, expr)

Wrap an expression with a TAU timer, given by name or as a [`TauTimer`](@ref)
handle, stopping it on normal and exceptional exit.

# Example
```julia
@tau "computation" begin
    result = expensive_computation()
end

t = TauTimer("computation")
@tau t begin
    result = expensive_computation()
end
```
"""
macro tau(name, expr)
    quote
        tau_start($(esc(name)))
        try
            $(esc(expr))
        finally
            tau_stop($(esc(name)))
        end
    end
end

"""
    @tau_func function_definition

Wrap a function definition's body in `@tau`, deriving the timer name from the
function name. Works with both long- and short-form definitions.

# Example
```julia
@tau_func function my_computation(x, y)
    return x * y + sqrt(x^2 + y^2)
end

@tau_func fast_add(x, y) = x + y
```
"""
macro tau_func(func_expr)
    if func_expr.head == :function || func_expr.head == :(=)
        # Extract function signature and body
        func_sig = func_expr.args[1]
        func_body = func_expr.args[2]

        # Get the function name
        # Handle: func_name(args...), Type.func_name(args...), or just func_name
        func_name = if func_sig isa Symbol
            string(func_sig)
        elseif func_sig.head == :call
            # Extract the function name from the call signature
            name_part = func_sig.args[1]
            if name_part isa Symbol
                string(name_part)
            elseif name_part.head == :(.)
                # Handle Type.method_name
                string(name_part.args[2].value)
            else
                "unknown"
            end
        elseif func_sig.head == :where
            # Handle parameterized functions: func_name(args...) where {T}
            inner_sig = func_sig.args[1]
            if inner_sig.head == :call
                string(inner_sig.args[1])
            else
                "unknown"
            end
        else
            "unknown"
        end

        # Wrap the body with @tau
        new_body = quote
            @tau $func_name begin
                $(func_body)
            end
        end

        # Return the modified function
        return esc(Expr(func_expr.head, func_sig, new_body))
    else
        error("@tau_func must be used with a function definition")
    end
end

# Unimplemented parts of the AbstractInterpreter instrumentor API
const _warned_deferred = Ref(false)
const _warned_min_complexity = Ref(false)

function tau_rewrite_deferred_contexts(enabled::Bool=true)
    if !_warned_deferred[]
        @warn "tau_rewrite_deferred_contexts is not implemented, and not needed, with the LLVM backend " *
              "as dynamically dispatched functions will be automatically instrumented"
        _warned_deferred[] = true
    end
    nothing
end

function tau_rewrite_set_min_complexity(n::Int; skip_loops::Bool=true)
    if !_warned_min_complexity[]
        @warn "The complexity filter is not implemented with the LLVM backend"
        _warned_min_complexity[] = true
    end
    nothing
end

# ============================================================================
# Public API
# ============================================================================

"""
    trace_code(f, args...) -> String

Compile `f` with LLVM-level instrumentation and return the LLVM IR as a string.
"""
function trace_code(@nospecialize(f), args...)
    tt = Tuple{map(typeof, args)...}
    ir, _ = _compile_traced(f, tt)
    return string(ir)
end

"""
    tau_rewrite(f, argtypes::Tuple) -> callable

Return a callable wrapper that traces `f` via the LLVM backend.
"""
function tau_rewrite(@nospecialize(f), @nospecialize(argtypes::Tuple))
    return (args...) -> tau_rewrite_and_call(f, args...)
end

"""
    tau_rewrite_and_call(f, args...) -> result

Compile `f` with LLVM-level instrumentation, JIT-compile the instrumented module,
and execute it.
"""
function tau_rewrite_and_call(@nospecialize(f), args...)
    # Time the rewrite (compile + instrument + JIT) as its own TAU timer.
    do_time = _time_phase1[]
    rewrite_stopped = Ref(false)
    stop_rewrite!() = (do_time && !rewrite_stopped[] &&
                       (_tau_stop_rewrite(); rewrite_stopped[] = true); nothing)

    do_time && _tau_start_rewrite()
    try
        tt = Tuple{map(typeof, args)...}
        mi = GPUCompiler.methodinstance(typeof(f), tt, Base.get_world_counter())
        target = GPUCompiler.NativeCompilerTarget(; jlruntime=true)
        params = TracingPluginParams()
        config = GPUCompiler.CompilerConfig(target, params; kernel=false, entry_abi=:func, validate=false)
        job = GPUCompiler.CompilerJob(mi, config)

        return GPUCompiler.JuliaContext() do ctx
            ir, meta = GPUCompiler.compile(:llvm, job)

            entry_fn = meta.entry
            func_name = LLVM.name(entry_fn)
            LLVM.linkage!(entry_fn, LLVM.API.LLVMExternalLinkage)

            fptr = _jit_and_lookup(LLVM.ThreadSafeModule(ir), func_name)

            _install_phase2!()
            # Phase 1 (rewrite + compile + JIT) is complete; stop its timer
            stop_rewrite!()
            try
                args_array = Any[args...]
                GC.@preserve args_array begin
                    result = ccall(fptr, Any, (Any, Ptr{Any}, Int32),
                                   f, args_array, Int32(length(args_array)))
                end
                return result
            finally
                _uninstall_phase2!()
            end
        end
    finally
        stop_rewrite!()
    end
end

"""
    @tau_rewrite f(args...)

Convenience macro that instruments `f` and all its callees with entry/exit tracing,
then immediately calls the traced version. Equivalent to `tau_rewrite_and_call(f, args...)`.
"""
macro tau_rewrite(call)
    if !Meta.isexpr(call, :call) || length(call.args) < 1
        error("@tau_rewrite expects a function call, e.g. @tau_rewrite f(x, y)")
    end
    func = call.args[1]
    args = call.args[2:end]
    return esc(:(tau_rewrite_and_call($func, $(args...))))
end
