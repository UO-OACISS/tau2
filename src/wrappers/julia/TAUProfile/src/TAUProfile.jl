module TAUProfile

export tau_start, tau_stop, @tau, @tau_func, tau_rewrite, @tau_rewrite

using Core: SSAValue, ReturnNode, Argument, OpaqueClosure
using Core.Compiler: IRCode, NewInstruction, insert_node!, compact!

# Path to libTAU.so
const libTAU = Ref{String}()

"""
    __init__()

Module initialization function. Reads the TAU_JULIA_LIB environment variable
to determine the path to libTAU.so, then initializes TAU and creates the
top-level timer.
"""
function __init__()
    tau_lib_path = get(ENV, "TAU_JULIA_LIB", "")

    if isempty(tau_lib_path)
        @warn "TAU_JULIA_LIB environment variable not set. TAU profiling will not be available."
        libTAU[] = ""
    elseif !isfile(tau_lib_path)
        @warn "TAU library not found at: $tau_lib_path. TAU profiling will not be available."
        libTAU[] = ""
    else
        libTAU[] = tau_lib_path
        init_result = ccall((:Tau_init_initializeTAU, libTAU[]), Cint, ())

        # Create top-level timer
        ccall((:Tau_create_top_level_timer_if_necessary, libTAU[]), Cvoid, ())
    end
end

"""
    tau_start(name::String)

Start a TAU timer with the given name.

# Arguments
- `name::String`: The name of the timer to start

# Example
```julia
tau_start("my_function")
```
"""
function tau_start(name::String)
    if isempty(libTAU[])
        return  # Silently skip if library not loaded
    end

    ccall((:Tau_start, libTAU[]), Cvoid, (Cstring,), name)
end

"""
    tau_stop(name::String)

Stop a TAU timer with the given name.

# Arguments
- `name::String`: The name of the timer to stop

# Example
```julia
tau_stop("my_function")
```
"""
function tau_stop(name::String)
    if isempty(libTAU[])
        return  # Silently skip if library not loaded
    end

    ccall((:Tau_stop, libTAU[]), Cvoid, (Cstring,), name)
end

"""
    @tau(name::String, expr)

Macro to automatically wrap an expression with TAU timing.

# Arguments
- `name::String`: The name for the TAU timer
- `expr`: The expression to time

# Example
```julia
@tau "computation" begin
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

Macro to automatically wrap a function definition with TAU timing.
The timer name is automatically derived from the function name.

# Example
```julia
@tau_func function my_computation(x, y)
    return x * y + sqrt(x^2 + y^2)
end

# Equivalent to:
function my_computation(x, y)
    @tau "my_computation" begin
        return x * y + sqrt(x^2 + y^2)
    end
end
```

Also works with short-form function definitions:
```julia
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

############################################################################################

# IR Rewriter


# ============================================================================
# Trace hooks — all tracing output funnels through these functions,
# making it easy to swap println for logging, profiling, etc.
# ============================================================================

"""
    _entry_hook(fname::String)

Called at function entry. Override or replace to customize tracing output.
"""
function _entry_hook(fname::String)
    tau_start(fname)
end

"""
    _exit_hook(fname::String)

Called at normal function exit (via ReturnNode).
"""
function _exit_hook(fname::String)
    tau_stop(fname)
end

"""
    _exit_hook(fname::String, exception::Bool)

Called at function exit via exception.
"""
function _exit_hook(fname::String, exception::Bool)
    tau_stop(fname)
end

"""
    LazyRewrittenCall(f)

When recursing downward through function calls, first replace
calls with calls to the original function, as otherwise we can
get stuck in an infinite loop when instrumenting recursive functions.
Target function is replaced with instrumented version when
returning from recursive instrumentation.
"""
mutable struct LazyRewrittenCall
    target::Any
end
(lt::LazyRewrittenCall)(args...) = lt.target(args...)

"""
    _wrap_for_exceptions(oc, fname::String) -> callable

Wraps OpaqueClosure in try/catch so that we can stop the timer
when we exit a function via an exception.
"""
function _wrap_for_exceptions(oc, fname::String)
    return (args...) -> begin
        try
            return oc(args...)
        catch
            _exit_hook(fname, true)
            rethrow()
        end
    end
end

"""
    _instrument_ir!(ir::IRCode, fname::String) -> IRCode

Modify `ir` in-place to add entry/exit tracing calls via `_entry_hook`
and `_exit_hook`. A call to `_entry_hook` is added as the first instruction,
and every return is instrumented with a call to `_exit_hook`.
"""
function _instrument_ir!(ir::IRCode, fname::String)
    _insert_hook!(ir, :_entry_hook, fname, SSAValue(1))

    # Iterate backwards so insertions don't shift later return positions
    for i in length(ir.stmts):-1:1
        if ir.stmts[i][:stmt] isa ReturnNode
            _insert_hook!(ir, :_exit_hook, fname, SSAValue(i))
        end
    end

    return compact!(ir)
end

"""
    _insert_hook!(ir::IRCode, hook::Symbol, fname::String, pos::SSAValue)

Insert call to a given symbol into the IR at a given position.
"""
function _insert_hook!(ir::IRCode, hook::Symbol, fname::String, pos::SSAValue)
    call = Expr(:call, GlobalRef(TAUProfile, hook), fname)
    insert_node!(ir, pos, NewInstruction(call, Nothing), false)
end

"""
Cache of already-instrumented functions to avoid re-instrumenting
the same function multiple times.
"""
const REWRITE_CACHE = Dict{Any, Any}()

# Blacklist of functions to not instrument. Avoid printing functions
# as we might print an error and can't println in a println.
# Avoid arithmetic operations as instrumentation overhead is high.
const BLACKLIST = Set{Symbol}([
    # I/O functions
    :println, :print, :string, :write, :error,
    # Our own instrumentation hooks
    :_entry_hook, :_exit_hook,
    # Arithmetic and comparison operators
    Symbol("+"), Symbol("-"), Symbol("*"), Symbol("/"),
    Symbol("<"), Symbol(">"), Symbol("=="), Symbol("<="), Symbol(">="),
    Symbol("!="), Symbol("≠"), Symbol("≤"), Symbol("≥"),
    Symbol("\\"), Symbol("÷"), Symbol("^"), Symbol("%"),
    Symbol("!"), Symbol("&&"), Symbol("||"), Symbol("~"),
    Symbol("&"), Symbol("|"), Symbol("⊻"), Symbol("⊼"),
    Symbol(">>>"), Symbol(">>"), Symbol("<<"), Symbol("="),
    Symbol("+="), Symbol("-="), Symbol("*="), Symbol("/="),
    Symbol("\\="), Symbol("÷="), Symbol("%="), Symbol("^="),
    Symbol("&="), Symbol("|="), Symbol("⊻="), Symbol(">>>="),
    Symbol(">>="), Symbol("<<="),
    :isequal, :isfinite, :isinf, :isnan,
    # Type conversion
    :convert, :promote, :cconvert, :unsafe_convert,
    # Array primitives
    :arrayref, :arrayset, :arraysize,
])

"""
    _should_rewrite(@nospecialize(f))  

Check if a function should be rewritten based on the blacklist.
Also avoid instrumenting Builtins and Intrinsics as they
don't have code to modify.
"""
function _should_rewrite(@nospecialize(f))
    f isa Core.Builtin && return false
    f isa Core.IntrinsicFunction && return false

    fname = try nameof(f) catch; return false end
    fname in BLACKLIST && return false

    mod = try parentmodule(f) catch; return false end
    mod === Core && return false

    return true
end

"""
    _replace_self_argument!(ir::IRCode, f) -> IRCode

Replace all references to `Argument(1)` (the function/self slot) in the IR
with the literal function value `f` so as to preserve the original argument
types of the function.
"""
function _replace_self_argument!(ir::IRCode, @nospecialize(f))
    for i in 1:length(ir.stmts)
        stmt = ir.stmts[i][:stmt]
        if stmt isa Expr
            for (j, arg) in enumerate(stmt.args)
                if arg isa Argument && arg.n == 1
                    stmt.args[j] = f
                end
            end
        end
    end
    return ir
end

"""
    _function_label(f, argtypes::Tuple) -> String

For a function `f` with argument types `argtypes`, construct a timer name
in TAU format: `funcname(arg1type, arg2type, ...) [{file} {line}]`.
"""            
function _function_label(@nospecialize(f), @nospecialize(argtypes::Tuple))
    # Special case for kwcall
    if f === Core.kwcall && length(argtypes) >= 2
        target_type = argtypes[2]
        if isconcretetype(target_type) && target_type <: Function && isdefined(target_type, :instance)
            target_f = target_type.instance
            pos_types = argtypes[3:end]
            name = string(target_f) * "(" * join(string.(pos_types), ", ") * ")"
            return name
        end
    end

    fname_str = string(f)

    # Special case for kwarg body: transform `#fname#NNN` into `fname`
    m_kw = match(r"^#(.+)#\d+$", fname_str)
    if m_kw !== nothing
        fname_str = m_kw.captures[1]
    end

    # Default case
    name = fname_str * "(" * join(string.(argtypes), ", ") * ")"

    # Get file name and line number if available
    ms = methods(f, Tuple{argtypes...})
    if length(ms) == 1
        m = only(ms)
        file = string(m.file)
        line = m.line
        return "$name [{$file} {$line}]"
    end

    # Otherwise return default without file/line
    return name
end

"""
    _ir_arg_type(ir::IRCode, arg) -> Type

Extract the widened concrete type of an IR argument (Argument, SSAValue, or literal).
"""
function _ir_arg_type(ir::IRCode, @nospecialize(arg))
    if arg isa Argument
        return Core.Compiler.widenconst(ir.argtypes[arg.n])
    elseif arg isa SSAValue
        return Core.Compiler.widenconst(ir.stmts[arg.id][:type])
    elseif arg isa QuoteNode
        return typeof(arg.value)
    elseif arg isa GlobalRef
        return typeof(getfield(arg.mod, arg.name))
    else
        return typeof(arg)
    end
end

"""
    _resolve_callee(ir::IRCode, callee) -> Union{Function, Nothing}

Try to resolve a call-site callee to a concrete function value.

Handles three cases:
  - `GlobalRef` — direct module lookup
  - `Argument` / `SSAValue` — if the IR type is a concrete singleton function
    type (e.g. `typeof(double)`), extract its `.instance`. This enables tracing
    of higher-order calls like `f(x)` where `f` is passed as an argument but
    the IR is specialized for a specific function.
"""
function _resolve_callee(ir::IRCode, @nospecialize(callee))
    if callee isa GlobalRef
        return try getfield(callee.mod, callee.name) catch; nothing end
    end
    if callee isa Argument || callee isa SSAValue
        T = _ir_arg_type(ir, callee)
        # Singleton function types have exactly one instance (e.g. typeof(double))
        if isconcretetype(T) && T <: Function && isdefined(T, :instance)
            return T.instance
        end
    end
    return nothing
end

"""
    tau_rewrite(f, argtypes::Tuple) -> callable

Recursively instrument function `f`. Uses partially-optimized, pre-inlining IR
(`optimize_until="compact 1"`) so that call sites are still visible, then
rewrites each rewritable call to use a recursively-rewritten OpaqueClosure.
"""

const DEFAULT_MAX_DEPTH = 20

function tau_rewrite(@nospecialize(f), @nospecialize(argtypes::Tuple);
                                   max_depth::Int = DEFAULT_MAX_DEPTH)
    empty!(REWRITE_CACHE)
    return _tau_rewrite_recursive(f, argtypes, 0, max_depth)
end

"""
    @tau_rewrite f(args...)

Convenience macro: instruments `f` and all its callees with entry/exit
tracing, then immediately calls the rewritten version with the given arguments.

    @tau_rewrite foo(5.0, -6.0)
"""
macro tau_rewrite(call_expr)
    Meta.isexpr(call_expr, :call) || error("@tau_rewrite expects a function call, got: $call_expr")
    f = call_expr.args[1]
    args = call_expr.args[2:end]
    return quote
        let _args = ($(esc.(args)...),)
            _rewritten = tau_rewrite($(esc(f)), map(typeof, _args))
            _rewritten(_args...)
        end
    end
end

function _tau_rewrite_recursive(@nospecialize(f), @nospecialize(argtypes::Tuple),
                           depth::Int, max_depth::Int)
    key = (f, argtypes)
    haskey(REWRITE_CACHE, key) && return REWRITE_CACHE[key]

    # Get pre-inlining IR ("compact 1") -- calls are still visible as :call with GlobalRef
    # If we fully optimize the code, some calls will have already been inlined and
    # we won't be able to instrument them.
    results = Base.code_ircode(f, argtypes; optimize_until="compact 1")
    if length(results) != 1
        error("Expected exactly 1 IRCode result for $f with argument types $argtypes, got $(length(results))")
    end
    ir, rettype = results[1]

    fname = _function_label(f, argtypes)

    _replace_self_argument!(ir, f)
    lazy = LazyRewrittenCall(f)  # initially calls uninstrumented function as fallback
    REWRITE_CACHE[key] = lazy

    # Walk IR and rewrite target call sites to use instrumented OpaqueClosures
    if depth < max_depth
        for i in 1:length(ir.stmts)
            stmt = ir.stmts[i][:stmt]
            if stmt isa Expr && stmt.head === :call
                callee_ref = stmt.args[1]
                callee_f = _resolve_callee(ir, callee_ref)
                # If we can't find the callee of the call, give up.
                callee_f === nothing && continue

                # Special case for instrumenting kwcalls. Allow instrumentation even though
                # it is in the Core module.
                if callee_f === Core.kwcall && length(stmt.args) >= 3
                    kwcall_argtypes = tuple([_ir_arg_type(ir, a) for a in stmt.args[2:end]]...)
                    rewritten_kwcall = try
                        _tau_rewrite_recursive(Core.kwcall, kwcall_argtypes, depth + 1, max_depth)
                    catch ex
                        @warn "Could not instrument Core.kwcall: $ex"
                        continue
                    end
                    stmt.args[1] = rewritten_kwcall
                    continue
                end

                _should_rewrite(callee_f) || continue

                # Determine callee argument types from IR type annotations
                call_argtypes = tuple([_ir_arg_type(ir, a) for a in stmt.args[2:end]]...)
                # Skip methods where we can't tell the callee's argtypes
                ms = try methods(callee_f, Tuple{call_argtypes...}) catch; continue end
                if length(ms) == 1 && only(ms).sig isa UnionAll
                    continue
                end
                # Recursively rewrite the callee (returns cached entry if already rewritten)
                rewritten_callee = try
                    _tau_rewrite_recursive(callee_f, call_argtypes, depth + 1, max_depth)
                catch ex
                    @warn "Could not instrument $callee_f: $ex"
                    continue
                end
                stmt.args[1] = rewritten_callee
            end
        end
    end

    # Instrument this method
    ir = _instrument_ir!(ir, fname)
    ir.argtypes[1] = Tuple{}

    # Special case for varargs
    n_ir_args = length(ir.argtypes) - 1   # subtract closure env slot
    caller_sig = Tuple{argtypes...}
    oc_sig = Tuple{(Core.Compiler.widenconst.(ir.argtypes[2:end]))...}
    last_oc_type = n_ir_args > 0 ? Core.Compiler.widenconst(ir.argtypes[end]) : Nothing
    is_varargs = !(caller_sig <: oc_sig) && last_oc_type <: Tuple

    oc = OpaqueClosure(ir)
    wrapped = _wrap_for_exceptions(oc, fname)

    # If varargs packing occurred, insert a shim that repacks the separate
    # caller args into the single tuple the OC expects.
    if is_varargs
        n_fixed = n_ir_args - 1
        wrapped = let w = wrapped, nf = n_fixed
            (args...) -> w(args[1:nf]..., args[nf+1:end])
        end
    end

    # Now that we've returned from the recursion and instrumented this function,
    # replace calls to this function with calls to the instrumented OpaqueClosure.
    lazy.target = wrapped

    return lazy
end

end # module TAUProfile
