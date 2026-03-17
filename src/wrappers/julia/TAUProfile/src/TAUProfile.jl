module TAUProfile

export tau_start, tau_stop, @tau, @tau_func, tau_rewrite, @tau_rewrite,
       set_rewrite_recursion_limit, set_rewrite_variant_limit, set_rewrite_ccall,
       rewrite_exclude_function, rewrite_exclude_module, rewrite_reset_exclusions

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

    # TODO remove forcing task sticky when we can handle task migration ~nchaimov
    current_task().sticky = true # workaround for lack of handling of task migration
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

# Trace hooks:
#   _entry_hook is called on function entry
#   _exit_hook is called on function exit by normal return or exception

"""
    _entry_hook(fname::String)

Called at function entry. 
"""
function _entry_hook(fname::String)
    tau_start(fname)
end

"""
    _exit_hook(fname::String)

Called at normal function exit or exit via exception.
"""
function _exit_hook(fname::String)
    tau_stop(fname)
end


"""
    LazyTraced(f)

A mutable callable wrapper used to break circular dependencies during
instrumentation of self-recursive or mutually-recursive functions.

"""
mutable struct LazyTraced
    target::Any
end
(lt::LazyTraced)(args...) = lt.target(args...)

"""
    _wrap_with_hooks(oc, fname::String) -> callable

Wrap an instrumented OpaqueClosure with entry/exit tracing.
"""
function _wrap_with_hooks(oc, fname::String)
    return (args...) -> begin
        _entry_hook(fname)
        try
            return oc(args...)
        finally
            _exit_hook(fname)
        end
    end
end


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
    # Accessors
    :getproperty, :getindex, :setproperty!, :setindex!
])

"""User-specified function objects to exclude from tracing."""
const USER_BLACKLIST_FUNCS = Set{Any}()

"""User-specified function names (as Symbols) to exclude from tracing."""
const USER_BLACKLIST_NAMES = Set{Symbol}()

"""User-specified modules to exclude from tracing."""
const USER_BLACKLIST_MODULES = Set{Module}()

"""User-specified module names (as Symbols) to exclude from tracing."""
const USER_BLACKLIST_MODULE_NAMES = Set{Symbol}()

"""Per-module recursion depth limits (by Module object). Value is (limit, exact)."""
const _module_depth_limits = Dict{Module, Tuple{Int, Bool}}()

"""Per-module recursion depth limits (by module name as Symbol). Value is (limit, exact)."""
const _module_depth_name_limits = Dict{Symbol, Tuple{Int, Bool}}()

"""
    rewrite_exclude_function(f)
    rewrite_exclude_function(name::Symbol)
    rewrite_exclude_function(name::String)

Exclude a function from being instrumented by `tau_rewrite`.
Accepts a function object, a `Symbol` (e.g. `:add`), or a `String` (e.g. `"add"`).
"""
function rewrite_exclude_function(@nospecialize(f))
    push!(USER_BLACKLIST_FUNCS, f)
    return nothing
end

function rewrite_exclude_function(name::Symbol)
    push!(USER_BLACKLIST_NAMES, name)
    return nothing
end

function rewrite_exclude_function(name::String)
    push!(USER_BLACKLIST_NAMES, Symbol(name))
    return nothing
end

"""
    rewrite_exclude_module(m::Module)
    rewrite_exclude_module(name::Symbol)
    rewrite_exclude_module(name::String)

Exclude all functions in a module from being instrumented by `tau_rewrite`.
Accepts a `Module` object, a `Symbol` (e.g. `:Base`), or a `String` (e.g. `"Base"`).
"""
function rewrite_exclude_module(m::Module)
    push!(USER_BLACKLIST_MODULES, m)
    return nothing
end

function rewrite_exclude_module(name::Symbol)
    push!(USER_BLACKLIST_MODULE_NAMES, name)
    return nothing
end

function rewrite_exclude_module(name::String)
    push!(USER_BLACKLIST_MODULE_NAMES, Symbol(name))
    return nothing
end

"""
    rewrite_reset_exclusions()

Clear all user-specified exclusions set via `rewrite_exclude_function`, `rewrite_exclude_module`,
`set_rewrite_recursion_limit`, and `set_rewrite_ccall`.
"""
function rewrite_reset_exclusions()
    empty!(USER_BLACKLIST_FUNCS)
    empty!(USER_BLACKLIST_NAMES)
    empty!(USER_BLACKLIST_MODULES)
    empty!(USER_BLACKLIST_MODULE_NAMES)
    empty!(_module_depth_limits)
    empty!(_module_depth_name_limits)
    _rewrite_ccall_enabled[] = false
    _max_multi_results[] = 16
    return nothing
end

"""
Check if a function should be rewritten based on the blacklist.
"""
function _should_rewrite(@nospecialize(f))
    f isa Core.Builtin && return false
    f isa Core.IntrinsicFunction && return false
    f in USER_BLACKLIST_FUNCS && return false

    fname = try nameof(f) catch; return false end
    fname in BLACKLIST && return false
    fname in USER_BLACKLIST_NAMES && return false

    mod = try parentmodule(f) catch; return false end
    mod === Core && return false
    mod in USER_BLACKLIST_MODULES && return false
    nameof(mod) in USER_BLACKLIST_MODULE_NAMES && return false

    return true
end

"""
    _replace_self_argument!(ir::IRCode, f) -> IRCode

Replace all references to `Argument(1)` (the function/self slot) in the IR
with the literal function value `f` for compatibility with OpaqueClosure.

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
    _map_sparams_to_args(f, argtypes::Tuple) -> Vector{Union{Nothing, Tuple{Int,Int}}}

Map each static parameter (TypeVar) to the (argument_index, type_parameter_index)
where it first appears in the method signature.

"""
function _map_sparams_to_args(@nospecialize(f), @nospecialize(argtypes::Tuple))
    ms = try methods(f, Tuple{argtypes...}) catch; return nothing end
    isempty(ms) && return nothing
    sig = first(ms).sig

    typevars = TypeVar[]
    body = sig
    while body isa UnionAll
        push!(typevars, body.var)
        body = body.body
    end
    isempty(typevars) && return nothing

    mapping = Vector{Union{Nothing, Tuple{Int,Int}}}(nothing, length(typevars))

    for (arg_idx, param) in enumerate(body.parameters[2:end])
        if param isa TypeVar
            for (tv_idx, tv) in enumerate(typevars)
                if param === tv && mapping[tv_idx] === nothing
                    mapping[tv_idx] = (arg_idx, 0)
                end
            end
        elseif param isa DataType && !isempty(param.parameters)
            is_type_param = param.name === Type.body.name
            for (tp_idx, tp) in enumerate(param.parameters)
                for (tv_idx, tv) in enumerate(typevars)
                    if tp === tv && mapping[tv_idx] === nothing
                        if is_type_param
                            mapping[tv_idx] = (arg_idx, -1)  # -1 = arg is the type itself
                        else
                            mapping[tv_idx] = (arg_idx, tp_idx)
                        end
                    end
                end
            end
        end
    end

    return mapping
end

"""
    _insert_sparam_extraction!(ir::IRCode, mapping) -> Vector{Union{Nothing, SSAValue}}

Insert IR instructions that extract type parameters from arguments at runtime.
For each mapped sparam, inserts `typeof(arg).parameters[idx]` and returns the
SSA values.

"""
function _insert_sparam_extraction!(ir::IRCode, mapping::Vector{Union{Nothing, Tuple{Int,Int}}})
    ssa_values = Vector{Any}(nothing, length(mapping))

    typeof_cache = Dict{Int, SSAValue}()
    params_cache = Dict{Int, SSAValue}()

    for (sp_idx, entry) in enumerate(mapping)
        entry === nothing && continue
        arg_idx, tp_idx = entry

        if !haskey(typeof_cache, arg_idx)
            typeof_inst = NewInstruction(
                Expr(:call, GlobalRef(Core, :typeof), Argument(arg_idx + 1)),
                DataType)
            typeof_cache[arg_idx] = insert_node!(ir, 1, typeof_inst, false)
        end

        if tp_idx == -1
            ssa_values[sp_idx] = Argument(arg_idx + 1)
        elseif tp_idx == 0
            ssa_values[sp_idx] = typeof_cache[arg_idx]
        else
            if !haskey(params_cache, arg_idx)
                params_inst = NewInstruction(
                    Expr(:call, GlobalRef(Core, :getfield), typeof_cache[arg_idx], QuoteNode(:parameters)),
                    Core.SimpleVector)
                params_cache[arg_idx] = insert_node!(ir, 1, params_inst, false)
            end

            extract_inst = NewInstruction(
                Expr(:call, GlobalRef(Base, :getindex), params_cache[arg_idx], tp_idx),
                Any)
            ssa_values[sp_idx] = insert_node!(ir, 1, extract_inst, false)
        end
    end

    return ssa_values
end

"""
    _substitute_static_parameters!(ir::IRCode, f, argtypes::Tuple) -> Bool

Resolve `Expr(:static_parameter, N)` nodes in the IR by looking up concrete
sparam values from the MethodInstance. When concrete values are available,
substitutes them directly. When sparams are abstract TypeVars, attempts runtime
type parameter extraction by inserting IR that reads type parameters from
arguments at runtime.

"""
function _substitute_static_parameters!(ir::IRCode, @nospecialize(f), @nospecialize(argtypes::Tuple))
    mi = ir.debuginfo.def
    !(mi isa Core.MethodInstance) && return true
    sparam_vals = mi.sparam_vals

    # Check whether any IR statements reference static parameters
    has_sparam_nodes = any(i -> _has_static_parameter(ir.stmts[i][:stmt]), 1:length(ir.stmts))

    if !has_sparam_nodes
        return true
    end

    # Check whether any sparams are abstract (TypeVar)
    has_abstract_sparams = any(i -> sparam_vals[i] isa TypeVar, 1:length(sparam_vals))

    if !has_abstract_sparams
        # All sparams are concrete — substitute directly
        for i in 1:length(ir.stmts)
            if _has_static_parameter(ir.stmts[i][:stmt])
                ir.stmts[i][:stmt] = _subst_sparams(ir.stmts[i][:stmt], sparam_vals)
            end
        end
        return true
    end

    # Abstract sparams with :static_parameter nodes in IR — try runtime extraction.
    mapping = _map_sparams_to_args(f, argtypes)
    mapping === nothing && return false

    # Verify all sparams that are TypeVars AND referenced in IR can be mapped
    for i in 1:length(sparam_vals)
        if sparam_vals[i] isa TypeVar && mapping[i] === nothing
            @debug "Cannot map sparam $i to argument type parameter"
            return false
        end
    end

    # Insert runtime extraction IR
    ssa_values = _insert_sparam_extraction!(ir, mapping)
    for i in 1:length(ir.stmts)
        if _has_static_parameter(ir.stmts[i][:stmt])
            ir.stmts[i][:stmt] = _subst_sparams_mixed(ir.stmts[i][:stmt], sparam_vals, ssa_values)
        end
    end
    return true
end

"""Check recursively whether an IR node contains any :static_parameter Expr."""
function _has_static_parameter(@nospecialize(node))
    if node isa Expr
        node.head === :static_parameter && return true
        for arg in node.args
            _has_static_parameter(arg) && return true
        end
    end
    return false
end

"""Recursively substitute :static_parameter nodes with concrete values."""
function _subst_sparams(@nospecialize(node), sparam_vals::Core.SimpleVector)
    if node isa Expr
        if node.head === :static_parameter
            return sparam_vals[node.args[1]::Int]
        end
        for (j, arg) in enumerate(node.args)
            node.args[j] = _subst_sparams(arg, sparam_vals)
        end
    end
    return node
end

"""Recursively substitute :static_parameter nodes using a mix of concrete values and SSA values."""
function _subst_sparams_mixed(@nospecialize(node), sparam_vals::Core.SimpleVector,
                              ssa_values::Vector{Any})
    if node isa Expr
        if node.head === :static_parameter
            idx = node.args[1]::Int
            # Use concrete value if available, otherwise SSA value from runtime extraction
            if !(sparam_vals[idx] isa TypeVar)
                return sparam_vals[idx]
            else
                return ssa_values[idx]
            end
        end
        for (j, arg) in enumerate(node.args)
            node.args[j] = _subst_sparams_mixed(arg, sparam_vals, ssa_values)
        end
    end
    return node
end

"""Module prefix string for a function, e.g. `"Base."`. """
_mod_prefix(@nospecialize(f)) = try string(parentmodule(f)) * "." catch; "" end

"""
    _function_label(f, argtypes::Tuple) -> String

Build a TAU timer name for a function including its name, argument types,
and source location.
"""
function _function_label(@nospecialize(f), @nospecialize(argtypes::Tuple))
    # For kwcall, build a more readable label showing the target function
    if f === Core.kwcall && length(argtypes) >= 2
        target_type = argtypes[2]
        if isconcretetype(target_type) && target_type <: Function && isdefined(target_type, :instance)
            target_f = target_type.instance
            pos_types = argtypes[3:end]
            return _mod_prefix(target_f) * string(target_f) * "(" * join(string.(pos_types), ", ") * ")"
        end
    end

    mod_prefix = _mod_prefix(f)
    fname_str = string(f)
    # Transform `#fname#NNN` kwarg body names into `fname`
    m_kw = match(r"^#(.+)#\d+$", fname_str)
    if m_kw !== nothing
        fname_str = m_kw.captures[1]
    end
    name = mod_prefix * fname_str * "(" * join(string.(argtypes), ", ") * ")"
    ms = methods(f, Tuple{argtypes...})
    if length(ms) == 1
        m = only(ms)
        file = string(m.file)
        line = m.line
        return "$name [{$file} {$line}]"
    end
    return name
end

"""
    _ir_arg_type(ir::IRCode, arg) -> Type

Extract the widened concrete type of an IR argument.
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

"""
function _resolve_callee(ir::IRCode, @nospecialize(callee))
    if callee isa GlobalRef
        return try getfield(callee.mod, callee.name) catch; nothing end
    end
    # Literal function constant embedded in IR
    if callee isa Function
        return callee
    end
    if callee isa Argument || callee isa SSAValue
        T = _ir_arg_type(ir, callee)
        if isconcretetype(T) && T <: Function && isdefined(T, :instance)
            return T.instance
        end
    end
    return nothing
end

"""
    _detect_varargs(ir::IRCode, f, argtypes::Tuple) -> (is_varargs::Bool, n_fixed::Int)

Check whether the IR expects a packed varargs tuple as its last argument.
Returns `(true, n_fixed)` if so, where `n_fixed` is the number of non-varargs parameters.

"""
function _detect_varargs(ir::IRCode, @nospecialize(f), @nospecialize(argtypes::Tuple))
    n_ir_args = length(ir.argtypes) - 1

    ms = try methods(f, Tuple{argtypes...}) catch; nothing end
    if ms !== nothing && length(ms) == 1
        is_varargs = only(ms).isva
        return (is_varargs, n_ir_args - (is_varargs ? 1 : 0))
    end

    caller_sig = Tuple{argtypes...}
    oc_sig = Tuple{(Core.Compiler.widenconst.(ir.argtypes[2:end]))...}
    last_oc_type = n_ir_args > 0 ? Core.Compiler.widenconst(ir.argtypes[end]) : Nothing
    is_varargs = !(caller_sig <: oc_sig) && last_oc_type <: Tuple
    return (is_varargs, n_ir_args - 1)
end

"""
    _max_depth_for(f, depth::Int, global_max::Int) -> Int

Resolve the effective max recursion depth for function `f` based on per-module limits.
Per-module limits are relative. Falls back to `global_max` if no per-module limit is set.

"""
function _max_depth_for(@nospecialize(f), depth::Int, global_max::Int)
    mod = try parentmodule(f) catch; return global_max end
    current = mod
    while true
        if haskey(_module_depth_limits, current)
            limit, exact = _module_depth_limits[current]
            if current === mod || !exact
                return min(depth + limit, global_max)
            end
        end
        if haskey(_module_depth_name_limits, nameof(current))
            limit, exact = _module_depth_name_limits[nameof(current)]
            if current === mod || !exact
                return min(depth + limit, global_max)
            end
        end
        parent = parentmodule(current)
        parent === current && break
        current = parent
    end
    return global_max
end

"""
    _foreigncall_label(stmt::Expr) -> String

Extract a human-readable label from a :foreigncall expression.

"""
function _foreigncall_label(stmt::Expr)
    name = stmt.args[1]
    if name isa QuoteNode
        name = name.value
    end
    return "@ccall $name"
end


"""
    _handle_invoke!(ir, i, stmt, depth, max_depth, cache) -> Bool

Handle `Core.invoke(f, Tuple{T...}, args...)` produced by `@invoke`.

"""
function _handle_invoke!(ir::IRCode, i::Int, stmt::Expr,
                         depth::Int, max_depth::Int, cache::Dict{Any,Any})
    length(stmt.args) >= 4 || return false
    target_ref = stmt.args[2]
    target_f = _resolve_callee(ir, target_ref)
    target_f === nothing && return true
    _should_rewrite(target_f) || return true

    callee_max = _max_depth_for(target_f, depth, max_depth)
    depth < callee_max || return true

    invoke_call_argtypes = tuple([_ir_arg_type(ir, a) for a in stmt.args[4:end]]...)
    traced_target = try
        _tau_rewrite_recursive(target_f, invoke_call_argtypes, depth + 1, callee_max, cache)
    catch ex
        @warn "Could not trace Core.invoke target $target_f: $ex"
        return true
    end
    ir.stmts[i][:stmt] = Expr(:call, traced_target, stmt.args[4:end]...)
    return true
end

"""
    _handle_kwcall!(ir, i, stmt, depth, max_depth, cache) -> Bool

Handle `Core.kwcall(kwargs::NamedTuple, f, args...)` produced by keyword argument calls.

"""
function _handle_kwcall!(ir::IRCode, i::Int, stmt::Expr,
                         depth::Int, max_depth::Int, cache::Dict{Any,Any})
    length(stmt.args) >= 3 || return false

    target_ref = stmt.args[3]
    target_f = _resolve_callee(ir, target_ref)
    (target_f !== nothing && !_should_rewrite(target_f)) && return true

    callee_max = target_f !== nothing ? _max_depth_for(target_f, depth, max_depth) : max_depth
    depth < callee_max || return true

    kwcall_argtypes = tuple([_ir_arg_type(ir, a) for a in stmt.args[2:end]]...)
    traced_kwcall = try
        _tau_rewrite_recursive(Core.kwcall, kwcall_argtypes, depth + 1, callee_max, cache)
    catch ex
        @warn "Could not trace Core.kwcall: $ex"
        return true
    end
    stmt.args[1] = traced_kwcall
    return true
end

"""
    _handle_apply_iterate!(ir, i, stmt, depth, max_depth, cache) -> Bool

Handle `Core._apply_iterate(iterate, f, args...)` produced by splatting.

"""
function _handle_apply_iterate!(ir::IRCode, i::Int, stmt::Expr,
                                depth::Int, max_depth::Int, cache::Dict{Any,Any})
    length(stmt.args) >= 3 || return false
    target_ref = stmt.args[3]
    target_f = _resolve_callee(ir, target_ref)
    target_f === nothing && return true
    _should_rewrite(target_f) || return true

    callee_max = _max_depth_for(target_f, depth, max_depth)
    depth < callee_max || return true

    splat_arg_types = tuple([_ir_arg_type(ir, a) for a in stmt.args[4:end]]...)

    unpacked_types = Type[]
    for T in splat_arg_types
        if T <: Tuple
            for j in 1:fieldcount(T)
                push!(unpacked_types, fieldtype(T, j))
            end
        else
            return true
        end
    end
    call_argtypes = tuple(unpacked_types...)

    traced_target = try
        _tau_rewrite_recursive(target_f, call_argtypes, depth + 1, callee_max, cache)
    catch ex
        @warn "Could not trace _apply_iterate target $target_f: $ex"
        return true
    end
    stmt.args[3] = traced_target
    return true
end

"""
    _handle_invokelatest!(ir, i, stmt, depth, max_depth, cache) -> Bool

Handle `Core._call_latest(f, args...)` produced by `Base.invokelatest`.

"""
function _handle_invokelatest!(ir::IRCode, i::Int, stmt::Expr,
                               depth::Int, max_depth::Int, cache::Dict{Any,Any})
    length(stmt.args) >= 2 || return false
    target_ref = stmt.args[2]
    target_f = _resolve_callee(ir, target_ref)
    target_f === nothing && return true
    _should_rewrite(target_f) || return true

    callee_max = _max_depth_for(target_f, depth, max_depth)
    depth < callee_max || return true

    call_argtypes = tuple([_ir_arg_type(ir, a) for a in stmt.args[3:end]]...)
    traced_target = try
        _tau_rewrite_recursive(target_f, call_argtypes, depth + 1, callee_max, cache)
    catch ex
        @warn "Could not trace invokelatest target $target_f: $ex"
        return true
    end
    stmt.args[2] = traced_target
    return true
end

"""
    _handle_invoke_in_world!(ir, i, stmt, depth, max_depth, cache) -> Bool

Handle `Core.invoke_in_world(world, f, args...)` produced by `Base.invoke_in_world`.
"""
function _handle_invoke_in_world!(ir::IRCode, i::Int, stmt::Expr,
                                  depth::Int, max_depth::Int, cache::Dict{Any,Any})
    length(stmt.args) >= 3 || return false
    target_ref = stmt.args[3]
    target_f = _resolve_callee(ir, target_ref)
    target_f === nothing && return true
    _should_rewrite(target_f) || return true

    callee_max = _max_depth_for(target_f, depth, max_depth)
    depth < callee_max || return true

    call_argtypes = tuple([_ir_arg_type(ir, a) for a in stmt.args[4:end]]...)
    traced_target = try
        _tau_rewrite_recursive(target_f, call_argtypes, depth + 1, callee_max, cache)
    catch ex
        @warn "Could not trace invoke_in_world target $target_f: $ex"
        return true
    end
    stmt.args[3] = traced_target
    return true
end

"""
    _instrument_single_ir(f, ir::IRCode, argtypes::Tuple, depth::Int, max_depth::Int, cache::Dict{Any,Any}) -> callable or nothing

Instrument an obtained IRCode: walk callees and rewrite call sites, 
create OpaqueClosure, wrap with entry/exit hooks.

Returns the wrapped callable on success, or `nothing` on failure.
"""
function _instrument_single_ir(@nospecialize(f), ir::IRCode, @nospecialize(argtypes::Tuple),
                                depth::Int, max_depth::Int, cache::Dict{Any,Any})
    fname = _function_label(f, argtypes)

    _replace_self_argument!(ir, f)

    trace_ccall = _rewrite_ccall_enabled[]
    foreigncall_positions = trace_ccall ? Int[] : nothing
    for i in 1:length(ir.stmts)
        stmt = ir.stmts[i][:stmt]
        stmt isa Expr || continue

        if stmt.head === :call
            callee_ref = stmt.args[1]
            callee_f = _resolve_callee(ir, callee_ref)
            callee_f === nothing && continue

            # Special-case handlers for alternate types of function dispatch
            if callee_f === Core.invoke
                _handle_invoke!(ir, i, stmt, depth, max_depth, cache) && continue
            elseif callee_f === Core.kwcall
                _handle_kwcall!(ir, i, stmt, depth, max_depth, cache) && continue
            elseif callee_f === Core._apply_iterate
                _handle_apply_iterate!(ir, i, stmt, depth, max_depth, cache) && continue
            elseif callee_f === invokelatest
                _handle_invokelatest!(ir, i, stmt, depth, max_depth, cache) && continue
            elseif callee_f === Core.invoke_in_world
                _handle_invoke_in_world!(ir, i, stmt, depth, max_depth, cache) && continue
            end

            # Standard case of bare call.
            
            # Handle exclusions
            _should_rewrite(callee_f) || continue

            # Handle max recursion depth
            callee_max = _max_depth_for(callee_f, depth, max_depth)
            depth < callee_max || continue

            # Recurse into called function and repeat instrumentation.
            call_argtypes = tuple([_ir_arg_type(ir, a) for a in stmt.args[2:end]]...)
            traced_callee = try
                _tau_rewrite_recursive(callee_f, call_argtypes, depth + 1, callee_max, cache)
            catch ex
                @warn "Could not trace $callee_f: $ex"
                continue
            end
            stmt.args[1] = traced_callee
        elseif trace_ccall && stmt.head === :foreigncall
            # Collect foreigncall (@ccall) for later instrumentation
            push!(foreigncall_positions, i)
        end
    end

    # Insert entry/exit hooks around foreigncall sites
    if foreigncall_positions !== nothing
        for pos in reverse(foreigncall_positions)
            label = _foreigncall_label(ir.stmts[pos][:stmt])
            entry_call = Expr(:call, GlobalRef(TracingDemo, :_entry_hook), label)
            exit_call = Expr(:call, GlobalRef(TracingDemo, :_exit_hook), label)
            insert_node!(ir, pos, NewInstruction(entry_call, Nothing), false)
            insert_node!(ir, pos, NewInstruction(exit_call, Nothing), true)
        end
    end

    # Compact IR after callee rewriting
    ir = compact!(ir)

    # Fix argtypes[1] for OpaqueClosure
    ir.argtypes[1] = Tuple{}

    # Detect varargs
    is_varargs, n_fixed = _detect_varargs(ir, f, argtypes)

    # Substitute static_parameter references with concrete values from MethodInstance.
    # (Need to replace to prevent segfaults during OpaqueClosure codegen.)
    if !_substitute_static_parameters!(ir, f, argtypes)
        # If failed, just wrap with calls. Prevents recursive instrumentation.
        @debug "Wrapping $f with hooks only"
        return _wrap_with_hooks(f, fname)
    end

    # Re-compact if runtime sparam extraction inserted new nodes
    if !isempty(ir.new_nodes.stmts)
        ir = compact!(ir)
    end

    oc = try
        OpaqueClosure(ir)
    catch ex
        @warn "OpaqueClosure creation failed for $f: $ex"
        return nothing
    end
    wrapped = _wrap_with_hooks(oc, fname)

    # Build varargs shim
    if is_varargs
        wrapped = let w = wrapped, nf = n_fixed
            (args...) -> w(args[1:nf]..., args[nf+1:end])
        end
    end

    return wrapped
end

# Maximum number of IR results to handle before bailing out
const _max_multi_results = Ref{Int}(100)

"""Validate and normalize a non-negative limit: 0 means unlimited."""
function _normalize_limit(n::Int, name::String)
    n < 0 && throw(ArgumentError("$name must be non-negative, got $n"))
    return n == 0 ? typemax(Int) : n
end

"""
    set_rewrite_variant_limit(n::Int)

Set the maximum number of IR results (method specializations) to handle
per function before falling back to the uninstrumented version.

"""
function set_rewrite_variant_limit(n::Int)
    _max_multi_results[] = _normalize_limit(n, "variant limit")
end

"""
    _rewrite_multi_result(f, argtypes::Tuple, results::Vector, depth::Int, max_depth::Int, cache::Dict{Any,Any}) -> LazyTraced

Handle the case where `code_ircode` returns multiple results (one per matching
method). Instead of trying to enumerate concrete type alternatives, instruments
each returned IR directly. The returned function does type dispatch.

"""
function _rewrite_multi_result(@nospecialize(f), @nospecialize(argtypes::Tuple),
                              results::Vector, depth::Int, max_depth::Int,
                              cache::Dict{Any,Any})
    key = (f, argtypes)

    # Cache sentinel to prevent infinite recursion
    lazy = LazyTraced(f)
    cache[key] = lazy

    # To reduce overhead, optionally skip instrumentation of functions
    # with very many specializations.
    if length(results) > _max_multi_results[]
        @debug "Too many IR results ($(length(results))) for $f, skipping"
        return lazy
    end

    traced_pairs = Tuple{Tuple, Any}[]

    for (ir, rettype) in results
        # code_ircode returns Method objects for builtins/generated functions,
        # which we can't instrument
        ir isa IRCode || continue

        concrete_argtypes = Tuple(Core.Compiler.widenconst.(ir.argtypes[2:end]))

        # Skip if already rewritten
        concrete_key = (f, concrete_argtypes)
        if haskey(cache, concrete_key) && concrete_key != key
            cached = cache[concrete_key]
            push!(traced_pairs, (concrete_argtypes, cached))
            continue
        end

        wrapped = try
            _instrument_single_ir(f, ir, concrete_argtypes, depth, max_depth, cache)
        catch ex
            @warn "Multi-result: could not instrument $f with $concrete_argtypes: $ex"
            nothing
        end

        if wrapped !== nothing
            concrete_lazy = LazyTraced(wrapped)
            cache[concrete_key] = concrete_lazy
            push!(traced_pairs, (concrete_argtypes, concrete_lazy))
        end
    end

    if isempty(traced_pairs)
        return lazy
    end

    # Dispatch shim
    lazy.target = (args...) -> begin
        for (types, traced_f) in traced_pairs
            length(args) == length(types) || continue
            matched = true
            for i in 1:length(types)
                if !(args[i] isa types[i])
                    matched = false
                    break
                end
            end
            matched && return traced_f(args...)
        end
        return f(args...)
    end
    return lazy
end

# Maximum recursion depth for rewriting callees.
const _max_depth = Ref{Int}(20)

# Whether @ccall/:foreigncall should be instrumented.
const _rewrite_ccall_enabled = Ref{Bool}(false)

"""
    set_rewrite_recursion_limit(n::Int)

Set global recursion limit.

    set_rewrite_recursion_limit(m::Module, n::Int, exact::Bool=false)
    set_rewrite_recursion_limit(m::Symbol, n::Int, exact::Bool=false)
    set_rewrite_recursion_limit(m::String, n::Int, exact::Bool=false)

Set the maximum recursion depth for a specific module.
If `exact`, match module exactly; otherwise, also match submodules.

"""
function set_rewrite_recursion_limit(n::Int)
    _max_depth[] = _normalize_limit(n, "recursion limit")
end

function set_rewrite_recursion_limit(m::Module, n::Int; exact::Bool=false)
    _module_depth_limits[m] = (_normalize_limit(n, "recursion limit"), exact)
end

function set_rewrite_recursion_limit(name::Symbol, n::Int; exact::Bool=false)
    _module_depth_name_limits[name] = (_normalize_limit(n, "recursion limit"), exact)
end

function set_rewrite_recursion_limit(name::String, n::Int; exact::Bool=false)
    set_rewrite_recursion_limit(Symbol(name), n; exact)
end

"""
    set_rewrite_ccall(enabled::Bool)

Enable or disable tracing of `@ccall` / `foreigncall` invocations.

"""
function set_rewrite_ccall(enabled::Bool)
    _rewrite_ccall_enabled[] = enabled
    return nothing
end

"""
    tau_rewrite(f, argtypes::Tuple) -> callable

Recursively instrument `f` AND all functions it calls. 

"""
function tau_rewrite(@nospecialize(f), @nospecialize(argtypes::Tuple);
                                   max_depth::Int = _max_depth[])
    cache = Dict{Any, Any}()
    return _tau_rewrite_recursive(f, argtypes, 0, max_depth, cache)
end

"""
    @tau_rewrite f(args...)

Instruments `f` and all its callees with entry/exit tracing, then 
calls the traced version with the given arguments.

    @tau_rewrite composite(5.0, -6.0)

is equivalent to:

    let args = (5.0, -6.0)
        traced = tau_rewrite(composite, map(typeof, args))
        traced(args...)
    end
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

# Internal recursion helper function
function _tau_rewrite_recursive(@nospecialize(f), @nospecialize(argtypes::Tuple),
                           depth::Int, max_depth::Int, cache::Dict{Any,Any})
    key = (f, argtypes)
    haskey(cache, key) && return cache[key]

    # Get pre-inlining IR — calls are still visible as :call with GlobalRef
    results = Base.code_ircode(f, argtypes; optimize_until="compact 1")

    # Case where we got more than one IRCode
    if length(results) != 1
        return _rewrite_multi_result(f, argtypes, results, depth, max_depth, cache)
    end

    # Case where we got only one IRCode
    ir, rettype = results[1]

    # Per the docs for code_ircode: 
    #   "Return an array of pairs of `IRCode` and inferred return type
    #   if type inference succeeds. The `Method` is included instead
    #   of `IRCode` otherwise.
    # We can't instrument without the IRCode, so if we got a Method,
    # skip instrumenting.
    if !(ir isa IRCode)
        lazy = LazyTraced(f)
        cache[key] = lazy
        return lazy
    end

    lazy = LazyTraced(f)
    cache[key] = lazy

    wrapped = _instrument_single_ir(f, ir, argtypes, depth, max_depth, cache)
    if wrapped !== nothing
        lazy.target = wrapped
    end

    return lazy
end


end # module TAUProfile
