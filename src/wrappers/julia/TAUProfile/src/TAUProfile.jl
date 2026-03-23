module TAUProfile

export tau_start, tau_stop, @tau, @tau_func, tau_rewrite_and_call, tau_rewrite, @tau_rewrite,
       tau_rewrite_exclude_function, tau_rewrite_exclude_prefix, 
       tau_rewrite_exclude_module, tau_rewrite_reset_exclusions,
       tau_rewrite_set_recursion_limit, tau_rewrite_include_module_only,
       tau_rewrite_deferred_contexts, tau_rewrite_set_min_complexity,
       tau_rewrite_include_types

using Core: SSAValue, ReturnNode, CodeInstance, MethodInstance, EnterNode

const CC = Core.Compiler

###############################################################

# Base TAU profiling functionality

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

# Blacklisting

const BLACKLIST = Set{Symbol}([
    # I/O
    :println, :print, :string, :write, :error, :throw,
    :showerror, :show, :display, :repr,
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
    # Type conversion
    :convert, :promote, :cconvert, :unsafe_convert, :oftype,
    # Array primitives
    :arrayref, :arrayset, :arraysize, :arraylen,
    :getindex, :setindex!,
    # Memory/pointer
    :unsafe_load, :unsafe_store!, :pointer, :pointer_from_objref,
    # Core operations that appear frequently
    :iterate, :length, :size, :eltype, :similar,
    :copy, :copyto!, :axes, :checkbounds,
    # String operations (hooks use strings)
    :sizeof, :codeunit, :ncodeunits, :isvalid,
    # Hook functions themselves
    :_entry_hook, :_exit_hook,
])

const USER_BLACKLIST_FUNCS = Set{Any}()
const USER_BLACKLIST_NAMES = Set{Symbol}()
const USER_BLACKLIST_PREFIXES = Set{String}()

# Module exclusions: (Module_or_Symbol, exact::Bool)
const USER_BLACKLIST_MODULES = Dict{Module, Bool}()
const USER_BLACKLIST_MODULE_NAMES = Dict{Symbol, Bool}()

"""
    tau_rewrite_exclude_function(f)
    tau_rewrite_exclude_function(name::Symbol)
    tau_rewrite_exclude_function(name::String)
    tau_rewrite_exclude_function(items...)

Exclude function(s) from tracing. Accepts function objects, Symbols, or Strings.
"""
function tau_rewrite_exclude_function(@nospecialize(f::Function))
    push!(USER_BLACKLIST_FUNCS, f)
end
function tau_rewrite_exclude_function(name::Symbol)
    push!(USER_BLACKLIST_NAMES, name)
end
function tau_rewrite_exclude_function(name::String)
    push!(USER_BLACKLIST_NAMES, Symbol(name))
end
function tau_rewrite_exclude_function(items...)
    for item in items
        tau_rewrite_exclude_function(item)
    end
end

"""
    tau_rewrite_exclude_prefix(prefix::String)
    tau_rewrite_exclude_prefix(prefix::Symbol)
    tau_rewrite_exclude_prefix(prefixes...)

Exclude all functions whose name starts with `prefix` from tracing.
"""
function tau_rewrite_exclude_prefix(prefix::String)
    push!(USER_BLACKLIST_PREFIXES, prefix)
end
function tau_rewrite_exclude_prefix(prefix::Symbol)
    push!(USER_BLACKLIST_PREFIXES, String(prefix))
end
function tau_rewrite_exclude_prefix(prefixes...)
    for p in prefixes
        tau_rewrite_exclude_prefix(p)
    end
end

"""
    tau_rewrite_exclude_module(m::Module; exact=false)
    tau_rewrite_exclude_module(name::Symbol; exact=false)
    tau_rewrite_exclude_module(name::String; exact=false)

Exclude a module from tracing. By default, also excludes submodules.
Pass `exact=true` to only exclude the exact module.
"""
function tau_rewrite_exclude_module(m::Module; exact::Bool=false)
    USER_BLACKLIST_MODULES[m] = exact
end
function tau_rewrite_exclude_module(name::Symbol; exact::Bool=false)
    USER_BLACKLIST_MODULE_NAMES[name] = exact
end
function tau_rewrite_exclude_module(name::String; exact::Bool=false)
    USER_BLACKLIST_MODULE_NAMES[Symbol(name)] = exact
end
function tau_rewrite_exclude_module(items...; exact::Bool=false)
    for item in items
        tau_rewrite_exclude_module(item; exact)
    end
end

# Module whitelist: when non-empty, ONLY whitelisted modules are traced
# Module → exact_flag (same pattern as blacklist)
const USER_WHITELIST_MODULES = Dict{Module, Bool}()
const USER_WHITELIST_MODULE_NAMES = Dict{Symbol, Bool}()

"""
    tau_rewrite_include_module_only(m::Module; exact=false)
    tau_rewrite_include_module_only(name::Symbol; exact=false)
    tau_rewrite_include_module_only(name::String; exact=false)
    tau_rewrite_include_module_only(items...; exact=false)

Add module(s) to the whitelist. When the whitelist is non-empty, ONLY
functions in whitelisted modules are instrumented.

    tau_rewrite_include_module_only(MyModule)                  # MyModule + submodules
    tau_rewrite_include_module_only(MyModule; exact=true)      # MyModule only
    tau_rewrite_include_module_only(ModA, ModB)                 # both + submodules
    tau_rewrite_include_module_only(:ModA, "ModB")              # by name
"""
function tau_rewrite_include_module_only(m::Module; exact::Bool=false)
    USER_WHITELIST_MODULES[m] = exact
end
function tau_rewrite_include_module_only(name::Symbol; exact::Bool=false)
    USER_WHITELIST_MODULE_NAMES[name] = exact
end
function tau_rewrite_include_module_only(name::String; exact::Bool=false)
    tau_rewrite_include_module_only(Symbol(name); exact)
end
function tau_rewrite_include_module_only(items...; exact::Bool=false)
    for item in items
        tau_rewrite_include_module_only(item; exact)
    end
end

"""
    _is_module_in_set(mod::Module, by_mod::Dict{Module,Bool}, by_name::Dict{Symbol,Bool}) -> Bool

Check if a module matches an entry in the given module/name dicts.
"""
function _is_module_in_set(mod::Module, by_mod::Dict{Module,Bool}, by_name::Dict{Symbol,Bool})
    # Direct match (exact or not, the module itself always matches)
    haskey(by_mod, mod) && return true
    haskey(by_name, nameof(mod)) && return true

    # Walk parent modules for non-exact entries
    current = mod
    parent = parentmodule(current)
    while parent !== current
        if haskey(by_mod, parent)
            !by_mod[parent] && return true
        end
        pname = nameof(parent)
        if haskey(by_name, pname)
            !by_name[pname] && return true
        end
        current = parent
        parent = parentmodule(current)
    end
    return false
end

_is_module_whitelisted(mod::Module) = _is_module_in_set(mod, USER_WHITELIST_MODULES, USER_WHITELIST_MODULE_NAMES)


# Deferred-execution context tracing (Phase 2b): off by default
const _trace_deferred = Ref(false)

"""
    tau_rewrite_deferred_contexts(enabled::Bool=true)

Enable or disable tracing of deferred-execution contexts (`@spawn`, `@async`,
callbacks). Disabled by default.

    tau_rewrite_deferred_contexts()       # enable
    tau_rewrite_deferred_contexts(true)   # enable
    tau_rewrite_deferred_contexts(false)  # disable
"""
function tau_rewrite_deferred_contexts(enabled::Bool=true)
    _trace_deferred[] = enabled
end

"""
    tau_rewrite_reset_exclusions()

Reset exclusions and limits to default values.
"""
function tau_rewrite_reset_exclusions()
    empty!(USER_BLACKLIST_FUNCS)
    empty!(USER_BLACKLIST_NAMES)
    empty!(USER_BLACKLIST_PREFIXES)
    empty!(USER_BLACKLIST_MODULES)
    empty!(USER_BLACKLIST_MODULE_NAMES)
    empty!(USER_WHITELIST_MODULES)
    empty!(USER_WHITELIST_MODULE_NAMES)
    empty!(_module_depth_limits)
    empty!(_module_depth_name_limits)
    _max_depth[] = typemax(Int)
    _trace_deferred[] = false
    _min_complexity[] = 0
    _complexity_skip_loops[] = true
    _include_types[] = false
    _gpu_exclude_static[] = true
    _gpu_exclude_ci_owner[] = true
end

# Minimum complexity filter

# Minimum number of "interesting" IR statements for a function to be instrumented.
# 0 = disabled (instrument everything). 
const _min_complexity = Ref{Int}(0)
const _complexity_skip_loops = Ref{Bool}(true)

"""
    tau_rewrite_set_min_complexity(n::Int; skip_loops::Bool=true)

Set the minimum number of "interesting" IR statements a function must have to be
instrumented. Functions with fewer statements are skipped to reduce tracing overhead
for trivial functions. "Interesting" statements are calls, foreigncalls, invoke 
expressions, and new allocations.

    tau_rewrite_set_min_complexity(5)              # skip functions with < 5 interesting stmts
    tau_rewrite_set_min_complexity(10; skip_loops=false)  # apply even to functions with loops
    tau_rewrite_set_min_complexity(0)              # disable (default)
"""
function tau_rewrite_set_min_complexity(n::Int; skip_loops::Bool=true)
    n < 0 && throw(ArgumentError("min complexity must be non-negative, got $n"))
    _min_complexity[] = n
    _complexity_skip_loops[] = skip_loops
end

"""
    _count_complexity(stmts, n::Int) -> (interesting::Int, has_loop::Bool)

Count "interesting" IR statements (calls, foreigncalls, invokes, allocations).
"""
function _count_complexity(get_stmt, n::Int)
    has_loop = false
    interesting = 0
    check_loops = _complexity_skip_loops[]

    for i in 1:n
        stmt = get_stmt(i)

        if !has_loop && check_loops
            if stmt isa Core.GotoNode
                stmt.label <= i && (has_loop = true)
            elseif stmt isa Core.GotoIfNot
                stmt.dest <= i && (has_loop = true)
            end
        end

        if stmt isa Expr
            head = stmt.head
            if head === :call || head === :invoke || head === :foreigncall || head === :new || head === :splatnew
                interesting += 1
            end
        end
    end

    return interesting, has_loop
end

"""
    _is_complex_enough(ir::CC.IRCode) -> Bool
    _is_complex_enough(src::Core.CodeInfo) -> Bool

Check whether IR has enough complexity to warrant instrumentation.
"""
function _is_complex_enough(ir::CC.IRCode)
    min_c = _min_complexity[]
    min_c == 0 && return true
    interesting, has_loop = _count_complexity(i -> ir.stmts[i][:stmt], length(ir.stmts))
    (has_loop && _complexity_skip_loops[]) && return true
    return interesting >= min_c
end

function _is_complex_enough(src::Core.CodeInfo)
    min_c = _min_complexity[]
    min_c == 0 && return true
    interesting, has_loop = _count_complexity(i -> src.code[i], length(src.code))
    (has_loop && _complexity_skip_loops[]) && return true
    return interesting >= min_c
end

# Label formatting options

# When true, include argument types in labels: "Mod.f(Int64, String) [{file} {line}]"
# When false (default), omit types: "Mod.f [{file} {line}]"
const _include_types = Ref(false)

"""
    tau_rewrite_include_types(enabled::Bool=true)

Control whether argument types are included in timer names.

- `true`: labels show types, e.g. `Module.func(Int64, String) [{file} {line}]`
- `false` (default): labels omit types, e.g. `Module.func [{file} {line}]`
"""
function tau_rewrite_include_types(enabled::Bool=true)
    _include_types[] = enabled
end

# Per-module recursion depth limits

# Global depth limit
const _max_depth = Ref{Int}(typemax(Int))

# Per-module limits
const _module_depth_limits = Dict{Module, Tuple{Int, Bool}}()
const _module_depth_name_limits = Dict{Symbol, Tuple{Int, Bool}}()

function _normalize_limit(n::Int, name::String)
    n < 0 && throw(ArgumentError("$name must be non-negative, got $n"))
    return n == 0 ? typemax(Int) : n  # 0 means unlimited
end

"""
    tau_rewrite_set_recursion_limit(n::Int)
    tau_rewrite_set_recursion_limit(mod::Module, n::Int; exact=false)
    tau_rewrite_set_recursion_limit(name::Symbol, n::Int; exact=false)
    tau_rewrite_set_recursion_limit(name::String, n::Int; exact=false)

Set recursion depth limits for tracing. The global limit controls the maximum
depth from the root function. Per-module limits are relative.
Setting n=0 means unlimited. 

    tau_rewrite_set_recursion_limit(20)              # global limit
    tau_rewrite_set_recursion_limit(Base, 1)         # trace top-level Base calls only
    tau_rewrite_set_recursion_limit(:Base, 1)        # same, by name
    tau_rewrite_set_recursion_limit(Base, 1; exact=true)  # only Base, not submodules
"""
function tau_rewrite_set_recursion_limit(n::Int)
    _max_depth[] = _normalize_limit(n, "recursion limit")
end
function tau_rewrite_set_recursion_limit(mod::Module, n::Int; exact::Bool=false)
    _module_depth_limits[mod] = (_normalize_limit(n, "recursion limit"), exact)
end
function tau_rewrite_set_recursion_limit(name::Symbol, n::Int; exact::Bool=false)
    _module_depth_name_limits[name] = (_normalize_limit(n, "recursion limit"), exact)
end
function tau_rewrite_set_recursion_limit(name::String, n::Int; exact::Bool=false)
    tau_rewrite_set_recursion_limit(Symbol(name), n; exact)
end

"""
    _is_base_or_core(mod::Module) -> Bool

Check if a module is Base, Core, or a submodule of either.
"""
function _is_base_or_core(mod::Module)
    current = mod
    while true
        (current === Base || current === Core) && return true
        parent = parentmodule(current)
        parent === current && return false
        current = parent
    end
end

"""
Check if a function name matches any user-defined blacklist prefix.
"""
function _matches_blacklist_prefix(fname::Symbol)
    isempty(USER_BLACKLIST_PREFIXES) && return false
    fname_str = String(fname)
    for prefix in USER_BLACKLIST_PREFIXES
        startswith(fname_str, prefix) && return true
    end
    return false
end

_is_module_excluded(mod::Module) = _is_module_in_set(mod, USER_BLACKLIST_MODULES, USER_BLACKLIST_MODULE_NAMES)

# ============================================================================
# Filtering — which MethodInstances to instrument
# ============================================================================

"""
    _within_depth_limit(mi::MethodInstance, depth::Int, mod_depth::Int) -> Bool

Check whether a MethodInstance at the given depth should be instrumented
based on global and per-module depth limits.
"""
function _within_depth_limit(mi::MethodInstance, depth::Int, mod_depth::Int)
    global_max = _max_depth[]
    global_max == typemax(Int) && isempty(_module_depth_limits) && isempty(_module_depth_name_limits) && return true

    # Check global limit
    depth >= global_max && return false

    # Check per-module limit
    method = mi.def
    isa(method, Method) || return true
    mod = method.module

    if !isempty(_module_depth_limits) || !isempty(_module_depth_name_limits)
        mod_limit = _module_limit_for(mod)
        if mod_limit !== nothing
            return mod_depth < mod_limit
        end
    end

    return true
end

"""
    _module_limit_for(mod::Module) -> Union{Int, Nothing}

Look up the per-module depth limit for `mod`.
"""
function _module_limit_for(mod::Module)
    current = mod
    while true
        if haskey(_module_depth_limits, current)
            limit, exact = _module_depth_limits[current]
            if current === mod || !exact
                return limit
            end
        end
        if haskey(_module_depth_name_limits, nameof(current))
            limit, exact = _module_depth_name_limits[nameof(current)]
            if current === mod || !exact
                return limit
            end
        end
        parent = parentmodule(current)
        parent === current && break
        current = parent
    end
    return nothing
end

"""
    _passes_common_exclusions(mod::Module, fname::Symbol) -> Bool

Shared exclusion checks used by both `_should_instrument` and `_should_patch_method`.
Returns `true` if the function passes all exclusion filters.
"""
function _passes_common_exclusions(mod::Module, fname::Symbol)
    mod === Core && return false
    mod === TAUProfile && return false
    fname in BLACKLIST && return false
    fname in USER_BLACKLIST_NAMES && return false
    _matches_blacklist_prefix(fname) && return false
    _is_module_excluded(mod) && return false
    if !isempty(USER_WHITELIST_MODULES) || !isempty(USER_WHITELIST_MODULE_NAMES)
        _is_module_whitelisted(mod) || return false
    end
    return true
end

"""
    _should_instrument(mi::MethodInstance, depth::Int, mod_depth::Int) -> Bool

Determine whether a MethodInstance should have tracing hooks inserted.
"""
function _should_instrument(mi::MethodInstance, depth::Int=0, mod_depth::Int=0)
    method = mi.def
    isa(method, Method) || return false

    _passes_common_exclusions(method.module, method.name) || return false

    # Check user exclusions by function identity
    try
        f = _mi_function(mi)
        if f !== nothing && f in USER_BLACKLIST_FUNCS
            return false
        end
    catch
    end

    _within_depth_limit(mi, depth, mod_depth) || return false

    return true
end

"""
Extract the function object from a MethodInstance, if possible.
"""
function _mi_function(mi::MethodInstance)
    ft = mi.specTypes.parameters[1]
    if isdefined(ft, :instance)
        return ft.instance
    end
    return nothing
end

# Label building for Methods and MethodInstances

"""
    _clean_fname(name::Symbol) -> String

Clean up kwarg body names: `#fname#NNN` -> `fname`.
"""
function _clean_fname(name::Symbol)
    s = string(name)
    m_kw = match(r"^#(.+)#\d+$", s)
    m_kw !== nothing ? m_kw.captures[1] : s
end

"""
    _format_argtypes(argtypes; widen::Bool=false) -> String

Format a list of type parameters as a comma-separated string.
"""
function _format_argtypes(argtypes; widen::Bool=false)
    type_strs = String[]
    for i in 1:length(argtypes)
        if !isassigned(argtypes, i)
            push!(type_strs, "?")
            continue
        end
        t = argtypes[i]
        if t isa Core.TypeofVararg
            push!(type_strs, (isdefined(t, :T) ? string(t.T) : "Any") * "...")
        elseif t isa TypeVar
            push!(type_strs, string(t.ub))
        else
            # string(Type) can throw for pathological types (e.g. NamedTuple
            # with Vararg after unwrap_unionall leaves free TypeVars —
            # Base.show_at_namedtuple assumes more parameters than exist)
            s = try
                string(widen ? CC.widenconst(t) : t)
            catch
                try; repr(t); catch; "?"; end
            end
            push!(type_strs, s)
        end
    end
    return join(type_strs, ", ")
end

"""
    _build_label(method::Method, args_str::String) -> String

Build "Module.name(types) [{file} {line}]" from a Method and formatted arg string.
"""
function _build_label(method::Method, args_str::String)
    mod_str = string(method.module)
    fname_str = _clean_fname(method.name)
    file = string(method.file)
    line = method.line
    if _include_types[]
        return "$mod_str.$fname_str($args_str) [{$file} {$line}]"
    else
        return "$mod_str.$fname_str [{$file} {$line}]"
    end
end

"""
    _mi_label(mi::MethodInstance) -> String

Build a TAU timer name from a MethodInstance.
"""
function _mi_label(mi::MethodInstance)
    method = mi.def
    isa(method, Method) || return string(mi)
    spec = Base.unwrap_unionall(mi.specTypes)
    nparams = length(spec.parameters)
    if nparams >= 2
        argtypes = spec.parameters[2:end]
    elseif nparams == 1
        argtypes = spec.parameters
    else
        argtypes = Core.svec()
    end
    return _build_label(method, _format_argtypes(argtypes; widen=true))
end

# TracingInterpreter — custom AbstractInterpreter

struct TracingInterpreter <: CC.AbstractInterpreter
    native::CC.NativeInterpreter
    codegen::IdDict{CodeInstance, Core.CodeInfo}
    instrumented::Set{MethodInstance}
    depth_map::IdDict{MethodInstance, Int}       # absolute depth from root
    mod_depth_map::IdDict{MethodInstance, Int}    # depth within current module chain
end

function TracingInterpreter(world::UInt = Base.get_world_counter())
    native = CC.NativeInterpreter(world)
    codegen = IdDict{CodeInstance, Core.CodeInfo}()
    TracingInterpreter(native, codegen, Set{MethodInstance}(),
                       IdDict{MethodInstance, Int}(), IdDict{MethodInstance, Int}())
end

# Delegate compilation to default compiler

CC.InferenceParams(interp::TracingInterpreter) = CC.InferenceParams(interp.native)
CC.OptimizationParams(interp::TracingInterpreter) = CC.OptimizationParams(interp.native)
CC.get_inference_world(interp::TracingInterpreter) = CC.get_inference_world(interp.native)
CC.get_inference_cache(interp::TracingInterpreter) = CC.get_inference_cache(interp.native)

# We own our own code
CC.cache_owner(interp::TracingInterpreter) = interp
CC.codegen_cache(interp::TracingInterpreter) = interp.codegen

CC.method_table(interp::TracingInterpreter) = CC.method_table(interp.native)
CC.may_optimize(interp::TracingInterpreter) = true
CC.may_compress(interp::TracingInterpreter) = false  # don't compress -— we need readable CodeInfo
CC.may_discard_trees(interp::TracingInterpreter) = false  # keep trees for codegen

# Override typeinf_edge to track inference depth

"""
Override `typeinf_edge` to record the depth of each MethodInstance in the
call graph. 
"""
function CC.typeinf_edge(interp::TracingInterpreter, method::Method,
                         @nospecialize(atype), sparams::Core.SimpleVector,
                         caller::CC.AbsIntState, edgecycle::Bool, edgelimited::Bool)
    # Determine caller depth from our map
    caller_mi = CC.frame_instance(caller)
    caller_depth = get(interp.depth_map, caller_mi, 0)

    # Record callee depth
    callee_mi = CC.specialize_method(method, atype, sparams)
    if !haskey(interp.depth_map, callee_mi)
        interp.depth_map[callee_mi] = caller_depth + 1

        # Track module-relative depth
        caller_mod = try
            caller_def = caller_mi.def
            caller_def isa Method ? caller_def.module : nothing
        catch
            nothing
        end
        callee_mod = method.module
        if caller_mod !== nothing && caller_mod === callee_mod
            # Same module chain —- increment module-relative depth
            caller_mod_depth = get(interp.mod_depth_map, caller_mi, 0)
            interp.mod_depth_map[callee_mi] = caller_mod_depth + 1
        else
            # Entering a new module —- reset module depth to 0
            interp.mod_depth_map[callee_mi] = 0
        end
    end

    # Delegate to the standard AbstractInterpreter implementation
    return @invoke CC.typeinf_edge(interp::CC.AbstractInterpreter, method,
                                   atype::Any, sparams::Core.SimpleVector,
                                   caller::CC.AbsIntState, edgecycle::Bool,
                                   edgelimited::Bool)
end

# IR Hook Insertion

"""
    _push_ir_stmt!(ir, stmt, type=Nothing)

Append a statement to an IRCode's InstructionStream with default metadata.
"""
function _push_ir_stmt!(ir::CC.IRCode, @nospecialize(stmt), @nospecialize(type=Nothing))
    push!(ir.stmts.stmt, stmt)
    push!(ir.stmts.type, type)
    push!(ir.stmts.info, CC.NoCallInfo())
    append!(ir.stmts.line, Int32[0, 0, 0])
    push!(ir.stmts.flag, UInt32(0))
end

"""
    _insert_entry_exit_hooks!(ir::CC.IRCode, mi::MethodInstance) -> IRCode

Wrap the function body in try/finally so that `_entry_hook(label)` fires at
entry and `_exit_hook(label)` fires on both normal return and exception
propagation. 
"""
function _insert_entry_exit_hooks!(ir::CC.IRCode, mi::MethodInstance)
    label = _mi_label(mi)

    # Insert entry hook at function start
    entry_stmt = Expr(:call, GlobalRef(TAUProfile, :_entry_hook), label)
    CC.insert_node!(ir, SSAValue(1), CC.NewInstruction(entry_stmt, Nothing), false)

    # Insert exit hook before every return
    for i in length(ir.stmts):-1:1
        stmt = ir.stmts[i][:stmt]
        if stmt isa ReturnNode && isdefined(stmt, :val)
            exit_stmt = Expr(:call, GlobalRef(TAUProfile, :_exit_hook), label)
            CC.insert_node!(ir, SSAValue(i), CC.NewInstruction(exit_stmt, Nothing), false)
        end
    end

    ir = CC.compact!(ir)
    n_stmts = length(ir.stmts)

    # Wrap function in try/catch

    # Fake catch destination -- will be fixed up after we know the number
    SENTINEL_CATCH_DEST = 999_999_999

    # Insert catch enter after entry hook
    enter_node = EnterNode(SENTINEL_CATCH_DEST)
    CC.insert_node!(ir, SSAValue(1), CC.NewInstruction(enter_node, Any), true)

    # Insert catch leave before every return 
    for i in n_stmts:-1:1
        stmt = ir.stmts[i][:stmt]
        if stmt isa ReturnNode && isdefined(stmt, :val)
            # Use a sentinel SSAValue for the :leave arg — will fix after compact!
            leave_stmt = Expr(:leave, SSAValue(SENTINEL_CATCH_DEST))
            CC.insert_node!(ir, SSAValue(i), CC.NewInstruction(leave_stmt, Nothing), false)
        end
    end

    ir = CC.compact!(ir)

    # Get number of our catch enter node
    enter_ssa = nothing
    n_stmts = length(ir.stmts)
    for i in 1:n_stmts
        stmt = ir.stmts[i][:stmt]
        if stmt isa EnterNode && stmt.catch_dest == SENTINEL_CATCH_DEST
            enter_ssa = SSAValue(i)
            break
        end
    end

    if enter_ssa === nothing
        return ir
    end

    # Fix destination of leaves
    for i in 1:n_stmts
        stmt = ir.stmts[i][:stmt]
        if stmt isa Expr && stmt.head === :leave && length(stmt.args) == 1
            arg = stmt.args[1]
            if arg isa SSAValue && arg.id > n_stmts
                ir.stmts[i][:stmt] = Expr(:leave, enter_ssa)
            end
        end
    end

    # Find which block the EnterNode is in
    enter_pos = enter_ssa.id
    enter_block = 0
    for (bi, bb) in enumerate(ir.cfg.blocks)
        if bb.stmts.start <= enter_pos <= bb.stmts.stop
            enter_block = bi
            break
        end
    end

    # Append catch block: exit_hook, rethrow, return
    catch_start = n_stmts + 1
    catch_block_idx = length(ir.cfg.blocks) + 1

    # Fix EnterNode catch_dest to point to the new catch block
    ir.stmts[enter_pos][:stmt] = EnterNode(catch_block_idx)

    # Add catch body
    _push_ir_stmt!(ir, Expr(:call, GlobalRef(TAUProfile, :_exit_hook), label))
    _push_ir_stmt!(ir, Expr(:call, GlobalRef(Base, :rethrow)), Union{})
    _push_ir_stmt!(ir, ReturnNode(), Union{})
    append!(ir.debuginfo.codelocs, Int32[0, 0, 0, 0, 0, 0, 0, 0, 0])
    catch_bb = CC.BasicBlock(CC.StmtRange(catch_start, catch_start + 2),
                             Int[enter_block], Int[])
    push!(ir.cfg.blocks, catch_bb)
    push!(ir.cfg.index, catch_start)

    if enter_block > 0
        push!(ir.cfg.blocks[enter_block].succs, catch_block_idx)
    end

    return ir
end

# Override optimize to insert hooks

const _TRACE_DEBUG = Ref(false)

function CC.optimize(interp::TracingInterpreter, opt::CC.OptimizationState, caller::CC.InferenceResult)
    # Run the standard optimization pipeline
    ir = CC.run_passes_ipo_safe(opt.src, opt)
    CC.ipo_dataflow_analysis!(interp, opt, ir, caller)

    # Then instrument if neccessary
    mi = opt.linfo
    depth = get(interp.depth_map, mi, 0)
    mod_depth = get(interp.mod_depth_map, mi, 0)
    should = _should_instrument(mi, depth, mod_depth)
    if should
        should = _is_complex_enough(ir)
    end
    if _TRACE_DEBUG[]
        method = mi.def
        if isa(method, Method)
            label = "$(method.module).$(method.name)"
            println(stderr, "[DEBUG optimize] $label depth=$depth mod_depth=$mod_depth should_instrument=$should")
        end
    end
    if should
        try
            ir = _insert_entry_exit_hooks!(ir, mi)
            push!(interp.instrumented, mi)
        catch ex
            bt = catch_backtrace()
            @warn "TAUProfile: Failed to instrument $(mi.def): $ex" exception=(ex, bt)
        end
    end

    return CC.finish(interp, opt, ir, caller)
end

# Compilation driver

"""
    _trace_compile(f, tt::Type{<:Tuple}) -> CodeInstance

Compile function `f` with argument types `tt` through the TracingInterpreter.
All callees are recursively compiled (and instrumented) by the compiler.
Returns a CodeInstance that can be executed via `invoke(f, ci, args...)`.
"""
function _trace_compile(@nospecialize(f), @nospecialize(tt::Type))
    world = Base.get_world_counter()
    interp = TracingInterpreter(world)

    # Build the full signature: Tuple{typeof(f), argtypes...}
    ft = f isa Type ? Type{f} : typeof(f)
    sig = Tuple{ft, tt.parameters...}

    # Resolve to a MethodInstance
    matches = Base._methods_by_ftype(sig, -1, world)
    if matches === nothing || isempty(matches)
        error("No method found for $f with argument types $tt")
    end
    if length(matches) > 1
        @warn "Multiple methods match $f($tt), using most specific"
    end
    mi = CC.specialize_method(first(matches))

    # Entrypoint has depth 0
    interp.depth_map[mi] = 0

    # Compile through our interpreter
    ci = CC.typeinf_ext_toplevel(interp, mi, CC.SOURCE_MODE_ABI)

    if !(ci isa CodeInstance)
        error("Compilation failed for $f with argument types $tt (got $(typeof(ci)))")
    end

    # Patch method sources so that any NEW concrete specialization compiled
    # at runtime (via dynamic dispatch) also includes entry/exit hooks.
    patched = _patch_method_sources!(interp)

    return ci, patched
end

"""
    _patch_codeinfo_with_hooks(src::Core.CodeInfo, label::String) -> Core.CodeInfo

Insert entry/exit hook calls into a lowered CodeInfo, correctly renumbering
all SSAValue references, branch targets (GotoNode, GotoIfNot), and PhiNode
edges using Julia's built-in `renumber_ir_elements!`.
"""
function _patch_codeinfo_with_hooks(src::Core.CodeInfo, label::String)
    old_code = src.code
    n = length(old_code)

    # ssachangemap[i] = number of NEW statements inserted AT position i
    ssachangemap = zeros(Int, n)
    labelchangemap = zeros(Int, n)

    # 2 new stmts before position 1 (entry_hook + EnterNode)
    ssachangemap[1] += 2
    labelchangemap[1] += 2

    # Exit hooks + :leave: 2 new stmts before each ReturnNode
    n_returns = 0
    for i in 1:n
        if old_code[i] isa ReturnNode && isdefined(old_code[i], :val)
            ssachangemap[i] += 2
            labelchangemap[i] += 2
            n_returns += 1
        end
    end

    # Apply renumbering
    new_body = copy(old_code)
    CC.renumber_ir_elements!(new_body, ssachangemap, labelchangemap)

    # Build final code with try/finally wrapper
    enter_ssa = SSAValue(2) 
    # Catch block starts after: 2 (entry) + n (original) + 2*n_returns (exit/leave)
    catch_target = 2 + n + 2 * n_returns + 1
    total_stmts = 2 + n + 2 * n_returns + 3
    final_code = sizehint!(Any[], total_stmts)
    final_flags = sizehint!(UInt32[], total_stmts)

    emit!(stmt) = (push!(final_code, stmt); push!(final_flags, UInt32(0)))

    # Entry hook
    emit!(Expr(:call, GlobalRef(TAUProfile, :_entry_hook), label))

    # EnterNode (begin try scope)
    emit!(EnterNode(catch_target))

    # exit_hook + :leave before each ReturnNode
    return_positions = Int[]
    for i in 1:n
        stmt = old_code[i]
        if stmt isa ReturnNode && isdefined(stmt, :val)
            emit!(Expr(:call, GlobalRef(TAUProfile, :_exit_hook), label))
            emit!(Expr(:leave, enter_ssa))
        end
        emit!(new_body[i])
        if stmt isa ReturnNode && isdefined(stmt, :val)
            push!(return_positions, length(final_code))
        end
    end

    # Catch block: exit_hook, rethrow, return
    emit!(Expr(:call, GlobalRef(TAUProfile, :_exit_hook), label))
    emit!(Expr(:call, GlobalRef(Base, :rethrow)))
    emit!(ReturnNode())  # unreachable, structurally required

    # Rdirect gotos targeting a ReturnNode to the exit_hook before it
    return_set = Set(return_positions)
    for i in 1:length(final_code)
        stmt = final_code[i]
        if stmt isa Core.GotoNode && stmt.label in return_set
            final_code[i] = Core.GotoNode(stmt.label - 2)
        elseif stmt isa Core.GotoIfNot && stmt.dest in return_set
            final_code[i] = Core.GotoIfNot(stmt.cond, stmt.dest - 2)
        end
    end

    new_src = copy(src)
    new_src.code = final_code
    new_src.ssaflags = final_flags
    new_src.ssavaluetypes = length(final_code)
    return new_src
end

"""
    _method_label(method::Method) -> String

Build a TAU timer name from a Method object: "Module.name(types) [{file} {line}]"
"""
function _method_label(method::Method)
    sig = Base.unwrap_unionall(method.sig)
    argtypes = sig.parameters[2:end]
    return _build_label(method, _format_argtypes(argtypes))
end

# GPU exclusion 

const _gpu_exclude_static = Ref{Bool}(true)
const _gpu_exclude_ci_owner = Ref{Bool}(true)

"""
    _is_gpu_device_type(@nospecialize(T)) -> Bool

Check if a DataType is a known GPU device type. These types only appear in
GPU kernel signatures and indicate a method may be compiled for GPU execution.
"""
function _is_gpu_device_type(@nospecialize(T))
    T isa DataType || return false
    name_str = String(T.name.name)
    contains(name_str, "DeviceArray") && return true
    T.name.name === :CompilerMetadata && return true
    return false
end

"""
    _has_gpu_device_types(@nospecialize(T)) -> Bool

Recursively check if a type or any of its parameters contains a GPU device type.
"""
function _has_gpu_device_types(@nospecialize(T))
    if T isa UnionAll
        return _has_gpu_device_types(T.body)
    end
    if T isa Union
        return _has_gpu_device_types(T.a) || _has_gpu_device_types(T.b)
    end
    T isa DataType || return false
    _is_gpu_device_type(T) && return true
    for p in T.parameters
        p isa Type && _has_gpu_device_types(p) && return true
    end
    return false
end

"""
    _is_gpu_method(method::Method) -> Bool

Check if a method may be compiled for GPU execution by examining its signature
for GPU device types (e.g. CuDeviceArray, CompilerMetadata). Methods with GPU
device types should not have their source patched.
"""
function _is_gpu_method(method::Method)
    return _has_gpu_device_types(method.sig)
end

"""
    _root_module(mod::Module) -> Module

Walk up the module hierarchy to find the top-level package module (stops at Main).
"""
function _root_module(mod::Module)
    root = mod
    while true
        parent = parentmodule(root)
        (parent === root || parent === Main) && return root
        root = parent
    end
end

# GPU infrastructure root packages.
const _GPU_SEED_NAMES = Set{String}(["GPUCompiler", "KernelAbstractions", "GPUArraysCore"])

"""
    _build_gpu_ecosystem_packages() -> Set{Symbol}

Detect the GPU compilation ecosystem by walking the package dependency graph
using `Base.identify_package`. Returns the set of top-level package name

Returns an empty set if no GPU seed packages are loaded.
"""
function _build_gpu_ecosystem_packages()
    loaded_pkgs = Dict{String, Base.PkgId}()
    for (pkgid, mod) in Base.loaded_modules
        pkgid.name in ("Main", "Core", "Base") && continue
        loaded_pkgs[pkgid.name] = pkgid
    end

    # If no GPU seeds are loaded, nothing to exclude
    any(name -> haskey(loaded_pkgs, name), _GPU_SEED_NAMES) || return Set{Symbol}()

    # Start with packages that are actually loaded
    gpu_pkgs = Set{String}(filter(n -> haskey(loaded_pkgs, n), _GPU_SEED_NAMES))

    # Get packages that GPUCompiler uses
    compiler_seed = get(loaded_pkgs, "GPUCompiler", nothing)
    if compiler_seed !== nothing
        for (dep_name, _) in loaded_pkgs
            dep_name in gpu_pkgs && continue
            try
                if Base.identify_package(compiler_seed, dep_name) !== nothing
                    push!(gpu_pkgs, dep_name)
                end
            catch
            end
        end
    end

    # Get packages that use GPUCompiler
    seed_set = Set{String}(filter(n -> haskey(loaded_pkgs, n), _GPU_SEED_NAMES))
    changed = true
    while changed
        changed = false
        for (pkg_name, pkgid) in loaded_pkgs
            pkg_name in gpu_pkgs && continue
            for seed_name in seed_set
                try
                    if Base.identify_package(pkgid, seed_name) !== nothing
                        push!(gpu_pkgs, pkg_name)
                        changed = true
                        break  # already added
                    end
                catch
                end
            end
        end
    end

    if _TRACE_DEBUG[]
        println(stderr, "[DEBUG GPU ecosystem] detected packages: $(sort(collect(gpu_pkgs)))")
    end

    return Set{Symbol}(Symbol(n) for n in gpu_pkgs)
end

"""
    _should_patch_method(method::Method) -> Bool

Check whether a Method should have its source patched.
"""
function _should_patch_method(method::Method)
    _is_base_or_core(method.module) && return false
    _is_gpu_method(method) && return false
    return _passes_common_exclusions(method.module, method.name)
end

"""
    _discover_additional_methods(interp::TracingInterpreter) -> Set{Method}

Discover user-module methods that weren't in the compiler's inference walk
but could be called at runtime from spawned tasks, callbacks, or other
deferred-execution contexts.
"""
function _discover_additional_methods(interp::TracingInterpreter)
    extra_methods = Set{Method}()

    # Determine which user modules are involved
    user_modules = Set{Module}()
    for mi in interp.instrumented
        method = mi.def
        isa(method, Method) || continue
        mod = method.module
        _is_base_or_core(mod) && continue
        mod === TAUProfile && continue
        _is_module_excluded(mod) && continue
        push!(user_modules, mod)
    end

    # For each user module, collect methods of named functions
    for mod in user_modules
        for name in names(mod, all=true, imported=false)
            isdefined(mod, name) || continue
            local val
            try
                val = getfield(mod, name)
            catch
                continue
            end
            val isa Function || continue
            for m in methods(val)
                m.module === mod || continue
                _should_patch_method(m) || continue
                push!(extra_methods, m)
            end
        end
    end

    # Scan codegen cache for closure constructions from user modules
    world = CC.get_inference_world(interp)
    for (ci, codeinfo) in interp.codegen
        for stmt in codeinfo.code
            if stmt isa Expr && stmt.head === :new && length(stmt.args) >= 1
                typ = stmt.args[1]
                if typ isa DataType
                    typ_mod = typ.name.module
                    (_is_base_or_core(typ_mod) || typ_mod === TAUProfile) && continue
                    try
                        sig = Tuple{typ, Vararg{Any}}
                        matches = Base._methods_by_ftype(sig, -1, world)
                        if matches !== nothing
                            for match in matches
                                m = match.method
                                _should_patch_method(m) || continue
                                push!(extra_methods, m)
                            end
                        end
                    catch
                    end
                end
            end
        end
    end

    return extra_methods
end

"""
    _collect_function_methods!(dest::Set{Method}, f::Function)

Add all patchable methods of `f` to `dest`.
"""
function _collect_function_methods!(dest::Set{Method}, @nospecialize(f::Function))
    for m in methods(f)
        _should_patch_method(m) || continue
        push!(dest, m)
    end
end

"""
    _foreach_globalref(f, stmt)

Call `f(ref::GlobalRef)` for each GlobalRef found in `stmt`.
"""
function _foreach_globalref(f, @nospecialize(stmt))
    if stmt isa GlobalRef
        f(stmt)
    elseif stmt isa Expr
        for arg in stmt.args
            arg isa GlobalRef && f(arg)
        end
    end
end

"""
    _collect_globalref_methods!(dest::Set{Method}, stmt)

If `stmt` is or contains a GlobalRef that resolves to a Function, add its
patchable methods to `dest`.
"""
function _collect_globalref_methods!(dest::Set{Method}, @nospecialize(stmt))
    _foreach_globalref(ref -> _try_collect_ref!(dest, ref), stmt)
end

function _try_collect_ref!(dest::Set{Method}, ref::GlobalRef)
    try
        val = getfield(ref.mod, ref.name)
        val isa Function && _collect_function_methods!(dest, val)
    catch; end
end

"""
    _discover_invokelatest_targets(interp::TracingInterpreter) -> Set{Method}

Scan the codegen cache for calls to `invokelatest` and `Core.invoke_in_world`,
extracting target functions from their arguments.
"""
function _discover_invokelatest_targets(interp::TracingInterpreter)
    extra_methods = Set{Method}()
    world = CC.get_inference_world(interp)

    for (ci, codeinfo) in interp.codegen
        for stmt in codeinfo.code
            stmt isa Expr && stmt.head === :call || continue
            length(stmt.args) >= 2 || continue

            callee = stmt.args[1]
            local target_arg

            # Match invokelatest(target, args...)
            if callee === invokelatest || (callee isa GlobalRef && callee.name === :invokelatest)
                target_arg = stmt.args[2]
            # Match Core.invoke_in_world(world, target, args...)
            elseif callee === Core.invoke_in_world ||
                   (callee isa GlobalRef && callee.name === :invoke_in_world)
                length(stmt.args) >= 3 || continue
                target_arg = stmt.args[3]
            else
                continue
            end

            # Resolve the target function
            local target_func
            try
                if target_arg isa GlobalRef
                    target_func = getfield(target_arg.mod, target_arg.name)
                elseif target_arg isa Function
                    target_func = target_arg
                else
                    continue
                end
            catch
                continue
            end

            # Add all methods of the target function and discover its callees.
            _collect_function_methods!(extra_methods, target_func)

            # Scan for GlobalRef nodes that resolve to functions.
            for m in methods(target_func)
                _should_patch_method(m) || continue
                try
                    src = Base.uncompressed_ir(m)
                    for inner_stmt in src.code
                        _collect_globalref_methods!(extra_methods, inner_stmt)
                    end
                catch
                end
            end
        end
    end

    return extra_methods
end

"""
    _discover_gpu_kernel_methods(candidate_methods) -> Set{Method}

Scan candidate methods' lowered CodeInfo for GPU device intrinsic calls
(threadIdx, blockIdx, blockDim, sync_threads, etc.) to identify GPU kernel
functions. Then transitively discover their device-side callees (helper
functions called from kernels). Returns a set of Methods to exclude.
"""
function _discover_gpu_kernel_methods(candidate_methods)
    kernel_methods = Set{Method}()
    callee_visited = Set{Method}()

    for method in candidate_methods
        # Skip methods that would never be patched anyway
        _is_base_or_core(method.module) && continue
        try
            src = Base.uncompressed_ir(method)
            if _calls_gpu_device_intrinsics(src)
                push!(kernel_methods, method)
                _collect_kernel_callees!(kernel_methods, src, callee_visited)
            end
        catch
        end
    end

    return kernel_methods
end

# Known GPU device intrinsic names
const _GPU_DEVICE_INTRINSIC_NAMES = Set{Symbol}([
    # Thread/block indexing
    :threadIdx, :blockIdx, :blockDim, :gridDim, :warpsize, :laneid,
    # Synchronization
    :sync_threads, :sync_warp, :sync_threads_count, :sync_threads_and, :sync_threads_or,
    # Memory fences
    :threadfence, :threadfence_block, :threadfence_system,
    # Shared memory (macro expansions)
    :CuStaticSharedArray, :CuDynamicSharedArray,
    # Warp-level primitives
    :shfl_sync, :shfl_up_sync, :shfl_down_sync, :shfl_xor_sync,
    :vote_all_sync, :vote_any_sync, :vote_uni_sync, :vote_ballot_sync,
    :active_mask,
    # Timing
    :clock, :nanosleep,
])

"""
    _calls_gpu_device_intrinsics(src::Core.CodeInfo) -> Bool

Check if a method's lowered CodeInfo calls any GPU device intrinsics.
"""
function _calls_gpu_device_intrinsics(src::Core.CodeInfo)
    found = Ref(false)
    for stmt in src.code
        _foreach_globalref(stmt) do ref
            _is_gpu_intrinsic_ref(ref) && (found[] = true)
        end
        found[] && return true
    end
    return false
end

function _is_gpu_intrinsic_ref(ref::GlobalRef)
    ref.name in _GPU_DEVICE_INTRINSIC_NAMES && return true
    try
        val = getfield(ref.mod, ref.name)
        val isa Function || return false
        return nameof(val) in _GPU_DEVICE_INTRINSIC_NAMES
    catch
        return false
    end
end

"""
    _collect_kernel_callees!(dest::Set{Method}, src::Core.CodeInfo, visited::Set{Method})

Recursively scan a CodeInfo for GlobalRef calls to user-module functions and add
their methods to `dest`. Used to discover device helper functions called from GPU.
"""
function _collect_kernel_callees!(dest::Set{Method}, src::Core.CodeInfo, visited::Set{Method})
    for stmt in src.code
        _foreach_globalref(ref -> _try_collect_kernel_ref!(dest, ref, visited), stmt)
    end
end

function _try_collect_kernel_ref!(dest::Set{Method}, ref::GlobalRef, visited::Set{Method})
    _is_base_or_core(ref.mod) && return
    ref.mod === TAUProfile && return
    try
        val = getfield(ref.mod, ref.name)
        val isa Function || return
        # Only follow into user-module functions, not GPU package functions
        pm = parentmodule(val)
        _is_base_or_core(pm) && return
        for m in methods(val)
            m in visited && continue
            push!(visited, m)
            push!(dest, m)
            # Recurse into this callee to find its callees
            try
                inner_src = Base.uncompressed_ir(m)
                _collect_kernel_callees!(dest, inner_src, visited)
            catch
            end
        end
    catch
    end
end

# CI-owner-based foreign interpreter detection

"""
    _has_foreign_ci(method::Method) -> Bool

Check if any specialization of a Method has a CodeInstance owned by a foreign
AbstractInterpreter (i.e., not `nothing` and not our TracingInterpreter).
"""
function _has_foreign_ci(method::Method)
    for mi in Base.specializations(method)
        mi === nothing && continue
        _mi_has_foreign_ci(mi) && return true
    end
    return false
end

function _mi_has_foreign_ci(mi::MethodInstance)
    isdefined(mi, :cache) || return false
    ci = mi.cache
    while true
        if ci.owner !== nothing && !(ci.owner isa TracingInterpreter)
            return true
        end
        isdefined(ci, :next) || return false
        ci = ci.next
    end
end

"""
    _discover_foreign_ci_methods(candidate_methods) -> Set{Method}

Scan candidate methods for any that have CodeInstances from foreign interpreters.
Returns the set of methods that should be excluded from source patching.
"""
function _discover_foreign_ci_methods(candidate_methods)
    foreign_methods = Set{Method}()
    for method in candidate_methods
        _is_base_or_core(method.module) && continue
        try
            if _has_foreign_ci(method)
                push!(foreign_methods, method)
            end
        catch
        end
    end
    return foreign_methods
end

"""
    _patch_method_set!(methods, seen, patched, debug_tag, gpu_packages)

Patch Method sources with entry/exit hooks.
"""
function _patch_method_set!(methods, seen::Set{Method},
                            patched::Vector{Tuple{Method, Any}}, debug_tag::String,
                            gpu_packages::Set{Symbol})
    for method in methods
        method in seen && continue
        push!(seen, method)

        _is_base_or_core(method.module) && continue
        # Static GPU detection
        if _gpu_exclude_static[]
            _is_gpu_method(method) && continue
            # Skip methods from GPU ecosystem packages (detected via dependency graph).
            # These methods may be compiled for GPU or used by the GPU compilation
            # toolchain (e.g. LLVM.jl), and patching their sources breaks GPU compilation.
            if !isempty(gpu_packages)
                root_name = nameof(_root_module(method.module))
                if root_name in gpu_packages
                    if _TRACE_DEBUG[]
                        println(stderr, "[DEBUG patch $debug_tag SKIP GPU pkg] $(method.module).$(method.name) (package $root_name)")
                    end
                    continue
                end
            end
        end
        isdefined(method, :source) || continue
        original_source = method.source
        try
            src = Base.uncompressed_ir(method)
            _is_complex_enough(src) || continue

            label = _method_label(method)
            new_src = _patch_codeinfo_with_hooks(src, label)
            compressed = ccall(:jl_compress_ir, Any, (Any, Any), method, new_src)
            method.source = compressed
            push!(patched, (method, original_source))

            if _TRACE_DEBUG[]
                println(stderr, "[DEBUG patch $debug_tag] $(method.module).$(method.name) ($(length(src.code)) → $(length(new_src.code)) stmts)")
            end
        catch ex
            if _TRACE_DEBUG[]
                println(stderr, "[DEBUG patch $debug_tag FAILED] $(method.module).$(method.name): $ex")
            end
        end
    end
end

"""
    _patch_method_sources!(interp::TracingInterpreter) -> Vector{Tuple{Method, Any}}

For each instrumented Method, patch its source CodeInfo to include
entry/exit hook calls. 
"""
function _patch_method_sources!(interp::TracingInterpreter)
    patched_methods = Vector{Tuple{Method, Any}}()
    seen_methods = Set{Method}()

    # Detect GPU ecosystem packages via dependency graph (static detection).
    # Methods from these packages are skipped because their Method.source may be
    # read by GPU compilers (GPUCompiler.jl) during kernel compilation, and our
    # hooks (try/finally, ccall to write, rethrow) are invalid in GPU IR.
    gpu_packages = _gpu_exclude_static[] ? _build_gpu_ecosystem_packages() : Set{Symbol}()

    phase2a_methods = Set{Method}(mi.def for mi in interp.instrumented if mi.def isa Method)
    invoke_targets = _discover_invokelatest_targets(interp)
    extra_methods = _trace_deferred[] ? _discover_additional_methods(interp) : Set{Method}()

    all_candidates = union(phase2a_methods, invoke_targets, extra_methods)

    # CI-owner detection: check if another AbstractInterpreter (e.g. GPUCompiler)
    # is the owner of this CI
    if _gpu_exclude_ci_owner[]
        foreign_ci_methods = _discover_foreign_ci_methods(all_candidates)
        for m in foreign_ci_methods
            push!(seen_methods, m)
            if _TRACE_DEBUG[]
                println(stderr, "[DEBUG GPU skip (foreign CI)] $(m.module).$(m.name)")
            end
        end
    end

    # Scan CodeInfo for GPU device intrinsic calls.
    if _gpu_exclude_static[]
        gpu_kernel_methods = _discover_gpu_kernel_methods(all_candidates)
        for m in gpu_kernel_methods
            if !(m in seen_methods)
                push!(seen_methods, m)
                if _TRACE_DEBUG[]
                    println(stderr, "[DEBUG GPU kernel skip (static)] $(m.module).$(m.name)")
                end
            end
        end
    end

    # Patch methods that were instrumented during compilation
    _patch_method_set!(phase2a_methods, seen_methods, patched_methods, "phase2a", gpu_packages)

    # Patch targets of invokelatest/invoke_in_world
    _patch_method_set!(invoke_targets, seen_methods, patched_methods, "invokelatest target", gpu_packages)

    # Patch additional user-module methods
    if _trace_deferred[]
        _patch_method_set!(extra_methods, seen_methods, patched_methods, "extra", gpu_packages)
    end

    return patched_methods
end

"""
    _restore_method_sources!(patched::Vector{Tuple{Method, Any}})

Restore original method sources after traced execution.
"""
function _restore_method_sources!(patched::Vector{Tuple{Method, Any}})
    for (method, original_source) in patched
        method.source = original_source
    end
end

# Public API

"""
    tau_rewrite_and_call(f, args...) -> result

Compile `f` with tracing instrumentation for the given argument types,
then execute it. All callees are automatically instrumented.

    tau_rewrite_and_call(sin, 1.0)
    tau_rewrite_and_call(sort, [3, 1, 2])
"""
function tau_rewrite_and_call(@nospecialize(f), args...)
    tt = Tuple{map(typeof, args)...}
    ci, patched = _trace_compile(f, tt)
    try
        return invoke(f, ci, args...)
    finally
        _restore_method_sources!(patched)
    end
end

"""
    tau_rewrite(f, argtypes::Tuple) -> callable

Compile `f` with tracing instrumentation and return a callable wrapper.
    traced = tau_rewrite(my_func, (Float64, Float64))
    traced(1.0, 2.0)
"""
function tau_rewrite(@nospecialize(f), @nospecialize(argtypes::Tuple))
    tt = Tuple{argtypes...}
    ci, patched = _trace_compile(f, tt)
    return function(args...)
        try
            return invoke(f, ci, args...)
        finally
            _restore_method_sources!(patched)
        end
    end
end

"""
    @tau_rewrite f(args...)

Convenience macro: instruments `f` and all its callees with entry/exit
tracing, then immediately calls the traced version with the given arguments.

    @tau_rewrite composite(5.0, -6.0)
"""
macro tau_rewrite(call_expr)
    Meta.isexpr(call_expr, :call) || error("@tau_rewrite expects a function call, got: $call_expr")
    f = call_expr.args[1]
    args = call_expr.args[2:end]
    return quote
        let _args = ($(esc.(args)...),)
            _tt = Tuple{map(typeof, _args)...}
            _ci, _patched = $_trace_compile($(esc(f)), _tt)
            try
                invoke($(esc(f)), _ci, _args...)
            finally
                $_restore_method_sources!(_patched)
            end
        end
    end
end

end # module TAUProfile
