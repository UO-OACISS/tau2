# ============================================================================
# config.jl — user-configurable options
#
# Part of the TAUProfile module (LLVM backend).
# ============================================================================

# ============================================================================
# Configuration
# ============================================================================

const _tracing_enabled = Ref{Bool}(true)

"""Enable tracing instrumentation."""
enable_tracing!() = (_tracing_enabled[] = true; nothing)

"""Disable tracing instrumentation."""
disable_tracing!() = (_tracing_enabled[] = false; nothing)

"""Check if tracing is currently enabled."""
tracing_enabled() = _tracing_enabled[]

# ============================================================================
# Exclusion API
# ============================================================================

const _excluded_functions = Set{Symbol}()
const _excluded_modules = Dict{Symbol, Bool}()  # module name => exact?
const _excluded_prefixes = Set{String}()
const _whitelisted_modules = Dict{Symbol, Bool}()  # module name => exact?

"""Exclude a function from instrumentation by function object, Symbol, or String."""
tau_rewrite_exclude_function(f::Function) = (push!(_excluded_functions, nameof(f)); nothing)
tau_rewrite_exclude_function(s::Symbol) = (push!(_excluded_functions, s); nothing)
tau_rewrite_exclude_function(s::String) = (push!(_excluded_functions, Symbol(s)); nothing)
tau_rewrite_exclude_function(items...) = (for it in items; tau_rewrite_exclude_function(it); end; nothing)

"""
Exclude all functions in a module from instrumentation.
When `exact=false` (default), submodules are also excluded; `exact=true` excludes only the named module.
"""
tau_rewrite_exclude_module(m::Module; exact::Bool=false) = (_excluded_modules[nameof(m)] = exact; nothing)
tau_rewrite_exclude_module(s::Symbol; exact::Bool=false) = (_excluded_modules[s] = exact; nothing)
tau_rewrite_exclude_module(s::String; exact::Bool=false) = (_excluded_modules[Symbol(s)] = exact; nothing)
tau_rewrite_exclude_module(items...; exact::Bool=false) = (for it in items; tau_rewrite_exclude_module(it; exact); end; nothing)

"""Exclude all functions whose name starts with `prefix`."""
tau_rewrite_exclude_prefix(prefix::String) = (push!(_excluded_prefixes, prefix); nothing)
tau_rewrite_exclude_prefix(prefix::Symbol) = tau_rewrite_exclude_prefix(string(prefix))
tau_rewrite_exclude_prefix(prefixes::String...) = (for p in prefixes; tau_rewrite_exclude_prefix(p); end; nothing)

"""
    tau_rewrite_include_module_only(m; exact=false)

Whitelist mode: ONLY instrument functions from whitelisted modules.
When `exact=false` (default), submodules are also whitelisted.
Call with no arguments or use `tau_rewrite_reset_exclusions()` to clear the whitelist.
"""
tau_rewrite_include_module_only(m::Module; exact::Bool=false) = (_whitelisted_modules[nameof(m)] = exact; nothing)
tau_rewrite_include_module_only(s::Symbol; exact::Bool=false) = (_whitelisted_modules[s] = exact; nothing)
tau_rewrite_include_module_only(s::String; exact::Bool=false) = (_whitelisted_modules[Symbol(s)] = exact; nothing)
tau_rewrite_include_module_only(items...; exact::Bool=false) = (for it in items; tau_rewrite_include_module_only(it; exact); end; nothing)

# ============================================================================
# Depth limit API
# ============================================================================

const _max_depth = Ref{Int}(typemax(Int))  # global limit (0 = unlimited)
const _module_depth_limits = Dict{Symbol, Tuple{Int, Bool}}()  # module name => (limit, exact?)

"""
    tau_rewrite_set_recursion_limit(n::Int)

Set global recursion depth limit. Functions deeper than `n` levels from the root
are not instrumented. `n=0` means unlimited.
"""
function tau_rewrite_set_recursion_limit(n::Int)
    _max_depth[] = n == 0 ? typemax(Int) : n
    nothing
end

"""
    tau_rewrite_set_recursion_limit(mod, n::Int; exact=false)

Set per-module recursion depth limit. `n` means "trace `n` levels into this module."
Module-relative depth resets to 0 on module boundary crossings.
When `exact=false` (default), the limit applies to submodules too.
"""
tau_rewrite_set_recursion_limit(m::Module, n::Int; exact::Bool=false) = (_module_depth_limits[nameof(m)] = (n, exact); nothing)
tau_rewrite_set_recursion_limit(s::Symbol, n::Int; exact::Bool=false) = (_module_depth_limits[s] = (n, exact); nothing)
tau_rewrite_set_recursion_limit(s::String, n::Int; exact::Bool=false) = (_module_depth_limits[Symbol(s)] = (n, exact); nothing)

"""
    tau_rewrite_include_types(enabled::Bool=true)

Include argument types in hook labels. Off by default.
"""
const _include_types = Ref{Bool}(false)
tau_rewrite_include_types(enabled::Bool=true) = (_include_types[] = enabled; nothing)

# If set, force inlining off for every function, enabling every function
# to be traced, albeit with very large performance degradation
const _disable_julia_inlining = Ref{Bool}(false)

"""
    tau_rewrite_force_noinline(enabled::Bool=true)

Force inlining to be turned off for all traced functions.
"""
tau_rewrite_force_noinline(enabled::Bool=true) =
    (_disable_julia_inlining[] = enabled; nothing)

# Whether to wrap Phase 1 rewriting in a timer
const _time_phase1 = Ref{Bool}(true)

"""
    tau_rewrite_time_phase1(enabled::Bool=true)

Toggle timing of the Phase 1 rewrite.
"""
tau_rewrite_time_phase1(enabled::Bool=true) = (_time_phase1[] = enabled; nothing)

# Phase 2 rewrite timing mode.
const _PHASE2_TIMING_MODES = (:off, :separate, :combined)
const _phase2_timing_mode = Ref{Symbol}(:separate)

"""
    tau_rewrite_time_phase2(mode::Symbol=:separate)

Select how the Phase 2 rewrite is timed:

- `:separate` (default) — accumulate into a distinct `.TAU Julia Phase 2 Rewrite` timer.
- `:combined` — accumulate into the same `.TAU Julia Rewrite` timer as Phase 1.
- `:off` — do not time Phase 2.
"""
function tau_rewrite_time_phase2(mode::Symbol=:separate)
    mode in _PHASE2_TIMING_MODES ||
        throw(ArgumentError("mode must be one of " *
                            "$(_PHASE2_TIMING_MODES), got :$mode"))
    _phase2_timing_mode[] = mode
    nothing
end

"""Clear all exclusions, whitelists, depth limits, and type inclusion."""
function tau_rewrite_reset_exclusions()
    empty!(_excluded_functions)
    empty!(_excluded_modules)
    empty!(_excluded_prefixes)
    empty!(_whitelisted_modules)
    _max_depth[] = typemax(Int)
    empty!(_module_depth_limits)
    _include_types[] = false
    _disable_julia_inlining[] = false
    empty!(_phase2_modules)
    nothing
end

# ============================================================================
# Module hierarchy checks for whitelist/exclusion/depth limits
# ============================================================================

"""
    _find_in_ancestors(f, mod::Module)

Walk `mod` and its ancestors; return the first non-nothing result of `f(m, is_self)`,
where `is_self` is true only for `mod` itself. Returns `nothing` if `f` never matches.
"""
function _find_in_ancestors(f, mod::Module)
    current = mod
    while true
        r = f(current, current === mod)
        r === nothing || return r
        parent = parentmodule(current)
        parent === current && return nothing
        current = parent
    end
end

"""
    _module_limit_for(mod::Module) -> Union{Int, Nothing}

Look up the per-module depth limit, walking up the module hierarchy for non-exact limits.
"""
function _module_limit_for(mod::Module)
    _find_in_ancestors(mod) do m, is_self
        entry = get(_module_depth_limits, nameof(m), nothing)
        entry === nothing && return nothing
        limit, exact = entry
        return (is_self || !exact) ? limit : nothing
    end
end

"""
    _is_module_whitelisted(mod::Module) -> Bool

Check if a module (or any ancestor) is in the whitelist. An empty whitelist passes everything.
"""
function _is_module_whitelisted(mod::Module)
    isempty(_whitelisted_modules) && return true  # no whitelist = everything passes
    found = _find_in_ancestors(mod) do m, is_self
        exact = get(_whitelisted_modules, nameof(m), nothing)
        return (exact !== nothing && (is_self || !exact)) ? true : nothing
    end
    return found === true
end

"""
    _is_module_excluded(mod::Module) -> Bool

Check if a module (or any ancestor) is in the exclusion set.
"""
function _is_module_excluded(mod::Module)
    isempty(_excluded_modules) && return false
    found = _find_in_ancestors(mod) do m, is_self
        exact = get(_excluded_modules, nameof(m), nothing)
        return (exact !== nothing && (is_self || !exact)) ? true : nothing
    end
    return found === true
end

"""
    _passes_config_filter(funcname, mod) -> Bool

Apply the user-configured filters shared by Phase 1 and Phase 2: tracing enabled,
function name and prefix exclusions, module exclusion and module whitelist.
Passing `nothing` for `funcname` or `mod` skips the checks that need it.
"""
function _passes_config_filter(funcname::Union{String, Nothing}, mod::Union{Module, Nothing})
    _tracing_enabled[] || return false

    if funcname !== nothing
        Symbol(funcname) in _excluded_functions && return false
        for prefix in _excluded_prefixes
            startswith(funcname, prefix) && return false
        end
    end

    if mod !== nothing
        _is_module_excluded(mod) && return false
        _is_module_whitelisted(mod) || return false
    end

    return true
end
