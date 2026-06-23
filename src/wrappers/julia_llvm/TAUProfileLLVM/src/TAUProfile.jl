#=
/****************************************************************************
**			TAU Portable Profiling Package                                 **
**			http://www.cs.uoregon.edu/research/tau                         **
*****************************************************************************
**    Copyright 2009-2026                    						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**	File 		: TracingLLVMPlugin.jl 		          	                   **
**	Description 	: TAU Tracing for native OTF2 generation			   **
**  Author      : Nicholas Chaimov                                         **
**	Contact		: tau-bugs@cs.uoregon.edu               	               **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau           **
**                                                                         **
**      Description     :  LLVM Plugin-based Julia rewriter                **
**                                                                         **
****************************************************************************/
#
# This file contains code derived from GPUCompiler.jl, which is licensed under the terms of the 
# MIT "Expat" License:
#
# Copyright (c) 2019-present: Julia Computing and other contributors
#
#Copyright (c) 2014-2018: Tim Besard
#
#All Rights Reserved.
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#

#=
TracingLLVMPlugin.jl — Function entry/exit tracing via LLVM pass plugin
=#

module TAUProfile

export tau_rewrite_and_call, trace_code, @tau_rewrite,
       tau_rewrite_exclude_function, tau_rewrite_exclude_module,
       tau_rewrite_exclude_prefix, tau_rewrite_reset_exclusions,
       tau_rewrite_include_module_only,
       tau_rewrite_set_recursion_limit, tau_rewrite_include_types,
       tau_rewrite_force_noinline,
       enable_tracing!, disable_tracing!, tracing_enabled,
       # Classic-parity public API (Phase 3) so the same script runs on either backend
       tau_start, tau_stop, @tau, @tau_func, tau_rewrite,
       tau_rewrite_deferred_contexts, tau_rewrite_set_min_complexity

using LLVM

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

function __init__()
    _entry_hook_ptr[] = @cfunction(_entry_hook, Cvoid, (Ptr{UInt8},))
    _exit_hook_ptr[] = @cfunction(_exit_hook, Cvoid, (Ptr{UInt8},))
    _jl_fptr_args[] = unsafe_load(cglobal(:jl_fptr_args_addr, Ptr{Cvoid}))
    _jl_fptr_const_return[] = unsafe_load(cglobal(:jl_fptr_const_return_addr, Ptr{Cvoid}))

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

# Manual instrumentation API

"""
    tau_start(name::String)

Start a TAU timer with the given name. No-op when libTAU is not loaded.
"""
function tau_start(name::String)
    if isempty(_libTAU[])
        return  # Silently skip if library not loaded
    end

    # TODO remove forcing task sticky when we can handle task migration ~nchaimov
    current_task().sticky = true # workaround for lack of handling of task migration
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

Wrap an expression with a TAU timer named `name`, stopping it on normal and
exceptional exit.

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
# Depth computation from CI call graph
# ============================================================================

"""
    _compute_depth_maps(populated, root_mi) -> (depth_map, mod_depth_map)

Compute absolute depth and module-relative depth for each MethodInstance.
"""
function _compute_depth_maps(populated, root_mi::Core.MethodInstance)
    ci_to_mi = IdDict{Core.CodeInstance, Core.MethodInstance}()
    ci_to_src = Dict{Core.CodeInstance, Core.CodeInfo}()
    mi_to_ci = IdDict{Core.MethodInstance, Core.CodeInstance}()

    for (ci, src) in populated
        mi = ci.def::Core.MethodInstance
        ci_to_mi[ci] = mi
        ci_to_src[ci] = src
        if !haskey(mi_to_ci, mi)
            mi_to_ci[mi] = ci
        end
    end

    mi_callees = IdDict{Core.MethodInstance, Vector{Core.MethodInstance}}()
    for (ci, src) in populated
        caller_mi = ci_to_mi[ci]
        callees = get!(Vector{Core.MethodInstance}, mi_callees, caller_mi)
        for stmt in src.code
            if isa(stmt, Expr) && stmt.head === :invoke
                callee_ci = stmt.args[1]
                if isa(callee_ci, Core.CodeInstance) && haskey(ci_to_mi, callee_ci)
                    push!(callees, ci_to_mi[callee_ci])
                end
            end
        end
    end

    depth_map = IdDict{Core.MethodInstance, Int}()
    mod_depth_map = IdDict{Core.MethodInstance, Int}()

    # BFS from root
    depth_map[root_mi] = 0
    mod_depth_map[root_mi] = 0
    queue = Core.MethodInstance[root_mi]

    while !isempty(queue)
        caller_mi = popfirst!(queue)
        caller_depth = depth_map[caller_mi]
        caller_mod_depth = mod_depth_map[caller_mi]

        caller_mod = try
            d = caller_mi.def
            d isa Method ? d.module : nothing
        catch
            nothing
        end

        for callee_mi in get(mi_callees, caller_mi, Core.MethodInstance[])
            haskey(depth_map, callee_mi) && continue  # first path wins (shortest)
            depth_map[callee_mi] = caller_depth + 1

            callee_mod = try
                d = callee_mi.def
                d isa Method ? d.module : nothing
            catch
                nothing
            end

            if caller_mod !== nothing && callee_mod !== nothing && caller_mod === callee_mod
                mod_depth_map[callee_mi] = caller_mod_depth + 1
            else
                mod_depth_map[callee_mi] = 0
            end

            push!(queue, callee_mi)
        end
    end

    return depth_map, mod_depth_map
end

"""
    _module_limit_for(mod_name::Symbol) -> Union{Int, Nothing}

Look up the per-module depth limit, walking up the module hierarchy for non-exact limits.
"""
function _module_limit_for(mod::Module)
    current = mod
    while true
        name = nameof(current)
        if haskey(_module_depth_limits, name)
            limit, exact = _module_depth_limits[name]
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
    _within_depth_limit(mi, depth, mod_depth) -> Bool

Check whether an MI at the given depth should be instrumented.
"""
function _within_depth_limit(mi::Core.MethodInstance, depth::Int, mod_depth::Int)
    global_max = _max_depth[]
    global_max == typemax(Int) && isempty(_module_depth_limits) && return true

    depth >= global_max && return false

    method = mi.def
    isa(method, Method) || return true
    mod = method.module

    if !isempty(_module_depth_limits)
        mod_limit = _module_limit_for(mod)
        if mod_limit !== nothing
            return mod_depth < mod_limit
        end
    end

    return true
end

# ============================================================================
# LLVM function metadata 
# ============================================================================

"""Mapping from LLVM function names to Julia module name strings."""
const _llvm_func_module_map = Dict{String, String}()

"""Mapping from LLVM function names to (depth, mod_depth) for depth limit checks."""
const _llvm_func_depth_map = Dict{String, Tuple{Int, Int}}()

"""Mapping from LLVM function names to MethodInstance for exclusion checks."""
const _llvm_func_mi_map = Dict{String, Core.MethodInstance}()

"""Mapping from LLVM function names to argument type strings (for include_types mode)."""
const _llvm_func_argtypes_map = Dict{String, String}()

"""Set of LLVM function names whose Julia source has @noinline annotation."""
const _llvm_func_noinline_set = Set{String}()

"""Fallback map from (funcname, basename_file, line) to (module_name, MethodInstance)."""
const _mi_info_fallback_map = Dict{Tuple{String, String, Int}, Tuple{String, Core.MethodInstance}}()

# ============================================================================
# Function label building from LLVM debug info
# ============================================================================

"""
    _build_function_label(fn::LLVM.Function) -> String

Build a human-readable label from the LLVM function's debug info (DISubprogram).
Format: "Module.name [{file} {line}]" or "Module.name(types) [{file} {line}]".
"""
function _build_function_label(fn::LLVM.Function)
    llvm_name = LLVM.name(fn)
    mod_prefix = get(_llvm_func_module_map, llvm_name, nothing)
    argtypes = _include_types[] ? get(_llvm_func_argtypes_map, llvm_name, nothing) : nothing

    sp = LLVM.subprogram(fn)
    if sp !== nothing
        funcname = LLVM.name(sp)
        if funcname !== nothing && !isempty(funcname)
            line_num = LLVM.line(sp)
            di_file = LLVM.file(sp)
            filename = LLVM.filename(di_file)
            display_name = mod_prefix !== nothing ? "$mod_prefix.$funcname" : funcname
            if argtypes !== nothing
                display_name = "$display_name($argtypes)"
            end
            if filename !== nothing && !isempty(filename) && filename != "none"
                return "$display_name [{$filename} {$line_num}]"
            else
                return display_name
            end
        end
    end
    clean = _clean_llvm_name(llvm_name)
    display_name = mod_prefix !== nothing ? "$mod_prefix.$clean" : clean
    if argtypes !== nothing
        display_name = "$display_name($argtypes)"
    end
    return display_name
end

"""
    _clean_llvm_name(name::String) -> String

Clean up LLVM function names. Julia emits names like `julia_simple_add_1234`
or `j_simple_add_1234`. Strip the prefix and trailing numeric ID.
"""
function _clean_llvm_name(name::String)
    clean = name
    if startswith(clean, "julia_")
        clean = clean[7:end]
    elseif startswith(clean, "j_")
        clean = clean[3:end]
    end
    m = match(r"^(.+)_\d+$", clean)
    if m !== nothing
        clean = String(m.captures[1])
    end
    return clean
end

"""
    _get_function_name(fn::LLVM.Function) -> Union{String, Nothing}

Extract the clean function name from debug info, for exclusion checks.
"""
function _get_function_name(fn::LLVM.Function)
    sp = LLVM.subprogram(fn)
    sp === nothing && return nothing
    name = LLVM.name(sp)
    (name === nothing || isempty(name)) && return nothing
    return name
end

"""
    _format_argtypes(mi::Core.MethodInstance) -> String

Format argument types from a MethodInstance's specTypes for label display.
"""
function _format_argtypes(mi::Core.MethodInstance)
    spec = Base.unwrap_unionall(mi.specTypes)
    nparams = length(spec.parameters)
    if nparams >= 2
        argtypes = spec.parameters[2:end]
    else
        return ""
    end
    parts = String[]
    for t in argtypes
        push!(parts, string(t))
    end
    return join(parts, ", ")
end

# ============================================================================
# Module hierarchy check for whitelist/exclusion
# ============================================================================

"""
    _is_module_in_set(mod::Module, modset, default_on_empty::Bool) -> Bool

Check if a module (or any of its ancestors) is in the given set.
For whitelisting: returns true if module is whitelisted
"""
function _is_module_whitelisted(mod::Module)
    isempty(_whitelisted_modules) && return true  # no whitelist = everything passes
    current = mod
    while true
        name = nameof(current)
        if haskey(_whitelisted_modules, name)
            exact = _whitelisted_modules[name]
            if current === mod || !exact
                return true
            end
        end
        parent = parentmodule(current)
        parent === current && break
        current = parent
    end
    return false
end

"""
    _is_module_excluded(mod::Module) -> Bool

Check if a module (or any ancestor) is in the exclusion set.
"""
function _is_module_excluded(mod::Module)
    isempty(_excluded_modules) && return false
    current = mod
    while true
        name = nameof(current)
        if haskey(_excluded_modules, name)
            exact = _excluded_modules[name]
            if current === mod || !exact
                return true
            end
        end
        parent = parentmodule(current)
        parent === current && break
        current = parent
    end
    return false
end

# ============================================================================
# LLVM instrumentation pass
# ============================================================================

"""
    _should_instrument(fn::LLVM.Function) -> Bool

Determine whether an LLVM function should be instrumented.
"""
function _should_instrument(fn::LLVM.Function)
    _tracing_enabled[] || return false
    LLVM.isdeclaration(fn) && return false
    LLVM.isintrinsic(fn) && return false
    fname = LLVM.name(fn)
    (startswith(fname, "julia_") || startswith(fname, "j_")) || return false

    funcname = _get_function_name(fn)

    # Function name exclusion
    if funcname !== nothing
        sym = Symbol(funcname)
        sym in _excluded_functions && return false

        # Prefix exclusion
        if !isempty(_excluded_prefixes)
            for prefix in _excluded_prefixes
                startswith(funcname, prefix) && return false
            end
        end
    end

    # Module exclusion and whitelist
    mi = get(_llvm_func_mi_map, fname, nothing)
    if mi !== nothing
        method = mi.def
        if isa(method, Method)
            mod = method.module
            _is_module_excluded(mod) && return false
            _is_module_whitelisted(mod) || return false
        end
    else
        # Fall back to module name string from the map
        mod_name = get(_llvm_func_module_map, fname, nothing)
        if mod_name !== nothing
            mod_sym = Symbol(mod_name)
            haskey(_excluded_modules, mod_sym) && return false
            if !isempty(_whitelisted_modules)
                haskey(_whitelisted_modules, mod_sym) || return false
            end
        elseif !isempty(_whitelisted_modules)
            # No module info available
            return false
        end
    end

    # Depth limit check
    if haskey(_llvm_func_depth_map, fname)
        depth, mod_depth = _llvm_func_depth_map[fname]
        if mi !== nothing
            _within_depth_limit(mi, depth, mod_depth) || return false
        else
            # No MI available
            depth >= _max_depth[] && return false
        end
    end

    return true
end

# ============================================================================
# Phase 2 Reentrancy Guard
# ============================================================================

const _PHASE2_REENTRANCY_KEY = :_tracing_llvmplugin_phase2_reentrancy

"""
    _is_phase2_reentrant()::Bool

Check if Phase 2 instrumentation is currently executing on this task.
"""
function _is_phase2_reentrant()
    t = current_task()
    storage = t.storage
    storage === nothing && return false
    return get(storage, _PHASE2_REENTRANCY_KEY, false)::Bool
end

"""
    _set_phase2_reentrant!(val::Bool)

Set the Phase 2 reentrancy flag for the current task.
"""
function _set_phase2_reentrant!(val::Bool)
    t = current_task()
    if t.storage === nothing
        t.storage = IdDict()
    end
    t.storage[_PHASE2_REENTRANCY_KEY] = val
    nothing
end

# Global flag indicating Phase 2 is active
const _phase2_active = Ref(false)

# ============================================================================
# Phase 2 Module Scoping
# ============================================================================

"""Set of modules to instrument in Phase 2. Empty = all modules allowed."""
const _phase2_modules = Set{Module}()

"""
    _is_phase2_module(mod::Module) -> Bool

Check if a module (or any ancestor) is in the Phase 2 module set.
If the set is empty, all modules are allowed (returns true).
"""
function _is_phase2_module(mod::Module)
    isempty(_phase2_modules) && return true
    current = mod
    while true
        current in _phase2_modules && return true
        parent = parentmodule(current)
        parent === current && return false
        current = parent
    end
end

"""
    _is_base_or_core(mod::Module) -> Bool

Check if a module is Base, Core, or a submodule thereof.
This prevents infinite recursion when hook code calls Base functions.
"""
function _is_base_or_core(mod::Module)
    current = mod
    while true
        current === Base && return true
        current === Core && return true
        parent = parentmodule(current)
        parent === current && return false
        current = parent
    end
end

"""
    _is_own_module(mod::Module) -> Bool

Detect if `mod` is TracingLLVMPlugin or a submodule thereof.
Walks up the module hierarchy checking for this module.
"""
function _is_own_module(mod::Module)
    current = mod
    while true
        current === @__MODULE__() && return true
        parent = parentmodule(current)
        parent === current && return false
        current = parent
    end
end

# ============================================================================
# Phase 2 Filter
# ============================================================================

"""
    _passes_phase2_filter(mi::Core.MethodInstance) -> Bool

Determine whether a MethodInstance should be instrumented in Phase 2.

Checks in order:
1. Tracing enabled
2. Method extraction: mi.def must be a Method
3. Module exclusions
4. Module whitelist
5. Function name exclusions
6. Prefix exclusions
7. Base/Core exclusion (prevents infinite recursion)
8. Phase 2 module scoping (check against _phase2_modules set)
"""
function _passes_phase2_filter(mi::Core.MethodInstance)
    _tracing_enabled[] || return false

    # Extract method
    method = mi.def
    isa(method, Method) || return false

    mod = method.module

    # Module exclusion
    _is_module_excluded(mod) && return false

    # Module whitelist
    _is_module_whitelisted(mod) || return false

    # Function name exclusion
    sym = Symbol(method.name)
    sym in _excluded_functions && return false

    # Prefix exclusion
    if !isempty(_excluded_prefixes)
        for prefix in _excluded_prefixes
            startswith(String(method.name), prefix) && return false
        end
    end

    # Base/Core exclusion
    _is_base_or_core(mod) && return false

    # Self-exclusion: never instrument TracingLLVMPlugin itself.
    _is_own_module(mod) && return false

    # Phase 2 module scoping
    if isempty(_whitelisted_modules)
        _is_phase2_module(mod) || return false
    end

    return true
end

# ============================================================================
# Exception handler helpers
# ============================================================================

"""
    _get_or_declare_fn!(mod, name, ft) -> LLVM.Function

Get an existing function declaration in the module, or create one.
"""
function _get_or_declare_fn!(mod::LLVM.Module, name::String, ft::LLVM.FunctionType)
    ref = LLVM.API.LLVMGetNamedFunction(mod, name)
    if ref != C_NULL
        return LLVM.Function(ref)
    else
        return LLVM.Function(mod, name, ft)
    end
end

"""
    _find_pgcstack(entry_bb) -> Union{LLVM.Value, Nothing}

Find the `tls_pgcstack` SSA value in the entry block.
"""
function _find_pgcstack(entry_bb::LLVM.BasicBlock)
    for inst in LLVM.instructions(entry_bb)
        iname = LLVM.name(inst)
        if iname == "tls_pgcstack" || iname == "pgcstack"
            return inst
        end
    end
    # Fallback: look for inline asm "movq %fs:0
    for inst in LLVM.instructions(entry_bb)
        if inst isa LLVM.CallInst
            s = string(inst)
            if contains(s, "movq %fs:0") || contains(s, r"%fs:")
                # Next instructions should be GEP then load
                found_gep = nothing
                past_asm = false
                for inst2 in LLVM.instructions(entry_bb)
                    if inst2 === inst
                        past_asm = true
                        continue
                    end
                    past_asm || continue
                    s2 = string(inst2)
                    if found_gep === nothing && contains(s2, "getelementptr") && contains(s2, "-8")
                        found_gep = inst2
                    elseif found_gep !== nothing && inst2 isa LLVM.LoadInst
                        return inst2
                    elseif found_gep !== nothing
                        break
                    end
                end
            end
        end
    end
    return nothing
end

"""
    _find_setup_end(entry_bb) -> Union{LLVM.Instruction, Nothing}

Find the last instruction of the pgcstack/safepoint setup in the entry block.
The setup consists of TLS access, GEP/load for pgcstack, and the safepoint
polling sequence (fence + volatile load + fence). Returns the instruction
AFTER which the entry hook should be inserted.
"""
function _find_setup_end(entry_bb::LLVM.BasicBlock)
    last_fence = nothing
    pgcstack_inst = nothing
    for inst in LLVM.instructions(entry_bb)
        iname = LLVM.name(inst)
        if iname == "tls_pgcstack" || iname == "pgcstack"
            pgcstack_inst = inst
        end
        if inst isa LLVM.FenceInst
            last_fence = inst
        end
    end
    # Return last fence if found, otherwise pgcstack load
    return last_fence !== nothing ? last_fence : pgcstack_inst
end

"""
    _fix_phi_incoming_blocks!(fn, entry_bb, try_body_bb)

After splitting the entry block, update PHI nodes in ALL blocks of the function
that reference entry_bb to reference try_body_bb instead.
"""
function _fix_phi_incoming_blocks!(fn::LLVM.Function, entry_bb::LLVM.BasicBlock, try_body_bb::LLVM.BasicBlock)
    entry_bb_ref = Base.unsafe_convert(LLVM.API.LLVMBasicBlockRef, entry_bb)

    for bb in LLVM.blocks(fn)
        bb === entry_bb && continue
        bb === try_body_bb && continue

        # Collect PHI nodes that need fixing
        phis_to_fix = LLVM.PHIInst[]
        for inst in LLVM.instructions(bb)
            inst isa LLVM.PHIInst || break
            inc = LLVM.incoming(inst)
            for i in 1:length(inc)
                _, block = inc[i]
                if Base.unsafe_convert(LLVM.API.LLVMBasicBlockRef, block) == entry_bb_ref
                    push!(phis_to_fix, inst)
                    break
                end
            end
        end

        isempty(phis_to_fix) && continue

        # Rebuild each affected PHI
        for old_phi in phis_to_fix
            phi_type = LLVM.value_type(old_phi)
            phi_name = LLVM.name(old_phi)

            # Collect corrected incoming edges
            inc = LLVM.incoming(old_phi)
            n = length(inc)
            edges = Tuple{LLVM.Value, LLVM.BasicBlock}[]
            for i in 1:n
                val, block = inc[i]
                if Base.unsafe_convert(LLVM.API.LLVMBasicBlockRef, block) == entry_bb_ref
                    push!(edges, (val, try_body_bb))
                else
                    push!(edges, (val, block))
                end
            end

            # Create new PHI before the old one
            @dispose builder=LLVM.IRBuilder() begin
                LLVM.position!(builder, old_phi)
                new_phi = LLVM.phi!(builder, phi_type, phi_name)
                append!(LLVM.incoming(new_phi), edges)
                LLVM.replace_uses!(old_phi, new_phi)
                LLVM.erase!(old_phi)
            end
        end
    end
end

"""
    _wrap_with_exception_handler!(fn, mod, entry_bb, entry_hook_call, name_gv, exit_ptr_val, hook_ft)

Wrap the function body in a julia.except_enter / jl_pop_handler try/catch
so that the exit hook fires on all exit paths including uncaught exceptions.

Transforms the entry block by:
1. Keeping pgcstack/safepoint setup + entry hook call in entry_bb
2. Moving remaining instructions to a new try_body block
3. Adding exception handler setup (julia.except_enter + ct->eh store)
4. Creating a catch block that fires exit hook then rethrows
5. Inserting ijl_pop_handler_noexcept before each ret's exit hook
"""
function _wrap_with_exception_handler!(fn::LLVM.Function, mod::LLVM.Module,
        entry_bb::LLVM.BasicBlock, entry_hook_call::LLVM.Instruction,
        name_gv::LLVM.Value, exit_ptr_val::LLVM.Value, hook_ft::LLVM.FunctionType)

    # Find pgcstack
    pgcstack = _find_pgcstack(entry_bb)
    if pgcstack === nothing
        @warn "TracingLLVMPlugin: Could not find pgcstack in entry block, skipping exception wrapping"
        return false
    end

    ptr_type = LLVM.PointerType()
    i8_type = LLVM.IntType(8)
    i32_type = LLVM.IntType(32)

    # Collect instructions to move (everything after entry_hook_call)
    to_move = LLVM.Instruction[]
    past_hook = false
    for inst in LLVM.instructions(entry_bb)
        if inst === entry_hook_call
            past_hook = true
            continue
        end
        if past_hook
            push!(to_move, inst)
        end
    end

    # Create new basic blocks
    try_body_bb = LLVM.BasicBlock(fn, "try_body")
    catch_bb = LLVM.BasicBlock(fn, "catch")
    # Position try_body right after entry
    LLVM.move_after(try_body_bb, entry_bb)

    # Move instructions from entry_bb to try_body_bb
    for inst in to_move
        LLVM.remove!(inst)
    end
    @dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, try_body_bb)
        for inst in to_move
            Base.insert!(builder, inst)
        end
    end

    # Fix PHI nodes in all blocks that referenced entry_bb
    _fix_phi_incoming_blocks!(fn, entry_bb, try_body_bb)

    # --- Declare runtime functions ---
    except_enter_ret_type = LLVM.StructType([i32_type, ptr_type])
    except_enter_ft = LLVM.FunctionType(except_enter_ret_type, [ptr_type])
    except_enter_fn = _get_or_declare_fn!(mod, "julia.except_enter", except_enter_ft)

    pop_handler_noexcept_ft = LLVM.FunctionType(LLVM.VoidType(), [ptr_type, i32_type])
    pop_handler_noexcept_fn = _get_or_declare_fn!(mod, "ijl_pop_handler_noexcept", pop_handler_noexcept_ft)

    pop_handler_ft = LLVM.FunctionType(LLVM.VoidType(), [ptr_type, i32_type])
    pop_handler_fn = _get_or_declare_fn!(mod, "ijl_pop_handler", pop_handler_ft)

    rethrow_ft = LLVM.FunctionType(LLVM.VoidType(), LLVM.LLVMType[])
    rethrow_fn = _get_or_declare_fn!(mod, "ijl_rethrow", rethrow_ft)

    # Build exception handler setup in entry_bb
    local current_task
    @dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, entry_bb)

        # current_task = pgcstack + task offset
        current_task = LLVM.gep!(builder, i8_type, pgcstack,
            [LLVM.ConstantInt(LLVM.IntType(64), _TASK_OFFSET_FROM_PGCSTACK[])], "current_task")

        # except = call {i32, ptr} @julia.except_enter(ptr %current_task)
        except_call = LLVM.call!(builder, except_enter_ft, except_enter_fn, [current_task], "except")
        push!(LLVM.function_attributes(except_call), LLVM.EnumAttribute("returns_twice", 0))

        # setjmp_result = extractvalue {i32, ptr} %except, 0
        setjmp_result = LLVM.extract_value!(builder, except_call, 0, "setjmp_result")

        # handler_buf = extractvalue {i32, ptr} %except, 1
        handler_buf = LLVM.extract_value!(builder, except_call, 1, "handler_buf")

        # eh_field = gep i8, pgcstack, 32  (ct->eh is at pgcstack + 32)
        eh_field = LLVM.gep!(builder, i8_type, pgcstack,
            [LLVM.ConstantInt(LLVM.IntType(64), 32)], "eh_field")

        # store ptr %handler_buf, ptr %eh_field
        LLVM.store!(builder, handler_buf, eh_field)

        # is_normal = icmp eq i32 %setjmp_result, 0
        is_normal = LLVM.icmp!(builder, LLVM.API.LLVMIntEQ, setjmp_result,
            LLVM.ConstantInt(i32_type, 0), "is_normal")

        # br i1 %is_normal, label %try_body, label %catch
        LLVM.br!(builder, is_normal, try_body_bb, catch_bb)
    end

    # Insert ijl_pop_handler_noexcept + exit_hook before each ret
    ret_insts = LLVM.Instruction[]
    for bb in LLVM.blocks(fn)
        bb === catch_bb && continue  # skip catch block
        term = LLVM.terminator(bb)
        if term !== nothing && term isa LLVM.RetInst
            push!(ret_insts, term)
        end
    end
    for ret_inst in ret_insts
        @dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, ret_inst)
            LLVM.call!(builder, pop_handler_noexcept_ft, pop_handler_noexcept_fn,
                [current_task, LLVM.ConstantInt(i32_type, 1)])
            LLVM.call!(builder, hook_ft, exit_ptr_val, [name_gv])
        end
    end

    # Build catch block
    @dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, catch_bb)
        LLVM.call!(builder, pop_handler_ft, pop_handler_fn,
            [current_task, LLVM.ConstantInt(i32_type, 1)])
        LLVM.call!(builder, hook_ft, exit_ptr_val, [name_gv])
        LLVM.call!(builder, rethrow_ft, rethrow_fn, LLVM.Value[])
        LLVM.unreachable!(builder)
    end

    return true
end

# ============================================================================
# Main instrumentation function
# ============================================================================

# Pin the current task to its thread (task.sticky = true) before the entry hook,
# so task migration can't corrupt TAU's per-thread timer stack while a timer is
# open.
function _emit_sticky_pin!(builder, pgcstack)
    i8 = LLVM.IntType(8)
    addr = LLVM.gep!(builder, i8, pgcstack,
        [LLVM.ConstantInt(LLVM.IntType(64), _STICKY_BYTE_OFFSET_FROM_PGCSTACK[])], "sticky_addr")
    LLVM.store!(builder, LLVM.ConstantInt(i8, 1), addr)
end

"""
    instrument_function!(fn::LLVM.Function, entry_hook_ptr::Ptr{Cvoid}, exit_hook_ptr::Ptr{Cvoid})

Insert entry/exit hook calls into an LLVM function with exception-safe exit hooks.
"""
function instrument_function!(fn::LLVM.Function, entry_hook_ptr::Ptr{Cvoid}, exit_hook_ptr::Ptr{Cvoid}, tau_mode::Bool=false)
    _should_instrument(fn) || return false

    # The sticky pin (TAU mode) and the exception handler's current-task gep both
    # depend on the pgcstack offsets computed in __init__.
    tau_mode && !_offsets_ready[] && error("TAUProfile: pgcstack/sticky offsets not " *
        "computed; __init__ did not run before instrumentation.")

    # Propagate Julia @noinline to LLVM noinline attribute so the optimizer
    # doesn't inline instrumented functions and eliminate their hooks
    fname = LLVM.name(fn)
    if fname in _llvm_func_noinline_set
        push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))
    end

    mod = LLVM.parent(fn)
    label = String(_build_function_label(fn))

    # Create a global string constant for the function label
    gv_name = String("trace_label_$(hash(label))")
    name_gv = LLVM.globalstring_ptr!(mod, label, gv_name)

    # Hook function type: void(ptr)
    ptr_type = LLVM.PointerType()
    hook_ft = LLVM.FunctionType(LLVM.VoidType(), [ptr_type])

    # Embed hook pointers as constants via inttoptr
    int_type = LLVM.IntType(sizeof(Ptr{Cvoid}) * 8)
    entry_ptr_int = LLVM.ConstantInt(int_type, reinterpret(UInt, entry_hook_ptr))
    entry_ptr_val = LLVM.const_inttoptr(entry_ptr_int, ptr_type)
    exit_ptr_int = LLVM.ConstantInt(int_type, reinterpret(UInt, exit_hook_ptr))
    exit_ptr_val = LLVM.const_inttoptr(exit_ptr_int, ptr_type)

    # Insert entry hook AFTER pgcstack/safepoint setup (not at function start)
    # The setup must remain in the entry block for Julia's LLVM passes
    entry_bb = first(LLVM.blocks(fn))
    setup_end = _find_setup_end(entry_bb)
    # pgcstack for the TAU-mode sticky pin (emitted right before the entry hook,
    # where pgcstack is already in scope after the safepoint setup).
    pgcstack = _find_pgcstack(entry_bb)
    local entry_hook_call
    if setup_end !== nothing
        # Find the instruction after setup_end to position before it
        found = false
        local insert_before
        for inst in LLVM.instructions(entry_bb)
            if found
                insert_before = inst
                @goto found_insert_point
            end
            if inst === setup_end
                found = true
            end
        end
        # setup_end was the last instruction — position at end of block
        @dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, entry_bb)
            tau_mode && pgcstack !== nothing && _emit_sticky_pin!(builder, pgcstack)
            entry_hook_call = LLVM.call!(builder, hook_ft, entry_ptr_val, [name_gv])
        end
        @goto done_entry_hook
        @label found_insert_point
        @dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, insert_before)
            tau_mode && pgcstack !== nothing && _emit_sticky_pin!(builder, pgcstack)
            entry_hook_call = LLVM.call!(builder, hook_ft, entry_ptr_val, [name_gv])
        end
        @label done_entry_hook
    else
        # Fallback: insert at function start
        first_inst = first(LLVM.instructions(entry_bb))
        @dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, first_inst)
            tau_mode && pgcstack !== nothing && _emit_sticky_pin!(builder, pgcstack)
            entry_hook_call = LLVM.call!(builder, hook_ft, entry_ptr_val, [name_gv])
        end
    end

    # Wrap function body in exception handler for exit hook safety
    _wrap_with_exception_handler!(fn, mod, entry_bb, entry_hook_call,
        name_gv, exit_ptr_val, hook_ft)

    return true
end

"""
    instrument_module!(mod::LLVM.Module)

Run the instrumentation pass on all eligible functions in an LLVM module.
"""
function instrument_module!(mod::LLVM.Module)
    entry_ptr, exit_ptr, tau_mode = _active_hook_ptrs()
    if entry_ptr == C_NULL || exit_ptr == C_NULL
        @warn "TracingLLVMPlugin: Hook pointers not initialized"
        return 0
    end

    count = 0
    for fn in LLVM.functions(mod)
        try
            if instrument_function!(fn, entry_ptr, exit_ptr, tau_mode)
                count += 1
            end
        catch ex
            fname = LLVM.name(fn)
            @warn "TracingLLVMPlugin: Failed to instrument $fname" exception=(ex, catch_backtrace())
        end
    end
    return count
end

# ============================================================================
# Compilation driver — GPUCompiler NativeCompilerTarget
# ============================================================================

using GPUCompiler
import Core.Compiler as CC

# --- GPUCompiler target setup ---

Base.Experimental.@MethodTable(_tracing_plugin_method_table)

struct TracingPluginParams <: GPUCompiler.AbstractCompilerParams
    method_table
    TracingPluginParams(mt=_tracing_plugin_method_table) = new(mt)
end

const TracingPluginJob = GPUCompiler.CompilerJob{GPUCompiler.NativeCompilerTarget, TracingPluginParams}

module TracingPluginRuntime end
GPUCompiler.runtime_module(::TracingPluginJob) = TracingPluginRuntime
GPUCompiler.method_table(@nospecialize(job::TracingPluginJob)) = job.config.params.method_table

# When force_noinline is on, disable Julia-level inlining for our compile job so
# that callees remain separate LLVM functions we can instrument.
function GPUCompiler.optimization_params(@nospecialize(job::TracingPluginJob))
    kwargs = NamedTuple()
    if job.config.always_inline
        kwargs = (kwargs..., inline_cost_threshold=Int(GPUCompiler.CC.MAX_INLINE_COST))
    end
    if _disable_julia_inlining[]
        kwargs = (kwargs..., inlining=false)
    end
    return GPUCompiler.CC.OptimizationParams(; kwargs...)
end

# Allow literal pointer calls.
GPUCompiler.valid_function_pointer(@nospecialize(job::TracingPluginJob), ptr::Ptr{Cvoid}) =
    ptr == _entry_hook_ptr[] || ptr == _exit_hook_ptr[] ||
    (ptr != C_NULL && (ptr == TAU_START_FPTR[] || ptr == TAU_STOP_FPTR[]))

# @cfunction closure detection

"""
    _has_cfunction_closure(src::Core.CodeInfo) -> Bool

Check if a CodeInfo contains a :cfunction expression.
"""
function _has_cfunction_closure(src::Core.CodeInfo)
    for stmt in src.code
        if isa(stmt, Expr) && stmt.head === :cfunction
            return true
        end
    end
    return false
end

"""
    _find_cfunction_cis(populated) -> Set{Core.CodeInstance}

Given a list of (CodeInstance, CodeInfo) pairs, find all CIs whose CodeInfo
contains @cfunction expressions.
"""
function _find_cfunction_cis(populated)
    tainted = Set{Core.CodeInstance}()
    for (ci, src) in populated
        _has_cfunction_closure(src) && push!(tainted, ci)
    end
    return tainted
end

# finish_module! hook

function GPUCompiler.finish_module!(job::TracingPluginJob, mod::LLVM.Module, entry::LLVM.Function)
    instrument_module!(mod)
    return entry
end

# Custom compile_method_instance with CI filtering and depth computation
#
# Overrides GPUCompiler's compile_method_instance for TracingPluginJob.
# After ci_cache_populate collects all reachable CIs, we:
# 1. Filter out CIs containing @cfunction closures
# 2. Compute depth maps via BFS over :invoke edges
# 3. Populate LLVM function metadata maps for use during instrumentation

function GPUCompiler.compile_method_instance(@nospecialize(job::TracingPluginJob))
    if job.source.def.primary_world > job.world
        error("Cannot compile $(job.source) for world $(job.world)")
    end

    interp = GPUCompiler.get_interpreter(job)
    cache = CC.code_cache(interp)
    populated = GPUCompiler.ci_cache_populate(interp, cache, job.source, job.world, job.world)

    # CI FILTER: Remove CIs containing @cfunction closures
    tainted = _find_cfunction_cis(populated)
    if !isempty(tainted)
        filter!(pair -> pair[1] ∉ tainted, populated)
    end

    # Compute depth maps from call graph
    depth_map, mod_depth_map = _compute_depth_maps(populated, job.source)

    # Lookup callback
    method_instances = Any[]
    function lookup_fun(mi, min_world, max_world)
        push!(method_instances, mi)
        GPUCompiler.ci_cache_lookup(cache, mi, min_world, max_world)
    end
    lookup_cb = @cfunction($lookup_fun, Any, (Any, UInt, UInt))

    debug_info_kind = GPUCompiler.llvm_debug_info(job)
    cgparams = (;
        track_allocations  = false,
        code_coverage      = false,
        prefer_specsig     = true,
        gnu_pubnames       = false,
        debug_info_kind    = Cint(debug_info_kind),
        safepoint_on_entry = GPUCompiler.can_safepoint(job),
        gcstack_arg        = false,
        force_emit_all     = true,
    )
    params = Base.CodegenParams(; cgparams...)

    GC.@preserve lookup_cb begin
        ts_mod = LLVM.ThreadSafeModule("start")
        ts_mod() do mod
            LLVM.triple!(mod, GPUCompiler.llvm_triple(job.config.target))
            if GPUCompiler.julia_datalayout(job.config.target) !== nothing
                LLVM.datalayout!(mod, GPUCompiler.julia_datalayout(job.config.target))
            end
            LLVM.flags(mod)["Dwarf Version", LLVM.API.LLVMModuleFlagBehaviorWarning] =
                LLVM.Metadata(LLVM.ConstantInt(GPUCompiler.dwarf_version(job.config.target)))
            LLVM.flags(mod)["Debug Info Version", LLVM.API.LLVMModuleFlagBehaviorWarning] =
                LLVM.Metadata(LLVM.ConstantInt(LLVM.DEBUG_METADATA_VERSION()))
        end

        codeinfos = Any[]
        for (ci, src) in populated
            push!(codeinfos, ci::Core.CodeInstance)
            push!(codeinfos, src::Core.CodeInfo)
        end

        native_code = @ccall jl_emit_native(
            codeinfos::Vector{Any},
            ts_mod::LLVM.API.LLVMOrcThreadSafeModuleRef,
            Ref(params)::Ptr{Base.CodegenParams},
            false::Cint
        )::Ptr{Cvoid}
        @assert native_code != C_NULL

        llvm_mod_ref = @ccall jl_get_llvm_module(
            native_code::Ptr{Cvoid}
        )::LLVM.API.LLVMOrcThreadSafeModuleRef
        @assert llvm_mod_ref != C_NULL

        llvm_ts_mod = LLVM.ThreadSafeModule(llvm_mod_ref)
        local llvm_mod
        llvm_ts_mod() do mod
            llvm_mod = mod
        end

        # Build gv_to_value map
        gv_to_value = Dict{String, Ptr{Cvoid}}()
        for gv in LLVM.globals(llvm_mod)
            if !haskey(LLVM.metadata(gv), "julia.constgv")
                continue
            end
            gv_to_value[LLVM.name(gv)] = C_NULL
            val = LLVM.initializer(gv)
            val === nothing && continue
            while isa(val, LLVM.ConstantExpr)
                op = LLVM.opcode(val)
                if op in (LLVM.API.LLVMBitCast, LLVM.API.LLVMPtrToInt,
                          LLVM.API.LLVMAddrSpaceCast, LLVM.API.LLVMIntToPtr)
                    val = LLVM.operands(val)[1]
                    continue
                end
                break
            end
            if isa(val, LLVM.ConstantInt)
                gv_to_value[LLVM.name(gv)] = reinterpret(Ptr{Cvoid}, convert(UInt, val))
            end
        end

        # Get compiled MIs via jl_get_llvm_mis
        num_mis = Ref{Csize_t}(0)
        @ccall jl_get_llvm_mis(native_code::Ptr{Cvoid}, num_mis::Ptr{Csize_t},
                               C_NULL::Ptr{Cvoid})::Nothing
        resize!(method_instances, num_mis[])
        @ccall jl_get_llvm_mis(native_code::Ptr{Cvoid}, num_mis::Ptr{Csize_t},
                               method_instances::Ptr{Cvoid})::Nothing

        code_instances = Core.CodeInstance[]
        for mi in method_instances
            ci = GPUCompiler.ci_cache_lookup(cache, mi, job.world, job.world)
            ci === nothing && continue
            llvm_func_idx = Ref{Int32}(-1)
            llvm_specfunc_idx = Ref{Int32}(-1)
            ccall(:jl_get_function_id, Nothing,
                  (Ptr{Cvoid}, Any, Ptr{Int32}, Ptr{Int32}),
                  native_code, ci, llvm_func_idx, llvm_specfunc_idx)
            llvm_func_idx[] == -1 && continue
            push!(code_instances, ci)
        end
        unique!(code_instances)

        resize!(method_instances, length(code_instances))
        for (i, ci) in enumerate(code_instances)
            method_instances[i] = ci.def::Core.MethodInstance
        end

        compiled = Dict()
        for (ci, mi) in zip(code_instances, method_instances)
            llvm_func_idx = Ref{Int32}(-1)
            llvm_specfunc_idx = Ref{Int32}(-1)
            ccall(:jl_get_function_id, Nothing,
                  (Ptr{Cvoid}, Any, Ptr{Int32}, Ptr{Int32}),
                  native_code, ci, llvm_func_idx, llvm_specfunc_idx)
            @assert llvm_func_idx[] != -1 || llvm_specfunc_idx[] != -1

            llvm_func = if llvm_func_idx[] >= 1
                ref = ccall(:jl_get_llvm_function, LLVM.API.LLVMValueRef,
                            (Ptr{Cvoid}, UInt32), native_code, llvm_func_idx[]-1)
                @assert ref != C_NULL
                LLVM.name(LLVM.Function(ref))
            else
                nothing
            end

            llvm_specfunc = if llvm_specfunc_idx[] >= 1
                ref = ccall(:jl_get_llvm_function, LLVM.API.LLVMValueRef,
                            (Ptr{Cvoid}, UInt32), native_code, llvm_specfunc_idx[]-1)
                @assert ref != C_NULL
                LLVM.name(LLVM.Function(ref))
            else
                nothing
            end

            haskey(compiled, mi) && continue
            compiled[mi] = (; ci, func=llvm_func, specfunc=llvm_specfunc)
        end

        @assert haskey(compiled, job.source) "Entry function $(job.source) not in compiled output"

        # Populate LLVM function metadata maps for use during finish_module!
        empty!(_llvm_func_module_map)
        empty!(_llvm_func_depth_map)
        empty!(_llvm_func_mi_map)
        empty!(_llvm_func_argtypes_map)
        empty!(_llvm_func_noinline_set)
        empty!(_mi_info_fallback_map)

        # Build fallback map from populated CIs
        for (ci, _src) in populated
            mi = ci.def
            isa(mi, Core.MethodInstance) || continue
            method = mi.def
            isa(method, Method) || continue
            key = (String(method.name), String(method.file), Int(method.line))
            haskey(_mi_info_fallback_map, key) && continue
            _mi_info_fallback_map[key] = (string(method.module), mi)
        end

        # Build MI -> noinline map from CodeInfo.inlining
        noinline_mis = IdDict{Core.MethodInstance, Bool}()
        for (ci, src) in populated
            if isdefined(src, :inlining) && src.inlining == 0x02
                noinline_mis[ci.def::Core.MethodInstance] = true
            end
        end

        for (mi, info) in compiled
            method = mi.def
            depth = get(depth_map, mi, 0)
            mod_depth = get(mod_depth_map, mi, 0)

            mod_name = isa(method, Method) ? string(method.module) : nothing
            argtypes_str = _include_types[] ? _format_argtypes(mi) : nothing
            is_noinline = haskey(noinline_mis, mi)

            # Add module to Phase 2 tracking set
            if isa(method, Method)
                push!(_phase2_modules, method.module)
            end

            for llvm_name in (info.func, info.specfunc)
                llvm_name === nothing && continue
                # Store under both raw and sanitized names.
                sanitized = replace(llvm_name, r"[^A-Za-z0-9]"=>"_")
                for key in (llvm_name, sanitized)
                    if mod_name !== nothing
                        _llvm_func_module_map[key] = mod_name
                    end
                    _llvm_func_depth_map[key] = (depth, mod_depth)
                    _llvm_func_mi_map[key] = mi
                    if argtypes_str !== nothing
                        _llvm_func_argtypes_map[key] = argtypes_str
                    end
                    if is_noinline
                        push!(_llvm_func_noinline_set, key)
                    end
                end
            end
        end

        # Second pass: fill metadata gaps for LLVM functions not in `compiled`.
        for fn in LLVM.functions(llvm_mod)
            llvm_name = LLVM.name(fn)
            haskey(_llvm_func_module_map, llvm_name) && continue
            LLVM.isdeclaration(fn) && continue
            sp = LLVM.subprogram(fn)
            sp === nothing && continue
            funcname = LLVM.name(sp)
            (funcname === nothing || isempty(funcname)) && continue
            di_file = LLVM.file(sp)
            filename = LLVM.filename(di_file)
            (filename === nothing || isempty(filename) || filename == "none") && continue
            line_num = LLVM.line(sp)
            key = (funcname, basename(filename), line_num)
            info = get(_mi_info_fallback_map, key, nothing)
            info === nothing && continue
            mod_name, mi = info
            _llvm_func_module_map[llvm_name] = mod_name
            _llvm_func_mi_map[llvm_name] = mi
            _llvm_func_depth_map[llvm_name] = (get(depth_map, mi, 0), get(mod_depth_map, mi, 0))
            if _include_types[]
                _llvm_func_argtypes_map[llvm_name] = _format_argtypes(mi)
            end
        end

        return llvm_mod, compiled, gv_to_value
    end
end

# --- Phase 2 Single-Function Emission ---

"""
    _get_ci_for_mi(mi::Core.MethodInstance) -> Union{Core.CodeInstance, Nothing}

Retrieve the first CodeInstance from the cache linked list of a MethodInstance.
"""
function _get_ci_for_mi(mi::Core.MethodInstance)
    isdefined(mi, :cache) || return nothing
    ci = mi.cache
    while ci !== nothing
        ci.owner === nothing && return ci
        isdefined(ci, :next) || return nothing
        ci = ci.next
    end
    return nothing
end

"""
    _has_active_llvm_context() -> Bool

Detect whether an LLVM context is already active on the current task.
"""
function _has_active_llvm_context()
    return haskey(task_local_storage(), :LLVMContext) &&
           !isempty(task_local_storage(:LLVMContext))
end

"""
    _emit_single_function(ci::Core.CodeInstance, src::Core.CodeInfo) -> Union{NamedTuple, Nothing}

Emit a single CodeInstance/CodeInfo pair to an LLVM module using Julia's `jl_emit_native`.
"""
function _emit_single_function(ci::Core.CodeInstance, src::Core.CodeInfo)
    # Input validation
    ci isa Core.CodeInstance || return nothing
    src isa Core.CodeInfo || return nothing

    function _do_emit(ctx)
        # Create CodegenParams with Phase 2 settings
        cgparams = (;
            track_allocations  = false,
            code_coverage      = false,
            prefer_specsig     = true,
            gnu_pubnames       = false,
            debug_info_kind    = Cint(LLVM.API.LLVMDWARFSourceLanguageJulia),
            safepoint_on_entry = true,
            gcstack_arg        = false,
            force_emit_all     = true,
        )
        params = Base.CodegenParams(; cgparams...)

        # Build single-item codeinfos vector
        codeinfos = Any[ci::Core.CodeInstance, src::Core.CodeInfo]

        # Create ThreadSafeModule for emission
        ts_mod = LLVM.ThreadSafeModule("phase2_emit")

        GC.@preserve codeinfos begin
            # Initialize module with basic properties
            ts_mod() do mod
                LLVM.triple!(mod, Sys.MACHINE)
                LLVM.flags(mod)["Dwarf Version", LLVM.API.LLVMModuleFlagBehaviorWarning] =
                    LLVM.Metadata(LLVM.ConstantInt(4))
                LLVM.flags(mod)["Debug Info Version", LLVM.API.LLVMModuleFlagBehaviorWarning] =
                    LLVM.Metadata(LLVM.ConstantInt(LLVM.DEBUG_METADATA_VERSION()))
            end

            # Emit native code via jl_emit_native
            native_code = @ccall jl_emit_native(
                codeinfos::Vector{Any},
                ts_mod::LLVM.API.LLVMOrcThreadSafeModuleRef,
                Ref(params)::Ptr{Base.CodegenParams},
                false::Cint
            )::Ptr{Cvoid}

            native_code == C_NULL && return nothing

            # Extract LLVM module
            llvm_mod_ref = @ccall jl_get_llvm_module(
                native_code::Ptr{Cvoid}
            )::LLVM.API.LLVMOrcThreadSafeModuleRef

            llvm_mod_ref == C_NULL && return nothing

            llvm_ts_mod = LLVM.ThreadSafeModule(llvm_mod_ref)
            local llvm_mod
            llvm_ts_mod() do mod
                llvm_mod = mod
            end

            # Extract function metadata via jl_get_function_id / jl_get_llvm_function
            # Build array of compiled MIs first
            method_instances = Any[]
            num_mis = Ref{Csize_t}(0)
            @ccall jl_get_llvm_mis(native_code::Ptr{Cvoid}, num_mis::Ptr{Csize_t},
                                   C_NULL::Ptr{Cvoid})::Nothing
            resize!(method_instances, num_mis[])
            @ccall jl_get_llvm_mis(native_code::Ptr{Cvoid}, num_mis::Ptr{Csize_t},
                                   method_instances::Ptr{Cvoid})::Nothing

            # Look up the input CI in the compiled set
            func_name = nothing
            specfunc_name = nothing

            for mi in method_instances
                mi_ci = ci  # We're only looking for our input CI

                llvm_func_idx = Ref{Int32}(-1)
                llvm_specfunc_idx = Ref{Int32}(-1)
                ccall(:jl_get_function_id, Nothing,
                      (Ptr{Cvoid}, Any, Ptr{Int32}, Ptr{Int32}),
                      native_code, mi_ci, llvm_func_idx, llvm_specfunc_idx)

                if llvm_func_idx[] >= 1
                    ref = ccall(:jl_get_llvm_function, LLVM.API.LLVMValueRef,
                                (Ptr{Cvoid}, UInt32), native_code, llvm_func_idx[]-1)
                    ref != C_NULL && (func_name = LLVM.name(LLVM.Function(ref)))
                end

                if llvm_specfunc_idx[] >= 1
                    ref = ccall(:jl_get_llvm_function, LLVM.API.LLVMValueRef,
                                (Ptr{Cvoid}, UInt32), native_code, llvm_specfunc_idx[]-1)
                    ref != C_NULL && (specfunc_name = LLVM.name(LLVM.Function(ref)))
                end

                (func_name !== nothing || specfunc_name !== nothing) && break
            end

            # Return result tuple
            # Caller must keep ts_mod alive while using mod
            return (;
                mod = llvm_mod,
                ts_mod = llvm_ts_mod,
                func_name = func_name,
                specfunc_name = specfunc_name,
                native_code = native_code,
            )
        end
    end

    if _has_active_llvm_context()
        ctx = LLVM.context()
        return _do_emit(ctx)
    else
        return GPUCompiler.JuliaContext() do ctx
            _do_emit(ctx)
        end
    end
end

"""
    _instrument_single_function!(emit_result, mi::Core.MethodInstance) -> Int

Instrument a single-function LLVM module produced by `_emit_single_function`.

Takes the output of `_emit_single_function` and instruments the LLVM module by:
1. Populating the LLVM function metadata maps for the target function
2. Calling `instrument_module!` to insert entry/exit hooks
"""
function _instrument_single_function!(emit_result, mi::Core.MethodInstance)
    method = mi.def
    isa(method, Method) || return 0

    mod_name = string(method.module)

    # Populate metadata maps for both func_name and specfunc_name
    for llvm_name in (emit_result.func_name, emit_result.specfunc_name)
        llvm_name === nothing && continue

        # Map LLVM name to module for label building
        _llvm_func_module_map[llvm_name] = mod_name

        # Map LLVM name to MI for exclusion checks
        _llvm_func_mi_map[llvm_name] = mi

        # Map LLVM name to argument types if enabled
        if _include_types[]
            _llvm_func_argtypes_map[llvm_name] = _format_argtypes(mi)
        end

        # No depth tracking for Phase 2
        _llvm_func_depth_map[llvm_name] = (0, 0)
    end

    # Fill metadata gaps for any other functions in the module (ABI thunks, etc.)
    for fn in LLVM.functions(emit_result.mod)
        fn_name = LLVM.name(fn)
        haskey(_llvm_func_module_map, fn_name) && continue
        LLVM.isdeclaration(fn) && continue
        _llvm_func_module_map[fn_name] = mod_name
        _llvm_func_mi_map[fn_name] = mi
        _llvm_func_depth_map[fn_name] = (0, 0)
        if _include_types[]
            _llvm_func_argtypes_map[fn_name] = _format_argtypes(mi)
        end
    end

    # Instrument the module using the standard pipeline
    count = instrument_module!(emit_result.mod)
    return count
end

"""
    _jit_and_lookup(emit_result) -> Ptr{Cvoid}

Compile an instrumented LLVM module through JuliaOJIT and look up the function pointer.

Takes the output of `_emit_single_function` (which contains `ts_mod`, `func_name`,
`specfunc_name`) and:

1. Gets Julia's global ORC JIT using @dispose (matching tau_rewrite_and_call pattern)
2. Gets the main JITDylib
3. Sets up process-wide symbol resolution (so callees resolve from Julia's existing
   compiled code)
4. Adds the instrumented LLVM module to the JIT
5. Looks up the compiled function pointer
6. Returns the function pointer as `Ptr{Cvoid}`
"""
function _jit_and_lookup(emit_result)
    lookup_name = emit_result.func_name
    if lookup_name === nothing
        lookup_name = emit_result.specfunc_name
    end
    lookup_name === nothing && return nothing

    @dispose jljit=LLVM.JuliaOJIT() begin
        jd = LLVM.JITDylib(jljit)
        prefix = LLVM.get_prefix(jljit)
        dg = LLVM.CreateDynamicLibrarySearchGeneratorForProcess(prefix)
        LLVM.add!(jd, dg)

        # Add the instrumented module to the JIT
        LLVM.add!(jljit, jd, emit_result.ts_mod)

        # Lookup the function pointer
        addr = LLVM.lookup(jljit, lookup_name)
        fptr = pointer(addr)
        return fptr
    end
end

"""
    _lower_julia_intrinsics!(mod::LLVM.Module)

Run Julia intrinsic lowering passes on an LLVM module emitted by `jl_emit_native`.

This is the minimal subset of GPUCompiler's `buildIntrinsicLoweringPipeline` needed to
eliminate Julia pseudo-intrinsics (`julia.safepoint`, `julia.get_pgcstack`,
`julia.except_enter`, etc.) without running optimization passes (InstCombine hangs on
instrumented IR).
"""
function _lower_julia_intrinsics!(mod::LLVM.Module)
    tm = LLVM.JITTargetMachine()

    @dispose pb=LLVM.NewPMPassBuilder() begin
        LLVM.register!(pb, GPUCompiler.GPULowerCPUFeaturesPass())
        LLVM.register!(pb, GPUCompiler.GPULowerPTLSPass())
        LLVM.register!(pb, GPUCompiler.GPULowerGCFramePass())

        LLVM.add!(pb, LLVM.NewPMModulePassManager()) do mpm
            LLVM.add!(mpm, GPUCompiler.RemoveNIPass())

            LLVM.add!(mpm, LLVM.NewPMFunctionPassManager()) do fpm
                if VERSION < v"1.13.0-DEV.36"
                    LLVM.add!(fpm, GPUCompiler.LowerExcHandlersPass())
                end
                LLVM.add!(fpm, GPUCompiler.GCInvariantVerifierPass())
                LLVM.add!(fpm, GPUCompiler.LateLowerGCPass())
                if VERSION >= v"1.11.0-DEV.208"
                    LLVM.add!(fpm, GPUCompiler.FinalLowerGCPass())
                end
            end

            LLVM.add!(mpm, GPUCompiler.LowerPTLSPass())
            LLVM.add!(mpm, GPUCompiler.RemoveJuliaAddrspacesPass())
            LLVM.add!(mpm, LLVM.GlobalDCEPass())
        end

        LLVM.run!(pb, mod, tm)
    end

    return nothing
end

# Compute field offsets once at module init
const _ci_invoke_offset = let
    idx = findfirst(==(:invoke), fieldnames(Core.CodeInstance))
    idx === nothing ? error("CodeInstance has no :invoke field") : fieldoffset(Core.CodeInstance, idx)
end

const _ci_specptr_offset = let
    idx = findfirst(==(:specptr), fieldnames(Core.CodeInstance))
    idx === nothing ? error("CodeInstance has no :specptr field") : fieldoffset(Core.CodeInstance, idx)
end

const _ci_specsigflags_offset = let
    idx = findfirst(==(:specsigflags), fieldnames(Core.CodeInstance))
    idx === nothing ? error("CodeInstance has no :specsigflags field") : fieldoffset(Core.CodeInstance, idx)
end

# jl_fptr_args function pointer (the generic ABI dispatcher) and
# jl_fptr_const_return function pointer (used for const-return detection).
const _jl_fptr_args = Ref{Ptr{Cvoid}}(C_NULL)
const _jl_fptr_const_return = Ref{Ptr{Cvoid}}(C_NULL)

"""
    _replace_ci_fptr!(ci::Core.CodeInstance, fptr::Ptr{Cvoid})

Replace a CodeInstance's function pointers to route through instrumented code.
"""
function _replace_ci_fptr!(ci::Core.CodeInstance, fptr::Ptr{Cvoid})
    ci_ptr = Base.pointer_from_objref(ci)

    # 1. Store instrumented function pointer in specptr.fptr1
    #    (generic ABI signature: jl_value_t*(jl_value_t*, jl_value_t**, uint32_t))
    unsafe_store!(Ptr{Ptr{Cvoid}}(ci_ptr + _ci_specptr_offset), fptr)

    # 2. Set invoke to jl_fptr_args (generic ABI dispatcher that reads specptr.fptr1)
    unsafe_store!(Ptr{Ptr{Cvoid}}(ci_ptr + _ci_invoke_offset), _jl_fptr_args[])

    # 3. Set specsigflags to 0x02 (bit 1 = invoke ready, bit 0 = 0 = no specsig)
    unsafe_store!(Ptr{UInt8}(ci_ptr + _ci_specsigflags_offset), 0x02)

    nothing
end

"""
    _is_const_return(ci::Core.CodeInstance) -> Bool

Detect if a CodeInstance is a const-return case.

Returns true if the CI's invoke function is `jl_fptr_const_return`.
"""
function _is_const_return(ci::Core.CodeInstance)
    ci_ptr = Base.pointer_from_objref(ci)
    invoke = unsafe_load(Ptr{Ptr{Cvoid}}(ci_ptr + _ci_invoke_offset))
    return invoke == _jl_fptr_const_return[]
end

# ============================================================================
# Phase 2 Dynamic Dispatch: jl_set_typeinf_func Replacement
# ============================================================================

"""
    _default_compile_phase2(mi::Core.MethodInstance, world::UInt, source_mode::UInt8, trim_mode::UInt8)

Standard compilation using NativeInterpreter (zero-overhead inference).
Used as fallback when Phase 2 is inactive.
"""
function _default_compile_phase2(mi::Core.MethodInstance, world::UInt, source_mode::UInt8, trim_mode::UInt8)
    inf_params = CC.InferenceParams(; force_enable_inference = trim_mode != CC.TRIM_NO)
    interp = CC.NativeInterpreter(world; inf_params)
    return CC.typeinf_ext_toplevel(interp, mi, source_mode)
end

"""
    _phase2_lowered_src(mi::Core.MethodInstance, world::UInt) -> Union{Core.CodeInfo, Nothing}

Retrieve the lowered `CodeInfo` to feed into `jl_emit_native` for Phase 2.
"""
function _phase2_lowered_src(mi::Core.MethodInstance, world::UInt)
    m = mi.def
    isa(m, Method) || return nothing
    if Base.hasgenerator(m)
        Base.may_invoke_generator(mi) || return nothing
        return ccall(:jl_code_for_staged, Ref{Core.CodeInfo},
                     (Any, UInt, Ptr{Cvoid}), mi, world, C_NULL)
    end
    return Base.uncompressed_ir(m)
end

"""
    _tracing_llvm_typeinf_toplevel(mi::Core.MethodInstance, world::UInt, source_mode::UInt8, trim_mode::UInt8)

Replacement for CC.typeinf_ext_toplevel installed via jl_set_typeinf_func.

When Phase 2 is active and the MI passes the filter:
1. Performs standard inference via NativeInterpreter
2. Checks for @cfunction closures (CUDA safety)
3. Emits single-function LLVM module
4. Instruments the module with entry/exit hooks
5. Lowers Julia pseudo-intrinsics
6. JIT compiles the instrumented module
7. Replaces the CI's function pointer with the instrumented version
"""
function _tracing_llvm_typeinf_toplevel(mi::Core.MethodInstance, world::UInt,
                                        source_mode::UInt8, trim_mode::UInt8)
    # Fast-path exits
    _phase2_active[] || return _default_compile_phase2(mi, world, source_mode, trim_mode)
    _is_phase2_reentrant() && return _default_compile_phase2(mi, world, source_mode, trim_mode)

    # Check Phase 2 filter (module scoping, exclusions, Base/Core skip)
    if !_passes_phase2_filter(mi)
        return _default_compile_phase2(mi, world, source_mode, trim_mode)
    end

    # Standard compilation first
    ci = _default_compile_phase2(mi, world, source_mode, trim_mode)

    # Skip const-return functions
    _is_const_return(ci) && return ci

    _set_phase2_reentrant!(true)
    try
        # Get source CodeInfo for the method.
        src = _phase2_lowered_src(mi, world)
        src === nothing && return ci

        # Safety: skip functions with @cfunction closures
        _has_cfunction_closure(src) && return ci

        function _run_phase2_pipeline!(ci, src, mi)
            emit_result = _emit_single_function(ci, src)
            emit_result === nothing && return nothing

            count = _instrument_single_function!(emit_result, mi)
            count == 0 && return nothing

            _lower_julia_intrinsics!(emit_result.mod)

            fptr = _jit_and_lookup(emit_result)
            (fptr === nothing || fptr == C_NULL) && return nothing

            _replace_ci_fptr!(ci, fptr)
            return fptr
        end

        if _has_active_llvm_context()
            _run_phase2_pipeline!(ci, src, mi)
        else
            GPUCompiler.JuliaContext() do ctx
                _run_phase2_pipeline!(ci, src, mi)
            end
        end
    catch ex
        @warn "Phase 2 instrumentation failed" exception=(ex, catch_backtrace())
    finally
        _set_phase2_reentrant!(false)
    end
    return ci
end

# ============================================================================
# Phase 2 Dry-Run Support
# ============================================================================

@noinline _phase2_dry_run_target(x::Int) = x + 1

"""
    _dry_run_phase2_pipeline!()

Perform a dry-run of the full Phase 2 pipeline on a trivial function to force
compilation of all LLVM.jl and GPUCompiler codepaths before world-age freeze.
"""
function _dry_run_phase2_pipeline!()
    try
        mi = GPUCompiler.methodinstance(typeof(_phase2_dry_run_target), Tuple{Int}, Base.get_world_counter())
        ci_result = CC.typeinf_ext_toplevel(
            CC.NativeInterpreter(Base.get_world_counter()),
            mi, UInt8(0))

        ci_result === nothing && return nothing

        src = Base.uncompressed_ir(mi.def)
        src === nothing && return nothing

        GPUCompiler.JuliaContext() do ctx
            emit_result = _emit_single_function(ci_result, src)
            emit_result === nothing && return nothing

            _instrument_single_function!(emit_result, mi)
            _lower_julia_intrinsics!(emit_result.mod)

            fptr = _jit_and_lookup(emit_result)
        end
    catch ex
        @warn "Phase 2 dry-run failed (non-fatal)" exception=(ex, catch_backtrace())
    end

    return nothing
end

"""
    _precompile_phase2!()

Precompile all Phase 2 codepath functions before world age freeze.
This ensures all LLVM.jl machinery is compiled before jl_set_typeinf_func is called.
"""
function _precompile_phase2!()
    # Core replacement function
    precompile(_tracing_llvm_typeinf_toplevel, (Core.MethodInstance, UInt, UInt8, UInt8))
    precompile(_default_compile_phase2, (Core.MethodInstance, UInt, UInt8, UInt8))

    # Reentrancy guard
    precompile(_is_phase2_reentrant, ())
    precompile(_set_phase2_reentrant!, (Bool,))

    # Phase 2 filter
    precompile(_passes_phase2_filter, (Core.MethodInstance,))
    precompile(_is_phase2_module, (Module,))
    precompile(_is_base_or_core, (Module,))

    # Emission and instrumentation pipeline
    precompile(_emit_single_function, (Core.CodeInstance, Core.CodeInfo))
    precompile(_instrument_single_function!, (NamedTuple, Core.MethodInstance))
    precompile(_lower_julia_intrinsics!, (LLVM.Module,))
    precompile(_jit_and_lookup, (NamedTuple,))
    precompile(_replace_ci_fptr!, (Core.CodeInstance, Ptr{Cvoid}))

    # Existing functions used in pipeline
    precompile(_has_cfunction_closure, (Core.CodeInfo,))

    _dry_run_phase2_pipeline!()

    nothing
end

"""
    _install_phase2!()

Install Phase 2 dynamic dispatch instrumentation.
Precompiles all Phase 2 functions and installs the jl_set_typeinf_func replacement.
"""
function _install_phase2!()
    _precompile_phase2!()
    _phase2_active[] = true
    ccall(:jl_set_typeinf_func, Cvoid, (Any,), _tracing_llvm_typeinf_toplevel)
    nothing
end

"""
    _uninstall_phase2!()

Uninstall Phase 2 dynamic dispatch instrumentation.
Restores the default CC.typeinf_ext_toplevel.
"""
function _uninstall_phase2!()
    if _phase2_active[]
        _phase2_active[] = false
        ccall(:jl_set_typeinf_func, Cvoid, (Any,), CC.typeinf_ext_toplevel)
    end
    nothing
end

# --- Compilation helpers ---

function _compile_traced(@nospecialize(f), @nospecialize(tt::Type))
    mi = GPUCompiler.methodinstance(typeof(f), tt, Base.get_world_counter())
    target = GPUCompiler.NativeCompilerTarget(; jlruntime=true)
    params = TracingPluginParams()
    config = GPUCompiler.CompilerConfig(target, params; kernel=false, validate=false)
    job = GPUCompiler.CompilerJob(mi, config)
    GPUCompiler.JuliaContext() do ctx
        ir, meta = GPUCompiler.compile(:llvm, job)
        return ir, meta
    end
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
    tt = Tuple{map(typeof, args)...}
    mi = GPUCompiler.methodinstance(typeof(f), tt, Base.get_world_counter())
    target = GPUCompiler.NativeCompilerTarget(; jlruntime=true)
    params = TracingPluginParams()
    config = GPUCompiler.CompilerConfig(target, params; kernel=false, entry_abi=:func, validate=false)
    job = GPUCompiler.CompilerJob(mi, config)

    GPUCompiler.JuliaContext() do ctx
        ir, meta = GPUCompiler.compile(:llvm, job)

        entry_fn = meta.entry
        func_name = LLVM.name(entry_fn)
        LLVM.linkage!(entry_fn, LLVM.API.LLVMExternalLinkage)

        @dispose jljit=LLVM.JuliaOJIT() begin
            jd = LLVM.JITDylib(jljit)
            prefix = LLVM.get_prefix(jljit)
            dg = LLVM.CreateDynamicLibrarySearchGeneratorForProcess(prefix)
            LLVM.add!(jd, dg)

            tsm = LLVM.ThreadSafeModule(ir)
            LLVM.add!(jljit, jd, tsm)

            addr = LLVM.lookup(jljit, func_name)
            fptr = pointer(addr)

            _install_phase2!()
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

end # module TAUProfile
