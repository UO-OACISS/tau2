# ============================================================================
# phase1.jl — GPUCompiler driver for static (Phase 1) instrumentation
#
# Part of the TAUProfile module (LLVM backend).
# ============================================================================

# The compile_method_instance override below is derived from GPUCompiler.jl
# under the MIT Expat License; see full license in TAUProfile.jl.

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

    codeinfos = Any[]
    for (ci, src) in populated
        push!(codeinfos, ci::Core.CodeInstance)
        push!(codeinfos, src::Core.CodeInfo)
    end

    emitted = _emit_native(codeinfos, params;
                           name = "start",
                           triple = GPUCompiler.llvm_triple(job.config.target),
                           datalayout = GPUCompiler.julia_datalayout(job.config.target),
                           dwarf_version = GPUCompiler.dwarf_version(job.config.target))
    @assert emitted !== nothing "jl_emit_native produced no module for $(job.source)"
    llvm_mod = emitted.mod
    native_code = emitted.native_code

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

    # Map each compiled MI to its CI and LLVM function names. A CI with no
    # generic-ABI entry (`func`) was not compiled into this module.
    code_instances = Core.CodeInstance[]
    for mi in emitted.method_instances
        ci = GPUCompiler.ci_cache_lookup(cache, mi, job.world, job.world)
        ci === nothing && continue
        func, _ = _llvm_names_for_ci(native_code, ci)
        func === nothing && continue
        push!(code_instances, ci)
    end
    unique!(code_instances)

    compiled = Dict()
    for ci in code_instances
        mi = ci.def::Core.MethodInstance
        haskey(compiled, mi) && continue
        func, specfunc = _llvm_names_for_ci(native_code, ci)
        compiled[mi] = (; ci, func, specfunc)
    end

    @assert haskey(compiled, job.source) "Entry function $(job.source) not in compiled output"

    # Populate LLVM function metadata maps for use during finish_module!
    _reset_metadata!()

    # Build fallback map from populated CIs
    for (ci, _src) in populated
        mi = ci.def
        isa(mi, Core.MethodInstance) || continue
        method = mi.def
        isa(method, Method) || continue
        key = (String(method.name), String(method.file), Int(method.line))
        haskey(_mi_info_fallback_map, key) && continue
        _mi_info_fallback_map[key] = mi
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
        depth = (get(depth_map, mi, 0), get(mod_depth_map, mi, 0))
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
                _register_llvm_function!(key, mi; depth, noinline=is_noinline)
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
        mi = get(_mi_info_fallback_map, key, nothing)
        mi === nothing && continue
        _register_llvm_function!(llvm_name, mi;
                                 depth=(get(depth_map, mi, 0), get(mod_depth_map, mi, 0)))
    end

    return llvm_mod, compiled, gv_to_value
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
