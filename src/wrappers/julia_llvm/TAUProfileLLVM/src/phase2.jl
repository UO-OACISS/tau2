# ============================================================================
# phase2.jl — runtime-dispatch (Phase 2) instrumentation via the typeinf hook
#
# Part of the TAUProfile module (LLVM backend).
# ============================================================================

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
    # Time the Phase 2 rewrite (started after the reentrancy guard so compiling the
    # timer helpers can't recurse into this pipeline). Excludes the standard
    # compilation above, which happens regardless of instrumentation.
    _p2_timer = _tau_start_phase2()
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
        # Stop before clearing the guard so any compilation triggered by the stop
        # can't recurse into the Phase 2 pipeline.
        _tau_stop_phase2(_p2_timer)
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

    # Phase 2 rewrite timing helpers (called from the typeinf hook post-freeze)
    precompile(_tau_start_phase2, ())
    precompile(_tau_stop_phase2, (Ptr{Cvoid},))

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

# Phase 2 half of module initialization: resolve the jl_fptr_* dispatcher
# addresses used by _replace_ci_fptr! and _is_const_return. Called from
# __init__ in TAUProfile.jl.
function _init_phase2!()
    _jl_fptr_args[] = unsafe_load(cglobal(:jl_fptr_args_addr, Ptr{Cvoid}))
    _jl_fptr_const_return[] = unsafe_load(cglobal(:jl_fptr_const_return_addr, Ptr{Cvoid}))
end
