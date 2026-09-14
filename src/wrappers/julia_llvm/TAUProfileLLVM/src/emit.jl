# ============================================================================
# emit.jl — jl_emit_native wrapper shared by the Phase 1 driver and Phase 2
#
# Part of the TAUProfile module (LLVM backend).
# ============================================================================

"""
    _emit_native(codeinfos, params; name, triple, datalayout, dwarf_version)
        -> Union{NamedTuple, Nothing}

Emit `codeinfos` (alternating CodeInstance/CodeInfo pairs) into a fresh
ThreadSafeModule via `jl_emit_native` and unwrap the result. Returns
`(; mod, ts_mod, native_code, code_instances, gv_to_value)`, or `nothing` if
Julia produced no module. The caller must keep `ts_mod` alive while using `mod`.
"""
function _emit_native(codeinfos::Vector{Any}, params::Base.CodegenParams;
                      name::String, triple::String,
                      datalayout::Union{LLVM.DataLayout, String, Nothing}, dwarf_version::Integer)
    ts_mod = LLVM.ThreadSafeModule(name)
    ts_mod() do mod
        LLVM.triple!(mod, triple)
        if datalayout !== nothing
            LLVM.datalayout!(mod, datalayout)
        end
        LLVM.flags(mod)["Dwarf Version", LLVM.API.LLVMModuleFlagBehaviorWarning] =
            LLVM.Metadata(LLVM.ConstantInt(dwarf_version))
        LLVM.flags(mod)["Debug Info Version", LLVM.API.LLVMModuleFlagBehaviorWarning] =
            LLVM.Metadata(LLVM.ConstantInt(LLVM.DEBUG_METADATA_VERSION()))
    end

    native_code = GC.@preserve codeinfos begin
        @ccall jl_emit_native(
            codeinfos::Vector{Any},
            ts_mod::LLVM.API.LLVMOrcThreadSafeModuleRef,
            Ref(params)::Ptr{Base.CodegenParams},
            false::Cint
        )::Ptr{Cvoid}
    end
    native_code == C_NULL && return nothing

    llvm_mod_ref = @ccall jl_get_llvm_module(
        native_code::Ptr{Cvoid}
    )::LLVM.API.LLVMOrcThreadSafeModuleRef
    llvm_mod_ref == C_NULL && return nothing

    llvm_ts_mod = LLVM.ThreadSafeModule(llvm_mod_ref)
    local llvm_mod
    llvm_ts_mod() do mod
        llvm_mod = mod
    end

    code_instances = Core.CodeInstance[codeinfos[i]::Core.CodeInstance
                                       for i in 1:2:length(codeinfos)]

    gv_to_value = _resolve_constant_globals!(llvm_mod, native_code)

    return (; mod = llvm_mod, ts_mod = llvm_ts_mod, native_code, code_instances, gv_to_value)
end

# Julia 1.13.0-DEV.623 (JuliaLang/julia#58423) stopped initializing `julia.constgv`
# globals in the module `jl_emit_native` produces; these are instead given
# separately by `jl_get_llvm_gvs` (the managed globals) and `jl_get_llvm_gv_inits`
const _JULIA_INITIALIZES_CONSTGV = VERSION < v"1.13.0-DEV.623"

"""
    _resolve_constant_globals!(mod, native_code) -> Dict{String, Ptr{Cvoid}}

Return the name -> object pointer map of the module's Julia-managed constant
globals, initializing each global with its pointer where Julia left it null.
"""
function _resolve_constant_globals!(mod::LLVM.Module, native_code::Ptr{Cvoid})
    gv_to_value = Dict{String, Ptr{Cvoid}}()
    if _JULIA_INITIALIZES_CONSTGV
        for gv in LLVM.globals(mod)
            haskey(LLVM.metadata(gv), "julia.constgv") || continue
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
    else
        num = Ref{Csize_t}(0)
        @ccall jl_get_llvm_gvs(native_code::Ptr{Cvoid}, num::Ptr{Csize_t},
                               C_NULL::Ptr{Cvoid})::Nothing
        gvs = Vector{Ptr{LLVM.API.LLVMOpaqueValue}}(undef, num[])
        @ccall jl_get_llvm_gvs(native_code::Ptr{Cvoid}, num::Ptr{Csize_t},
                               gvs::Ptr{LLVM.API.LLVMOpaqueValue})::Nothing
        inits = Vector{Ptr{Cvoid}}(undef, num[])
        @ccall jl_get_llvm_gv_inits(native_code::Ptr{Cvoid}, num::Ptr{Csize_t},
                                    inits::Ptr{Cvoid})::Nothing
        for (ref, init) in zip(gvs, inits)
            gv = LLVM.GlobalVariable(ref)
            gv_to_value[LLVM.name(gv)] = init
            cur = LLVM.initializer(gv)
            if cur === nothing || LLVM.isnull(cur)
                ty = cur === nothing ? LLVM.global_value_type(gv) : LLVM.value_type(cur)
                LLVM.initializer!(gv, LLVM.const_inttoptr(LLVM.ConstantInt(LLVM.Int64Type(), Int64(init)), ty))
            end
        end
    end
    return gv_to_value
end

"""
    _llvm_names_for_ci(native_code, ci) -> (func_name, specfunc_name)

Look up the names of the LLVM functions `jl_emit_native` produced for `ci`: the
generic-ABI entry and the specialized-signature body. Either may be `nothing`.
"""
function _llvm_names_for_ci(native_code::Ptr{Cvoid}, ci::Core.CodeInstance)
    llvm_func_idx = Ref{Int32}(-1)
    llvm_specfunc_idx = Ref{Int32}(-1)
    ccall(:jl_get_function_id, Nothing,
          (Ptr{Cvoid}, Any, Ptr{Int32}, Ptr{Int32}),
          native_code, ci, llvm_func_idx, llvm_specfunc_idx)

    function name_at(idx::Int32)
        idx >= 1 || return nothing
        ref = ccall(:jl_get_llvm_function, LLVM.API.LLVMValueRef,
                    (Ptr{Cvoid}, UInt32), native_code, idx - 1)
        ref == C_NULL && return nothing
        return LLVM.name(LLVM.Function(ref))
    end

    return name_at(llvm_func_idx[]), name_at(llvm_specfunc_idx[])
end
