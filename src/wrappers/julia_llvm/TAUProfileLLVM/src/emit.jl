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
`(; mod, ts_mod, native_code, method_instances)`, or `nothing` if Julia produced
no module. The caller must keep `ts_mod` alive while using `mod`.
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

    # Get compiled MIs via jl_get_llvm_mis
    method_instances = Any[]
    num_mis = Ref{Csize_t}(0)
    @ccall jl_get_llvm_mis(native_code::Ptr{Cvoid}, num_mis::Ptr{Csize_t},
                           C_NULL::Ptr{Cvoid})::Nothing
    resize!(method_instances, num_mis[])
    @ccall jl_get_llvm_mis(native_code::Ptr{Cvoid}, num_mis::Ptr{Csize_t},
                           method_instances::Ptr{Cvoid})::Nothing

    return (; mod = llvm_mod, ts_mod = llvm_ts_mod, native_code, method_instances)
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
