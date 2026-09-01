# ============================================================================
# metadata.jl — per-compile LLVM function metadata maps and label building
#
# Part of the TAUProfile module (LLVM backend).
# ============================================================================

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
