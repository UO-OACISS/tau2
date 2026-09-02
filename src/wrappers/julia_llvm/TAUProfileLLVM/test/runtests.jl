include("test_tracing_plugin.jl")
include("test_phase2_unit.jl")
include("test_tau_hooks.jl")
include("test_api_parity.jl")
include("test_tau_api.jl")
# The CUDA suite runs only when CUDA is a direct dependency of the active
# project, i.e. under `--project=test/cuda`. Checking the load path instead
# would also pick up a CUDA installed in the default environment, whose
# version need not co-resolve with this package's GPUCompiler and LLVM.
function _cuda_in_active_project()
    proj = Base.active_project()
    proj === nothing && return false
    isfile(proj) || return false
    return haskey(get(Base.parsed_toml(proj), "deps", Dict{String,Any}()), "CUDA")
end

if _cuda_in_active_project()
    include("test_cuda_safety.jl")
end
