include("test_tracing_plugin.jl")
include("test_phase2_unit.jl")
include("test_tau_hooks.jl")
include("test_api_parity.jl")
include("test_tau_api.jl")
# The CUDA suite runs only when CUDA.jl is resolvable in the active environment
if Base.identify_package("CUDA") !== nothing
    include("test_cuda_safety.jl")
end
