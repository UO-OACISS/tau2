include("test_tracing_plugin.jl")
include("test_phase2_unit.jl")
include("test_tau_hooks.jl")
include("test_api_parity.jl")
if get(ENV, "TAU_JULIA_LLVM_TEST_CUDA", "0") == "1"
    include("test_cuda_safety.jl")
end
