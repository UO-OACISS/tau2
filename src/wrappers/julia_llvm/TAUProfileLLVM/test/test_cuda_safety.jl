#=
test_cuda_safety.jl — Comprehensive CUDA GPU tests for TAUProfile.jl

Tests that LLVM-level instrumentation works correctly with CUDA:
- Host functions that use CUDA operations can be instrumented
- Host-side callees are traced
- CUDA kernel launches work during/after instrumentation
- Memory transfers, array manipulation, streams, in-place ops
- Multi-level call chains mixing CPU and GPU
- Correctness: traced results match untraced results

Run: julia --project=. src/test_cuda_safety.jl
=#

using TAUProfile
using CUDA
using Test

if !CUDA.functional()
    @warn "CUDA not functional — skipping CUDA safety tests"
    exit(0)
end

println("CUDA device: ", CUDA.name(CUDA.device()))

# ============================================================================
# Test helpers
# ============================================================================

function capture_trace(f)
    tmpfile = tempname()
    result = nothing
    try
        open(tmpfile, "w") do io
            old_stderr = stderr
            redirect_stderr(io)
            try
                result = f()
            finally
                redirect_stderr(old_stderr)
            end
        end
        trace = read(tmpfile, String)
        return result, trace
    finally
        rm(tmpfile; force=true)
    end
end

function has_entry(trace::String, name::String)
    for line in split(trace, '\n')
        if startswith(line, ">>> Enter: ") && contains(line, name)
            return true
        end
    end
    return false
end

function has_exit(trace::String, name::String)
    for line in split(trace, '\n')
        if startswith(line, "<<< Exit:  ") && contains(line, name)
            return true
        end
    end
    return false
end

has_trace(trace::String, name::String) = has_entry(trace, name) && has_exit(trace, name)
no_trace(trace::String, name::String) = !has_entry(trace, name)

function count_entries(trace::String, name::String)
    c = 0
    for line in split(trace, '\n')
        if startswith(line, ">>> Enter: ") && contains(line, name)
            c += 1
        end
    end
    return c
end

# ============================================================================
# GPU kernels (compiled by GPUCompiler for GPU, not by our pipeline)
# ============================================================================

function gpu_add_kernel!(y, x)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function saxpy_kernel!(y, a, x)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(y)
        @inbounds y[i] = a * x[i] + y[i]
    end
    return nothing
end

function scale_kernel!(x, alpha)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds x[i] *= alpha
    end
    return nothing
end

function clamp_kernel!(x, lo, hi)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds x[i] = clamp(x[i], lo, hi)
    end
    return nothing
end

# ============================================================================
# Host functions that use CUDA
# ============================================================================

function host_cuda_basic()
    a = CUDA.rand(128, 128)
    b = sin.(a)
    synchronize()
    return size(Array(b))
end

function host_cuda_arithmetic()
    a = CUDA.rand(64, 64)
    b = CUDA.rand(64, 64)
    c = a .+ b
    d = c .* 2.0f0
    synchronize()
    return size(Array(d))
end

function host_cuda_matmul()
    A = CUDA.rand(Float32, 32, 16)
    B = CUDA.rand(Float32, 16, 8)
    C = A * B
    synchronize()
    return size(Array(C))
end

function host_cuda_reduce()
    a = CUDA.ones(Float32, 1024)
    s = sum(a)
    return s
end

function host_launch_kernel()
    N = 1024
    x_d = CUDA.fill(1.0f0, N)
    y_d = CUDA.fill(2.0f0, N)
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks gpu_add_kernel!(y_d, x_d)
    synchronize()
    return all(Array(y_d) .== 3.0f0)
end

function host_launch_saxpy()
    N = 512
    a = 2.0f0
    x_d = CUDA.ones(Float32, N)
    y_d = CUDA.ones(Float32, N)
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks saxpy_kernel!(y_d, a, x_d)
    synchronize()
    return all(Array(y_d) .== 3.0f0)
end

@noinline function host_compute_on_cpu(x, y)
    return x * y + sqrt(x + y)
end

function host_mixed_cpu_gpu(x, y)
    cpu_result = host_compute_on_cpu(x, y)
    a = CUDA.fill(Float32(cpu_result), 64)
    b = a .* 2.0f0
    synchronize()
    gpu_result = sum(Array(b))
    return gpu_result
end

# --- Host callee tracing test functions ---

@noinline function host_inner_helper(x)
    return x * 3.0f0
end

@noinline function host_middle_layer(x)
    v = host_inner_helper(x)
    return v + 1.0f0
end

function host_outer_with_callees(x)
    v = host_middle_layer(x)
    a = CUDA.fill(v, 32)
    synchronize()
    return sum(Array(a))
end

# --- Memory transfer functions ---

function host_transfer_roundtrip()
    cpu_data = Float32[1.0, 2.0, 3.0, 4.0, 5.0]
    gpu_data = CuArray(cpu_data)
    gpu_result = gpu_data .* 2.0f0
    synchronize()
    cpu_result = Array(gpu_result)
    return cpu_result
end

function host_fill_and_copy()
    a = CUDA.fill(42.0f0, 128)
    b = copy(a)
    synchronize()
    return sum(Array(b))
end

function host_cuda_indexing()
    a = CUDA.rand(Float32, 16, 16)
    b = a[1:8, :]
    synchronize()
    return size(Array(b))
end

function host_cuda_reshape()
    a = CUDA.ones(Float32, 64)
    b = reshape(a, 8, 8)
    c = b .+ 1.0f0
    synchronize()
    return size(Array(c))
end

# --- Multi-level call chain functions ---

@noinline function level3_compute(x)
    return x^2
end

@noinline function level2_gpu_work(x)
    v = level3_compute(x)
    a = CUDA.fill(Float32(v), 16)
    synchronize()
    return sum(Array(a))
end

@noinline function level1_orchestrator(x, y)
    r1 = level2_gpu_work(x)
    r2 = level2_gpu_work(y)
    return r1 + r2
end

function host_deep_call_chain(x, y)
    return level1_orchestrator(x, y)
end

# --- Multiple reduction patterns ---

function host_multiple_reductions()
    a = CUDA.ones(Float32, 512)
    s = sum(a)
    b = CUDA.fill(2.0f0, 256)
    p = sum(b)
    return (s, p)
end

function host_reduce_prod()
    a = CUDA.fill(2.0f0, 10)
    return prod(a)
end

function host_reduce_minmax()
    a = CuArray(Float32[3, 1, 4, 1, 5, 9, 2, 6])
    lo = minimum(a)
    hi = maximum(a)
    return (lo, hi)
end

# --- Multi-kernel pipeline ---

function host_multi_kernel_pipeline()
    N = 256
    x = CUDA.ones(Float32, N)
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks scale_kernel!(x, 3.0f0)
    synchronize()
    @cuda threads=threads blocks=blocks clamp_kernel!(x, 0.0f0, 2.5f0)
    synchronize()
    return sum(Array(x))
end

# --- In-place GPU operations ---

function host_inplace_broadcast()
    a = CUDA.ones(Float32, 64)
    a .+= 1.0f0
    a .*= 3.0f0
    synchronize()
    return sum(Array(a))
end

# --- Higher-order host function calling GPU ops ---

@noinline function apply_gpu_op(op, data)
    result = op.(data)
    synchronize()
    return result
end

function host_higher_order_gpu()
    a = CUDA.fill(0.5f0, 32)
    b = apply_gpu_op(sin, a)
    return sum(Array(b))
end

# --- Chained CUDA calls through helper ---

@noinline function cuda_step_alloc(n, val)
    return CUDA.fill(val, n)
end

@noinline function cuda_step_transform(a)
    b = a .* 2.0f0 .+ 1.0f0
    synchronize()
    return b
end

@noinline function cuda_step_collect(a)
    return sum(Array(a))
end

function host_cuda_chained_calls()
    a = cuda_step_alloc(64, 3.0f0)
    b = cuda_step_transform(a)
    return cuda_step_collect(b)
end

# --- Correctness comparison helpers ---

function untraced_cuda_matmul_result()
    A = CUDA.rand(Float32, 32, 16)
    B = CUDA.rand(Float32, 16, 8)
    C = A * B
    synchronize()
    return size(Array(C))
end

# --- Phase 2 GPU safety test functions ---

function host_dynamic_with_cuda(x)
    a = CUDA.rand(Float32, 100)
    s = sum(a)
    return Base.invokelatest(identity, s)
end

function host_cuda_reduce_phase2()
    a = CUDA.rand(Float32, 1000)
    return sum(a), prod(a), minimum(a), maximum(a)
end

function host_threaded_gpu_phase2(n::Int)
    results = Vector{Float32}(undef, n)
    Threads.@threads for i in 1:n
        a = CUDA.rand(Float32, 100)
        results[i] = sum(a)
    end
    return results
end

# ============================================================================
# Tests
# ============================================================================

@testset "TAUProfile CUDA Safety" begin

    # ================================================================
    # Baseline: CUDA works before any tracing
    # ================================================================

    @testset "Baseline CUDA functionality" begin
        @test host_cuda_basic() == (128, 128)
        @test host_launch_kernel() == true
        @test host_cuda_reduce() == 1024.0f0
        println("  Baseline CUDA: OK")
    end

    # ================================================================
    # Core CUDA operations — entry function tracing + correctness
    # ================================================================

    @testset "Core CUDA operations" begin
        @testset "CUDA.rand + broadcast" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_cuda_basic)
            end
            @test result == (128, 128)
            @test has_trace(trace, "host_cuda_basic")
        end

        @testset "CUDA arithmetic broadcast" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_cuda_arithmetic)
            end
            @test result == (64, 64)
            @test has_trace(trace, "host_cuda_arithmetic")
        end

        @testset "CUDA matrix multiply (CUBLAS)" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_cuda_matmul)
            end
            @test result == (32, 8)
            @test has_trace(trace, "host_cuda_matmul")
        end

        @testset "kernel launch" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_launch_kernel)
            end
            @test result == true
            @test has_trace(trace, "host_launch_kernel")
        end

        @testset "SAXPY kernel launch" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_launch_saxpy)
            end
            @test result == true
            @test has_trace(trace, "host_launch_saxpy")
        end
    end

    # ================================================================
    # Known crash case: sum(CuArray) uses @cfunction callbacks
    # ================================================================

    @testset "CUDA reduction (sum) — known crash case" begin
        tau_rewrite_reset_exclusions()
        result, trace = capture_trace() do
            tau_rewrite_and_call(host_cuda_reduce)
        end
        @test result == 1024.0f0
        @test has_trace(trace, "host_cuda_reduce")
    end

    # ================================================================
    # Host callee tracing — verify callees are traced, not just entry
    # ================================================================

    @testset "Host callee tracing" begin
        @testset "mixed CPU+GPU traces host callee" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_mixed_cpu_gpu, 3.0f0, 4.0f0)
            end
            expected_cpu = 3.0f0 * 4.0f0 + sqrt(3.0f0 + 4.0f0)
            expected_gpu = 64 * expected_cpu * 2.0f0
            @test result ≈ expected_gpu
            @test has_trace(trace, "host_mixed_cpu_gpu")
            @test has_trace(trace, "host_compute_on_cpu")
        end

        @testset "multi-level call chain traces all levels" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_outer_with_callees, 2.0f0)
            end
            expected = 32 * (2.0f0 * 3.0f0 + 1.0f0)
            @test result ≈ expected
            @test has_trace(trace, "host_outer_with_callees")
            @test has_trace(trace, "host_middle_layer")
            @test has_trace(trace, "host_inner_helper")
        end

        @testset "deep call chain traces all levels" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_deep_call_chain, 3.0f0, 4.0f0)
            end
            expected = 16.0f0 * 9.0f0 + 16.0f0 * 16.0f0
            @test result ≈ expected
            @test has_trace(trace, "host_deep_call_chain")
            @test has_trace(trace, "level1_orchestrator")
            @test has_trace(trace, "level2_gpu_work")
            @test has_trace(trace, "level3_compute")
        end

        @testset "level2_gpu_work called twice shows two entries" begin
            tau_rewrite_reset_exclusions()
            _, trace = capture_trace() do
                tau_rewrite_and_call(host_deep_call_chain, 3.0f0, 4.0f0)
            end
            @test count_entries(trace, "level2_gpu_work") >= 2
        end

        @testset "chained CUDA helpers all traced" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_cuda_chained_calls)
            end
            expected = 64.0f0 * (3.0f0 * 2.0f0 + 1.0f0)
            @test result ≈ expected
            @test has_trace(trace, "host_cuda_chained_calls")
            @test has_trace(trace, "cuda_step_alloc")
            @test has_trace(trace, "cuda_step_transform")
            @test has_trace(trace, "cuda_step_collect")
        end
    end

    # ================================================================
    # Memory transfers and array manipulation
    # ================================================================

    @testset "Memory transfers and array ops" begin
        @testset "host-device-host roundtrip" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_transfer_roundtrip)
            end
            @test result == Float32[2.0, 4.0, 6.0, 8.0, 10.0]
            @test has_trace(trace, "host_transfer_roundtrip")
        end

        @testset "fill and copy" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_fill_and_copy)
            end
            @test result ≈ 128.0f0 * 42.0f0
            @test has_trace(trace, "host_fill_and_copy")
        end

        @testset "CuArray indexing (slicing)" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_cuda_indexing)
            end
            @test result == (8, 16)
            @test has_trace(trace, "host_cuda_indexing")
        end

        @testset "reshape + broadcast" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_cuda_reshape)
            end
            @test result == (8, 8)
            @test has_trace(trace, "host_cuda_reshape")
        end
    end

    # ================================================================
    # Multiple reductions — stress-test @cfunction filtering
    # ================================================================

    @testset "Multiple reductions" begin
        @testset "two sequential GPU sums" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_multiple_reductions)
            end
            @test result == (512.0f0, 512.0f0)
            @test has_trace(trace, "host_multiple_reductions")
        end

        @testset "GPU prod reduction" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_reduce_prod)
            end
            @test result ≈ 1024.0f0
            @test has_trace(trace, "host_reduce_prod")
        end

        @testset "GPU min/max reductions" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_reduce_minmax)
            end
            @test result == (1.0f0, 9.0f0)
            @test has_trace(trace, "host_reduce_minmax")
        end
    end

    # ================================================================
    # Multi-kernel pipelines
    # ================================================================

    @testset "Multi-kernel pipeline" begin
        tau_rewrite_reset_exclusions()
        result, trace = capture_trace() do
            tau_rewrite_and_call(host_multi_kernel_pipeline)
        end
        @test result ≈ 256 * 2.5f0
        @test has_trace(trace, "host_multi_kernel_pipeline")
    end

    # ================================================================
    # In-place GPU operations
    # ================================================================

    @testset "In-place broadcast operations" begin
        tau_rewrite_reset_exclusions()
        result, trace = capture_trace() do
            tau_rewrite_and_call(host_inplace_broadcast)
        end
        @test result ≈ 64 * (1.0f0 + 1.0f0) * 3.0f0
        @test has_trace(trace, "host_inplace_broadcast")
    end

    # ================================================================
    # Higher-order host functions with GPU
    # ================================================================

    @testset "Higher-order host function with GPU" begin
        tau_rewrite_reset_exclusions()
        result, trace = capture_trace() do
            tau_rewrite_and_call(host_higher_order_gpu)
        end
        @test result ≈ 32 * sin(0.5f0)
        @test has_trace(trace, "host_higher_order_gpu")
        # apply_gpu_op may be inlined by LLVM optimization (runs after instrumentation)
    end

    # ================================================================
    # Exclusion API with CUDA functions
    # ================================================================

    @testset "Exclusion API with CUDA" begin
        @testset "exclude a host callee" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_exclude_function(:host_inner_helper)
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_outer_with_callees, 2.0f0)
            end
            expected = 32 * (2.0f0 * 3.0f0 + 1.0f0)
            @test result ≈ expected
            @test has_trace(trace, "host_outer_with_callees")
            @test has_trace(trace, "host_middle_layer")
            @test no_trace(trace, "host_inner_helper")
            tau_rewrite_reset_exclusions()
        end
    end

    # ================================================================
    # Correctness under tracing — compare traced vs untraced results
    # ================================================================

    @testset "Correctness: traced matches untraced" begin
        @testset "matmul dimensions match" begin
            untraced = untraced_cuda_matmul_result()
            traced, _ = capture_trace() do
                tau_rewrite_and_call(host_cuda_matmul)
            end
            @test traced == untraced
        end

        @testset "reduction value matches" begin
            untraced = host_cuda_reduce()
            traced, _ = capture_trace() do
                tau_rewrite_and_call(host_cuda_reduce)
            end
            @test traced == untraced
        end

        @testset "kernel launch correctness" begin
            traced, _ = capture_trace() do
                tau_rewrite_and_call(host_launch_kernel)
            end
            @test traced == true
        end
    end

    # ================================================================
    # Post-tracing: CUDA still works after instrumented execution
    # ================================================================

    @testset "Post-tracing CUDA functionality" begin
        @testset "CUDA still works after traced execution" begin
            N = 512
            x_d = CUDA.fill(1.0f0, N)
            y_d = CUDA.fill(2.0f0, N)
            @cuda threads=256 blocks=cld(N, 256) gpu_add_kernel!(y_d, x_d)
            synchronize()
            @test all(Array(y_d) .== 3.0f0)
        end

        @testset "GPU sum still works after traced reductions" begin
            a = CUDA.ones(Float32, 256)
            @test sum(a) == 256.0f0
        end

        @testset "kernel launch still works after traced kernel launches" begin
            N = 128
            x = CUDA.fill(2.0f0, N)
            @cuda threads=128 blocks=1 scale_kernel!(x, 5.0f0)
            synchronize()
            @test all(Array(x) .== 10.0f0)
        end
    end

    # ================================================================
    # Phase 2 GPU safety — verifies Phase 2 doesn't break GPU ops
    # ================================================================

    @testset "Phase 2 GPU safety" begin

        @testset "Category 1: Phase 2 replacement not called for GPU compilations" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_cuda_basic)
            end
            @test result == (128, 128)
            @test has_trace(trace, "host_cuda_basic")
            println("    Category 1: OK (no crash, GPU compile succeeds)")
        end

        @testset "Category 2: Phase 2 instrumented code doesn't break GPU operations" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_dynamic_with_cuda, 1)
            end
            @test result !== nothing
            @test isa(result, Float32)
            @test has_trace(trace, "host_dynamic_with_cuda")
            println("    Category 2: OK (host-side dispatch with CUDA works)")
        end

        @testset "Category 3: @cfunction safety with Phase 2 (sum/prod/min/max)" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_cuda_reduce_phase2)
            end
            @test result !== nothing
            s, p, mn, mx = result
            @test isa(s, Float32) && s > 0
            @test isa(p, Float32)
            @test isa(mn, Float32) && mn >= 0
            @test isa(mx, Float32) && mx >= 0
            @test has_trace(trace, "host_cuda_reduce_phase2")
            println("    Category 3: OK (sum/prod/min/max with Phase 2 active)")
        end

        @testset "Category 4: Sequential GPU after traced execution" begin
            tau_rewrite_reset_exclusions()
            result1, trace = capture_trace() do
                tau_rewrite_and_call(host_cuda_basic)
            end
            @test result1 == (128, 128)

            a = CUDA.rand(Float32, 100)
            b = CUDA.rand(Float32, 100)
            c = a .+ b
            synchronize()
            @test length(c) == 100
            s = sum(c)
            @test s > 0
            println("    Category 4: OK (untraced CUDA after Phase 2 instrumentation)")
        end

        @testset "Category 5: Multi-threaded GPU work concurrent with Phase 2" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace() do
                tau_rewrite_and_call(host_threaded_gpu_phase2, 4)
            end
            @test result !== nothing
            @test length(result) == 4
            for r in result
                @test isa(r, Float32)
                @test r > 0
            end
            @test has_trace(trace, "host_threaded_gpu_phase2")
            println("    Category 5: OK (multi-threaded GPU with Phase 2 active)")
        end

    end

end

println("\nAll TAUProfile CUDA safety tests completed!")
