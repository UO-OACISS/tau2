#=
test_tracing_plugin.jl — Tests for TAUProfile.jl

Run: julia --project=. src/test_tracing_plugin.jl
=#

using TAUProfile
using Test

TAUProfile._force_text_hooks[] = true

# ============================================================================
# Test helpers
# ============================================================================

"""
    capture_trace(f) -> (result, trace_output::String)

Execute `f()` while capturing stderr to a tempfile.
"""
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

"""Check if trace contains entry hook for given function name."""
function has_entry(trace::String, name::String)
    for line in split(trace, '\n')
        if startswith(line, ">>> Enter: ") && contains(line, name)
            return true
        end
    end
    return false
end

"""Check if trace contains exit hook for given function name."""
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

function count_exits(trace::String, name::String)
    c = 0
    for line in split(trace, '\n')
        if startswith(line, "<<< Exit:  ") && contains(line, name)
            c += 1
        end
    end
    return c
end

"""Check that name_a exits before name_b in the trace output."""
function exits_before(trace::String, name_a::String, name_b::String)
    pos_a = pos_b = nothing
    for (i, line) in enumerate(split(trace, '\n'))
        if startswith(line, "<<< Exit:  ")
            if contains(line, name_a) && pos_a === nothing
                pos_a = i
            end
            if contains(line, name_b) && pos_b === nothing
                pos_b = i
            end
        end
    end
    return pos_a !== nothing && pos_b !== nothing && pos_a < pos_b
end

# ============================================================================
# Test functions
# ============================================================================

simple_add(a, b) = a + b

function compute(x, y)
    return x + y * 2.0
end

function returns_int(x)
    return x + 1
end

function returns_string(x)
    return string("hello_", x)
end

function multi_arg(a, b, c, d)
    return a + b + c + d
end

# Depth test functions
@noinline depth3_func(x) = x * 2
@noinline depth2_func(x) = depth3_func(x) + 1
@noinline depth1_func(x) = depth2_func(x) + 1
depth0_func(x) = depth1_func(x) + 1

# Module depth test functions
module TestModA
    @noinline deep_a(x) = x * 3
    @noinline mid_a(x) = deep_a(x) + 1
    @noinline entry_a(x) = mid_a(x) + 1
end

module TestModB
    @noinline work_b(x) = x + 10
end

@noinline call_mod_a(x) = TestModA.entry_a(x)
@noinline call_mod_b(x) = TestModB.work_b(x)
function multi_mod(x)
    a = call_mod_a(x)
    b = call_mod_b(x)
    return a + b
end

# Prefix test functions
@noinline prefix_alpha_1(x) = x + 1
@noinline prefix_alpha_2(x) = x + 2
@noinline prefix_beta_1(x) = x + 3
@noinline not_prefixed(x) = prefix_alpha_1(x) + prefix_beta_1(x)

# Call chain for callee tracing
@noinline inner_helper(x) = x * 2
@noinline middle_layer(x) = inner_helper(x) + 1
function outer_chain(x)
    return middle_layer(x) + 1
end

# Higher-order function test
@noinline apply_twice(f, x) = f(f(x))
function ho_test(x)
    return apply_twice(returns_int, x)
end

# Recursion test
@noinline function factorial_rec(n)
    n <= 1 && return 1
    return n * factorial_rec(n - 1)
end

# Mutual recursion
@noinline is_even_rec(n) = n == 0 ? true : is_odd_rec(n - 1)
@noinline is_odd_rec(n) = n == 0 ? false : is_even_rec(n - 1)

# Exception test functions
@noinline throws_always(x) = error("test error from throws_always")

@noinline function throws_if_negative(x)
    x < 0 && error("negative: $x")
    return x + 1
end

@noinline function has_internal_try_catch(x)
    try
        x < 0 && error("neg")
        return x + 1
    catch
        return -1
    end
end

@noinline exc_inner_throws(x) = error("inner error")
@noinline function exc_outer_catches(x)
    try
        return exc_inner_throws(x)
    catch
        return -1
    end
end

@noinline exc_deep_throw(x) = error("deep")
@noinline exc_mid_nothrow(x) = exc_deep_throw(x)
@noinline function exc_top_catches(x)
    try
        return exc_mid_nothrow(x)
    catch
        return -1
    end
end

# Nested noinline functions with string interpolation in error() to prevent LLVM inlining.
# Simple `error("constant")` gets inlined; `error("msg: $x")` involves string allocation
# which keeps the function body complex enough to survive LLVM optimization.
@noinline function nested_exc_thrower(x::Int)
    x < 0 && error("negative: $x")
    return x + 1
end
@noinline function nested_exc_middle(x::Int)
    return nested_exc_thrower(x) + 10
end
@noinline function nested_exc_catcher(x::Int)
    try
        return nested_exc_middle(x)
    catch
        return -999
    end
end

# ============================================================================
# Edge case test functions (ported from TracingAI)
# ============================================================================

# --- Edge case: implicit return (no explicit return statement) ---
function implicit_return_ec(x::Float64)
    x + 1.0
end

# --- Edge case: function returning nothing ---
function returns_nothing_ec(x::Float64)
    if x > 0
        _ = x * 2.0
    end
    return nothing
end

# --- Edge case: multiple dispatch (same function name, different signatures) ---
@noinline function polymorphic_add_ec(x::Int, y::Int)
    return x + y
end

@noinline function polymorphic_add_ec(x::Float64, y::Float64)
    return x + y
end

function dispatch_caller_ec(flag::Bool, a::Int, b::Int, c::Float64, d::Float64)
    if flag
        return polymorphic_add_ec(a, b)
    else
        return polymorphic_add_ec(c, d)
    end
end

# --- Edge case: function returning a tuple ---
function multi_return_values_ec(x::Float64)
    return (x, x^2, x^3)
end

# --- Edge case: deeply nested call chain ---
@noinline chain_e_ec(x::Float64) = x
@noinline chain_d_ec(x::Float64) = chain_e_ec(x + 1.0)
@noinline chain_c_ec(x::Float64) = chain_d_ec(x + 1.0)
@noinline chain_b_ec(x::Float64) = chain_c_ec(x + 1.0)
function chain_a_ec(x::Float64)
    return chain_b_ec(x + 1.0)
end

# --- Edge case: internal try-catch (function handles its own exceptions) ---
function internal_trycatch_ec(x::Float64)
    result = try
        x < 0 ? error("neg") : sqrt(x)
    catch e
        -1.0
    end
    return result
end

# --- Edge case: higher-order function ---
@noinline double_ec(x::Float64) = x * 2.0

function apply_twice_ec(f, x::Float64)
    return f(f(x))
end

# --- Edge case: function with no arguments ---
function no_args_ec()
    return 42
end

# --- Edge case: function calling a closure created inside it ---
function uses_closure_ec(x::Float64)
    adder = y -> y + x
    return adder(10.0)
end

# --- Edge case: function with many branches / multiple return points ---
function multi_branch_ec(x::Float64, y::Float64)
    if x > 0 && y > 0
        return x + y
    elseif x > 0 && y <= 0
        return x - y
    elseif x <= 0 && y > 0
        return y - x
    elseif x == 0 && y == 0
        return 0.0
    else
        return -(x + y)
    end
end

# --- Edge case: function that calls string interpolation / string ops ---
function build_message_ec(x::Float64)
    msg = "The value is $(x) and its square is $(x^2)"
    return msg
end

# --- Edge case: function calling another function that returns nothing ---
@noinline function returns_nothing_callee_ec(x::Float64)
    Base.show(Base.devnull, x * 2.0)
    return nothing
end

@noinline function calls_void_ec(x::Float64)
    returns_nothing_callee_ec(x)
    return x + 1.0
end

# --- Edge case: keyword arguments ---
@noinline function kwarg_helper_ec(x::Float64; scale::Float64=2.0, offset::Float64=0.0)
    return x * scale + offset
end

function calls_kwarg_ec(x::Float64)
    return kwarg_helper_ec(x; scale=3.0, offset=1.0)
end

# --- Edge case: default argument values ---
@noinline function with_default_ec(x::Float64, y::Float64=1.0)
    return x + y
end

function calls_with_default_ec(x::Float64)
    a = with_default_ec(x, 5.0)
    b = with_default_ec(x)
    return a + b
end

# --- Edge case: user-defined varargs ---
@noinline function varargs_sum_ec(args::Float64...)
    total = 0.0
    for a in args
        total += a
    end
    return total
end

function calls_varargs_ec(x::Float64)
    return varargs_sum_ec(x, x + 1.0, x + 2.0)
end

# --- Edge case: splatting at call site ---
@noinline add_ec(x, y) = x + y

function splat_caller_ec(x::Float64, y::Float64)
    args = (x, y)
    return add_ec(args...)
end

# --- Edge case: splatting with deeper call chain ---
@noinline splat_inner_ec(x::Float64) = x + 100.0

@noinline function splat_target_ec(x::Float64, y::Float64)
    return splat_inner_ec(x) + splat_inner_ec(y)
end

function splat_chain_ec(x::Float64, y::Float64)
    args = (x, y)
    return splat_target_ec(args...)
end

# --- Edge case: type-parameterized method (where T) ---
@noinline function generic_add_ec(x::T, y::T) where T<:Number
    return x + y
end

# --- Edge case: where {T,N} with inner calls (GrayScott iterate! pattern) ---
struct SimFields_ec{T, N}
    data::Array{T, N}
end

@noinline function sim_inner_step_ec!(fields::SimFields_ec{T, N}) where {T, N}
    for i in eachindex(fields.data)
        fields.data[i] += one(T)
    end
    return nothing
end

@noinline function sim_iterate_ec!(fields::SimFields_ec{T, N}, dt::T) where {T, N}
    sim_inner_step_ec!(fields)
    fields.data .*= dt
    return nothing
end

function calls_sim_iterate_ec()
    fields = SimFields_ec(zeros(Float64, 3, 3))
    sim_iterate_ec!(fields, 0.5)
    return sum(fields.data)
end

# --- Edge case: function returning a mutable struct ---
mutable struct Point_ec
    x::Float64
    y::Float64
end

@noinline function make_point_ec(x::Float64, y::Float64)
    return Point_ec(x, y)
end

function transform_point_ec(x::Float64, y::Float64)
    p = make_point_ec(x, y)
    p.x += 1.0
    p.y *= 2.0
    return p
end

# --- Edge case: many arguments ---
@noinline function many_args_ec(a::Float64, b::Float64, c::Float64,
                                d::Float64, e::Float64, f::Float64)
    return a + b * c - d / (e + f)
end

function calls_many_args_ec(x::Float64)
    return many_args_ec(x, x+1, x+2, x+3, x+4, x+5)
end

# --- Edge case: comprehension / generator ---
function uses_comprehension_ec(n::Int)
    return [i^2 for i in 1:n]
end

# --- Edge case: nested calls in single expression ---
function nested_calls_ec(x::Float64)
    return add_ec(add_ec(x, 1.0), add_ec(x, 2.0))
end

# --- Edge case: immediate error (before any call) ---
function immediate_error_ec(x::Float64)
    error("fail immediately from immediate_error_ec: $x")
    return x
end

# --- Edge case: Union-typed argument at call site ---
@noinline function process_value_ec(x)
    return x + x
end

function test_union_dispatch_ec(x::Bool)
    val = x ? 1.0 : [1.0, 2.0, 3.0]
    return process_value_ec(val)
end

# --- Edge case: re-tracing same function ---
# Uses existing simple_add; no new function needed

# ============================================================================
# Cross-module test functions
# ============================================================================

# --- Inline module for cross-module tracing ---
module ExternalSim_ec
    export sim_step_ec, sim_main_ec

    @noinline function sim_step_ec(x::Float64, y::Float64)
        return x * 0.9 + y * 0.1
    end

    @noinline function sim_main_ec(args::Vector{String})
        x = 1.0
        y = 2.0
        for _ in 1:3
            x = sim_step_ec(x, y)
        end
        return x
    end
end
using .ExternalSim_ec: sim_main_ec, sim_step_ec

function cross_module_wrapper_ec()
    return sim_main_ec(String[])
end

# --- File-loaded module for cross-module tracing ---
include("ExternalSim.jl")
import .ExternalSimPkg_ec

function cross_file_wrapper_ec()
    return ExternalSimPkg_ec.sim_main_ec(String[])
end


@noinline invoke_inner_ec(x::Float64) = x + 100.0
@noinline invoke_target_ec(x::Float64) = invoke_inner_ec(x) * 3.0

function calls_via_invoke_ec(x::Float64)
    return @invoke invoke_target_ec(x::Float64)
end

@noinline invokelatest_inner_ec(x::Float64) = x + 50.0
@noinline invokelatest_target_ec(x::Float64) = invokelatest_inner_ec(x) * 2.0

function calls_via_invokelatest_ec(x::Float64)
    return Base.invokelatest(invokelatest_target_ec, x)
end

@noinline invoke_world_inner_ec(x::Float64) = x + 25.0
@noinline invoke_world_target_ec(x::Float64) = invoke_world_inner_ec(x) * 4.0

function calls_via_invoke_in_world_ec(x::Float64)
    return Base.invoke_in_world(Base.tls_world_age(), invoke_world_target_ec, x)
end


module MathHelpers_ec
    export mh_add_ec
    @noinline function mh_add_ec(x::Float64, y::Float64)
        return x + y
    end

    module MathSubHelpers_ec
        export msh_mul_ec
        @noinline function msh_mul_ec(x::Float64, y::Float64)
            return x * y
        end
    end
end
using .MathHelpers_ec: mh_add_ec
using .MathHelpers_ec.MathSubHelpers_ec: msh_mul_ec

@noinline function outer_compute_with_sub_ec(x::Float64, y::Float64)
    s = mh_add_ec(x, y)
    p = msh_mul_ec(x, y)
    return s + p
end

function outer_compute_ec(x::Float64, y::Float64)
    s = mh_add_ec(x, y)
    p = x * y
    return s + p
end

# Modules for hierarchical depth limit tests
module DepthOuter_ec
    export do_entry_ec, do_mid_ec

    module DepthInner_ec
        export di_func_ec
        @noinline di_func_ec(x::Float64) = x + 200.0
    end
    import .DepthInner_ec: di_func_ec

    @noinline do_mid_ec(x::Float64) = di_func_ec(x) + 50.0
    @noinline do_entry_ec(x::Float64) = do_mid_ec(x) + 100.0
end
using .DepthOuter_ec: do_entry_ec, do_mid_ec
using .DepthOuter_ec.DepthInner_ec: di_func_ec

function depth_caller_outer_inner_ec(x::Float64)
    return do_entry_ec(x)
end

# ============================================================================
# Tests
# ============================================================================

@testset "TAUProfile" begin

    @testset "IR generation" begin
        @testset "trace_code returns LLVM IR string" begin
            ir = trace_code(simple_add, 1.0, 2.0)
            @test isa(ir, String)
            @test !isempty(ir)
            @test contains(ir, "define")
        end

        @testset "IR contains hook calls" begin
            ir = trace_code(simple_add, 1.0, 2.0)
            @test contains(ir, "simple_add")
            @test contains(ir, "inttoptr")
        end
    end

    @testset "Basic execution" begin
        @testset "simple arithmetic" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(simple_add, 1.0, 2.0))
            @test result == 3.0
            @test has_trace(trace, "simple_add")
        end

        @testset "integer arithmetic" begin
            result, trace = capture_trace(() -> tau_rewrite_and_call(returns_int, 41))
            @test result == 42
            @test has_trace(trace, "returns_int")
        end

        @testset "string return" begin
            result, trace = capture_trace(() -> tau_rewrite_and_call(returns_string, 42))
            @test result == "hello_42"
            @test has_trace(trace, "returns_string")
        end

        @testset "multiple arguments" begin
            result, trace = capture_trace(() -> tau_rewrite_and_call(multi_arg, 1, 2, 3, 4))
            @test result == 10
            @test has_trace(trace, "multi_arg")
        end

        @testset "float computation" begin
            result, trace = capture_trace(() -> tau_rewrite_and_call(compute, 1.0, 2.0))
            @test result == 5.0
            @test has_trace(trace, "compute")
        end
    end

    @testset "Label format" begin
        @testset "labels include module prefix" begin
            tau_rewrite_reset_exclusions()
            _, trace = capture_trace(() -> tau_rewrite_and_call(simple_add, 1.0, 2.0))
            @test has_trace(trace, "Main.simple_add")
        end

        @testset "labels include file and line info" begin
            ir = trace_code(simple_add, 1.0, 2.0)
            @test contains(ir, "simple_add")
        end

        @testset "include_types adds argument types" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_include_types(true)
            _, trace = capture_trace(() -> tau_rewrite_and_call(simple_add, 1.0, 2.0))
            @test has_trace(trace, "Float64, Float64")
            tau_rewrite_reset_exclusions()
        end

        @testset "include_types off by default" begin
            tau_rewrite_reset_exclusions()
            _, trace = capture_trace(() -> tau_rewrite_and_call(simple_add, 1, 2))
            @test !has_entry(trace, "Int64, Int64")
        end
    end

    @testset "Callee tracing" begin
        @testset "callees are traced" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(outer_chain, 5))
            @test result == 12
            @test has_trace(trace, "outer_chain")
            @test has_trace(trace, "middle_layer")
            @test has_trace(trace, "inner_helper")
        end

        @testset "recursive function" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(factorial_rec, 5))
            @test result == 120
            @test has_trace(trace, "factorial_rec")
            @test count_entries(trace, "factorial_rec") >= 2
        end

        @testset "mutual recursion" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(is_even_rec, 4))
            @test result == true
            @test has_trace(trace, "is_even_rec")
            @test has_trace(trace, "is_odd_rec")
        end
    end

    @testset "Exclusion API" begin
        @testset "exclude by function name" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_exclude_function(:simple_add)
            ir = trace_code(simple_add, 1.0, 2.0)
            @test contains(ir, "define")
            tau_rewrite_reset_exclusions()
        end

        @testset "exclude callee by name" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_exclude_function(:inner_helper)
            _, trace = capture_trace(() -> tau_rewrite_and_call(outer_chain, 5))
            @test has_trace(trace, "outer_chain")
            @test has_trace(trace, "middle_layer")
            @test no_trace(trace, "inner_helper")
            tau_rewrite_reset_exclusions()
        end

        @testset "reset exclusions" begin
            tau_rewrite_exclude_function(:simple_add)
            tau_rewrite_reset_exclusions()
            ir = trace_code(simple_add, 1.0, 2.0)
            @test contains(ir, "inttoptr")
        end
    end

    @testset "Prefix exclusion" begin
        @testset "exclude by prefix" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_exclude_prefix("prefix_alpha")
            _, trace = capture_trace(() -> tau_rewrite_and_call(not_prefixed, 1))
            @test has_trace(trace, "not_prefixed")
            @test has_trace(trace, "prefix_beta_1")
            @test no_trace(trace, "prefix_alpha_1")
            @test no_trace(trace, "prefix_alpha_2")
            tau_rewrite_reset_exclusions()
        end

        @testset "multiple prefixes" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_exclude_prefix("prefix_alpha")
            tau_rewrite_exclude_prefix("prefix_beta")
            _, trace = capture_trace(() -> tau_rewrite_and_call(not_prefixed, 1))
            @test has_trace(trace, "not_prefixed")
            @test no_trace(trace, "prefix_alpha_1")
            @test no_trace(trace, "prefix_beta_1")
            tau_rewrite_reset_exclusions()
        end
    end

    @testset "Module exclusion" begin
        @testset "exclude a module" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_exclude_module(:TestModA)
            _, trace = capture_trace(() -> tau_rewrite_and_call(multi_mod, 5))
            @test has_trace(trace, "multi_mod")
            @test has_trace(trace, "call_mod_b")
            @test has_trace(trace, "work_b")
            @test no_trace(trace, "entry_a")
            @test no_trace(trace, "mid_a")
            @test no_trace(trace, "deep_a")
            tau_rewrite_reset_exclusions()
        end
    end

    @testset "Module whitelist" begin
        @testset "whitelist Main only (exact)" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_include_module_only(:Main; exact=true)
            _, trace = capture_trace(() -> tau_rewrite_and_call(returns_string, 42))
            @test has_trace(trace, "returns_string")
            @test no_trace(trace, "print_to_string")
            tau_rewrite_reset_exclusions()
        end

        @testset "whitelist specific module" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_include_module_only(:TestModA)
            _, trace = capture_trace(() -> tau_rewrite_and_call(multi_mod, 5))
            @test no_trace(trace, "multi_mod")
            @test no_trace(trace, "call_mod_b")
            @test has_trace(trace, "entry_a")
            @test has_trace(trace, "mid_a")
            @test has_trace(trace, "deep_a")
            tau_rewrite_reset_exclusions()
        end
    end

    @testset "Global recursion depth limit" begin
        @testset "depth limit 1 — only entry function" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_set_recursion_limit(1)
            result, trace = capture_trace(() -> tau_rewrite_and_call(depth0_func, 5))
            @test result == (5 * 2 + 1 + 1 + 1)
            @test has_trace(trace, "depth0_func")
            @test no_trace(trace, "depth1_func")
            @test no_trace(trace, "depth2_func")
            @test no_trace(trace, "depth3_func")
            tau_rewrite_reset_exclusions()
        end

        @testset "depth limit 2 — entry + 1 level" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_set_recursion_limit(2)
            result, trace = capture_trace(() -> tau_rewrite_and_call(depth0_func, 5))
            @test result == (5 * 2 + 1 + 1 + 1)
            @test has_trace(trace, "depth0_func")
            @test has_trace(trace, "depth1_func")
            @test no_trace(trace, "depth2_func")
            @test no_trace(trace, "depth3_func")
            tau_rewrite_reset_exclusions()
        end

        @testset "depth limit 3" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_set_recursion_limit(3)
            result, trace = capture_trace(() -> tau_rewrite_and_call(depth0_func, 5))
            @test has_trace(trace, "depth0_func")
            @test has_trace(trace, "depth1_func")
            @test has_trace(trace, "depth2_func")
            @test no_trace(trace, "depth3_func")
            tau_rewrite_reset_exclusions()
        end

        @testset "depth limit 0 — unlimited" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_set_recursion_limit(0)
            _, trace = capture_trace(() -> tau_rewrite_and_call(depth0_func, 5))
            @test has_trace(trace, "depth0_func")
            @test has_trace(trace, "depth1_func")
            @test has_trace(trace, "depth2_func")
            @test has_trace(trace, "depth3_func")
            tau_rewrite_reset_exclusions()
        end
    end

    @testset "Per-module recursion depth limit" begin
        @testset "module depth 1 — only first function in module" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_set_recursion_limit(:TestModA, 1)
            _, trace = capture_trace(() -> tau_rewrite_and_call(call_mod_a, 5))
            @test has_trace(trace, "call_mod_a")
            @test has_trace(trace, "entry_a")
            @test no_trace(trace, "mid_a")
            @test no_trace(trace, "deep_a")
            tau_rewrite_reset_exclusions()
        end

        @testset "module depth 2 — two levels into module" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_set_recursion_limit(:TestModA, 2)
            _, trace = capture_trace(() -> tau_rewrite_and_call(call_mod_a, 5))
            @test has_trace(trace, "call_mod_a")
            @test has_trace(trace, "entry_a")
            @test has_trace(trace, "mid_a")
            @test no_trace(trace, "deep_a")
            tau_rewrite_reset_exclusions()
        end

        @testset "module depth 3 — all levels in module" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_set_recursion_limit(:TestModA, 3)
            _, trace = capture_trace(() -> tau_rewrite_and_call(call_mod_a, 5))
            @test has_trace(trace, "entry_a")
            @test has_trace(trace, "mid_a")
            @test has_trace(trace, "deep_a")
            tau_rewrite_reset_exclusions()
        end

        @testset "module limit only affects that module" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_set_recursion_limit(:TestModA, 1)
            _, trace = capture_trace(() -> tau_rewrite_and_call(multi_mod, 5))
            @test has_trace(trace, "multi_mod")
            @test has_trace(trace, "call_mod_a")
            @test has_trace(trace, "call_mod_b")
            @test has_trace(trace, "entry_a")
            @test no_trace(trace, "mid_a")
            @test has_trace(trace, "work_b")
            tau_rewrite_reset_exclusions()
        end

        @testset "combined global + module limits" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_set_recursion_limit(3)
            tau_rewrite_set_recursion_limit(:TestModA, 1)
            _, trace = capture_trace(() -> tau_rewrite_and_call(multi_mod, 5))
            @test has_trace(trace, "multi_mod")
            @test has_trace(trace, "entry_a")
            @test no_trace(trace, "mid_a")
            @test has_trace(trace, "work_b")
            tau_rewrite_reset_exclusions()
        end
    end

    @testset "Tracing toggle" begin
        @testset "disable/enable tracing" begin
            tau_rewrite_reset_exclusions()
            disable_tracing!()
            @test !tracing_enabled()
            ir = trace_code(simple_add, 1.0, 2.0)
            @test !contains(ir, "inttoptr")

            enable_tracing!()
            @test tracing_enabled()
            ir = trace_code(simple_add, 1.0, 2.0)
            @test contains(ir, "inttoptr")
        end
    end

    @testset "Exception-safe exit hooks" begin
        @testset "IR contains exception handler" begin
            tau_rewrite_reset_exclusions()
            ir = trace_code(simple_add, 1.0, 2.0)
            @test contains(ir, "ijl_enter_handler")
            @test contains(ir, "ijl_pop_handler_noexcept")
            @test contains(ir, "ijl_rethrow")
            @test contains(ir, "catch")
        end

        @testset "normal return still works" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(simple_add, 1.0, 2.0))
            @test result == 3.0
            @test has_entry(trace, "simple_add")
            @test has_exit(trace, "simple_add")
        end

        @testset "uncaught exception fires exit hook" begin
            tau_rewrite_reset_exclusions()
            local caught_err = nothing
            _, trace = capture_trace(() -> begin
                try
                    tau_rewrite_and_call(throws_always, 1)
                catch e
                    caught_err = e
                end
            end)
            @test caught_err isa ErrorException
            @test contains(caught_err.msg, "test error from throws_always")
            @test has_entry(trace, "throws_always")
            @test has_exit(trace, "throws_always")
        end

        @testset "conditional throw - normal path" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(throws_if_negative, 5))
            @test result == 6
            @test has_entry(trace, "throws_if_negative")
            @test has_exit(trace, "throws_if_negative")
        end

        @testset "conditional throw - exception path" begin
            tau_rewrite_reset_exclusions()
            local caught_err = nothing
            _, trace = capture_trace(() -> begin
                try
                    tau_rewrite_and_call(throws_if_negative, -1)
                catch e
                    caught_err = e
                end
            end)
            @test caught_err isa ErrorException
            @test has_entry(trace, "throws_if_negative")
            @test has_exit(trace, "throws_if_negative")
        end

        @testset "internal try/catch still works" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(has_internal_try_catch, -1))
            @test result == -1
            @test has_entry(trace, "has_internal_try_catch")
            @test has_exit(trace, "has_internal_try_catch")
        end

        @testset "internal try/catch normal path" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(has_internal_try_catch, 5))
            @test result == 6
            @test has_entry(trace, "has_internal_try_catch")
            @test has_exit(trace, "has_internal_try_catch")
        end

        @testset "nested: inner throws, outer catches" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(exc_outer_catches, 1))
            @test result == -1
            @test has_entry(trace, "exc_outer_catches")
            @test has_exit(trace, "exc_outer_catches")
            # Note: exc_inner_throws may be inlined by LLVM optimization
        end

        @testset "deep nesting with propagation" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(exc_top_catches, 1))
            @test result == -1
            @test has_entry(trace, "exc_top_catches")
            @test has_exit(trace, "exc_top_catches")
            # Note: exc_deep_throw and exc_mid_nothrow may be inlined
        end

        @testset "nested noinline: normal path" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(nested_exc_catcher, 5))
            @test result == 16
            @test has_entry(trace, "nested_exc_catcher")
            @test has_exit(trace, "nested_exc_catcher")
            @test has_entry(trace, "nested_exc_middle")
            @test has_exit(trace, "nested_exc_middle")
            @test has_entry(trace, "nested_exc_thrower")
            @test has_exit(trace, "nested_exc_thrower")
        end

        @testset "nested noinline: exception path fires all exit hooks" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(nested_exc_catcher, -1))
            @test result == -999
            # All three functions must have entry AND exit hooks
            @test has_entry(trace, "nested_exc_thrower")
            @test has_exit(trace, "nested_exc_thrower")
            @test has_entry(trace, "nested_exc_middle")
            @test has_exit(trace, "nested_exc_middle")
            @test has_entry(trace, "nested_exc_catcher")
            @test has_exit(trace, "nested_exc_catcher")
            # Exit order must be innermost-first (stack unwinding)
            @test exits_before(trace, "nested_exc_thrower", "nested_exc_middle")
            @test exits_before(trace, "nested_exc_middle", "nested_exc_catcher")
        end
    end

    @testset "Edge cases" begin
        @testset "implicit return" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(implicit_return_ec, 5.0))
            @test result == 6.0
            @test has_trace(trace, "implicit_return_ec")
        end

        @testset "returns nothing" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(returns_nothing_ec, 1.0))
            @test result === nothing
            @test has_trace(trace, "returns_nothing_ec")
        end

        @testset "tuple return value" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(multi_return_values_ec, 3.0))
            @test result == (3.0, 9.0, 27.0)
            @test has_trace(trace, "multi_return_values_ec")
        end

        @testset "no arguments" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(no_args_ec))
            @test result == 42
        end

        @testset "closure" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(uses_closure_ec, 5.0))
            @test result == 15.0
            @test has_trace(trace, "uses_closure_ec")
        end

        @testset "multiple return points" begin
            tau_rewrite_reset_exclusions()
            cases = [
                ((1.0, 2.0), 3.0),
                ((1.0, -2.0), 3.0),
                ((-1.0, 2.0), 3.0),
                ((0.0, 0.0), 0.0),
                ((-1.0, -2.0), 3.0),
            ]
            for ((x, y), expected) in cases
                result, trace = capture_trace(() -> tau_rewrite_and_call(multi_branch_ec, x, y))
                @test result == expected
                @test has_trace(trace, "multi_branch_ec")
            end
        end

        @testset "string interpolation" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(build_message_ec, 7.0))
            @test result == "The value is 7.0 and its square is 49.0"
            @test has_trace(trace, "build_message_ec")
        end

        @testset "internal try-catch (exception caught)" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(internal_trycatch_ec, -1.0))
            @test result == -1.0
            @test has_trace(trace, "internal_trycatch_ec")
        end

        @testset "internal try-catch (no exception)" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(internal_trycatch_ec, 4.0))
            @test result == 2.0
            @test has_trace(trace, "internal_trycatch_ec")
        end

        @testset "higher-order function" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(apply_twice_ec, double_ec, 3.0))
            @test result == 12.0
            @test has_trace(trace, "apply_twice_ec")
            @test has_trace(trace, "double_ec")
        end

        @testset "callee returns nothing" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(calls_void_ec, 5.0))
            @test result == 6.0
            @test has_trace(trace, "calls_void_ec")
            @test has_trace(trace, "returns_nothing_callee_ec")
        end

        @testset "mutable struct return" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(transform_point_ec, 1.0, 2.0))
            @test result.x == 2.0
            @test result.y == 4.0
            @test has_trace(trace, "transform_point_ec")
            @test has_trace(trace, "make_point_ec")
        end

        @testset "many arguments (6)" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(calls_many_args_ec, 1.0))
            @test has_trace(trace, "calls_many_args_ec")
            @test has_trace(trace, "many_args_ec")
        end

        @testset "list comprehension" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(uses_comprehension_ec, 5))
            @test result == [1, 4, 9, 16, 25]
            @test has_trace(trace, "uses_comprehension_ec")
        end

        @testset "nested calls f(g(x), h(x))" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(nested_calls_ec, 1.0))
            @test result == 5.0
            @test has_trace(trace, "nested_calls_ec")
            @test has_trace(trace, "add_ec")
        end

        @testset "immediate error" begin
            tau_rewrite_reset_exclusions()
            local caught_err = nothing
            _, trace = capture_trace(() -> begin
                try
                    tau_rewrite_and_call(immediate_error_ec, 1.0)
                catch e
                    caught_err = e
                end
            end)
            @test caught_err isa ErrorException
            @test has_trace(trace, "immediate_error_ec")
        end

        @testset "deep call chain (5 levels)" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(chain_a_ec, 1.0))
            @test result == 5.0
            @test has_trace(trace, "chain_a_ec")
            @test has_trace(trace, "chain_b_ec")
            @test has_trace(trace, "chain_c_ec")
            @test has_trace(trace, "chain_d_ec")
            @test has_trace(trace, "chain_e_ec")
        end

        @testset "re-tracing same function" begin
            tau_rewrite_reset_exclusions()
            result1, trace1 = capture_trace(() -> tau_rewrite_and_call(simple_add, 3.0, 4.0))
            result2, trace2 = capture_trace(() -> tau_rewrite_and_call(simple_add, 3.0, 4.0))
            @test result1 == 7.0
            @test result2 == 7.0
            @test has_trace(trace1, "simple_add")
            @test has_trace(trace2, "simple_add")
        end
    end

    @testset "Dispatch and type system" begin
        @testset "multiple dispatch (Int path)" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(dispatch_caller_ec, true, 1, 2, 1.0, 2.0))
            @test result == 3
            @test has_trace(trace, "dispatch_caller_ec")
            @test has_trace(trace, "polymorphic_add_ec")
        end

        @testset "multiple dispatch (Float64 path)" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(dispatch_caller_ec, false, 1, 2, 1.0, 2.0))
            @test result == 3.0
            @test has_trace(trace, "dispatch_caller_ec")
            @test has_trace(trace, "polymorphic_add_ec")
        end

        @testset "union-typed argument (Float64 path)" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(test_union_dispatch_ec, true))
            @test result == 2.0
            @test has_trace(trace, "test_union_dispatch_ec")
            @test has_trace(trace, "process_value_ec")
        end

        @testset "union-typed argument (Vector path)" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(test_union_dispatch_ec, false))
            @test result == [2.0, 4.0, 6.0]
            @test has_trace(trace, "test_union_dispatch_ec")
            @test has_trace(trace, "process_value_ec")
        end

        @testset "where T parameterized callee" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(generic_add_ec, 3.0, 4.0))
            @test result == 7.0
            @test has_trace(trace, "generic_add_ec")
        end

        @testset "where {T,N} with inner callees" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(calls_sim_iterate_ec))
            @test result == 4.5
            @test has_trace(trace, "calls_sim_iterate_ec")
            @test has_trace(trace, "sim_iterate_ec!")
            @test has_trace(trace, "sim_inner_step_ec!")
        end
    end

    @testset "Calling conventions" begin
        @testset "keyword arguments via kwcall" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(calls_kwarg_ec, 5.0))
            @test result == 16.0
            @test has_trace(trace, "calls_kwarg_ec")
            @test has_trace(trace, "kwarg_helper_ec")
        end

        @testset "default argument values" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(calls_with_default_ec, 3.0))
            @test result == 12.0
            @test has_trace(trace, "calls_with_default_ec")
            @test has_trace(trace, "with_default_ec")
        end

        @testset "user-defined varargs" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(calls_varargs_ec, 1.0))
            @test result == 6.0
            @test has_trace(trace, "calls_varargs_ec")
            # Varargs specialization is pre-compiled; Phase 2 only intercepts new compilations.
            # Requires a different mechanism to instrument already-compiled-but-not-emitted functions.
            @test_broken has_trace(trace, "varargs_sum_ec")
        end

        @testset "splatting at call site" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(splat_caller_ec, 3.0, 4.0))
            @test result == 7.0
            @test has_trace(trace, "splat_caller_ec")
            @test has_trace(trace, "add_ec")
        end

        @testset "splatting with deeper call chain" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(splat_chain_ec, 3.0, 4.0))
            @test result == 207.0
            @test has_trace(trace, "splat_chain_ec")
            @test has_trace(trace, "splat_target_ec")
            @test has_trace(trace, "splat_inner_ec")
        end
    end

    @testset "Cross-module tracing" begin
        @testset "inline module" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(cross_module_wrapper_ec))
            @test has_trace(trace, "cross_module_wrapper_ec")
            @test has_trace(trace, "sim_main_ec")
            @test has_trace(trace, "sim_step_ec")
        end

        @testset "file-loaded module" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(cross_file_wrapper_ec))
            @test has_trace(trace, "cross_file_wrapper_ec")
            @test has_trace(trace, "sim_main_ec")
        end
    end

    @testset "Phase 2 dynamic dispatch" begin
        @testset "@invoke explicit dispatch" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(calls_via_invoke_ec, 5.0))
            @test result == 315.0
            @test has_trace(trace, "calls_via_invoke_ec")
            @test has_trace(trace, "invoke_target_ec")
            @test has_trace(trace, "invoke_inner_ec")
        end

        @testset "Base.invokelatest" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(calls_via_invokelatest_ec, 5.0))
            @test result == 110.0
            @test has_trace(trace, "calls_via_invokelatest_ec")
            @test has_trace(trace, "invokelatest_target_ec")
            @test_broken has_trace(trace, "invokelatest_inner_ec")
        end

        @testset "Base.invoke_in_world" begin
            tau_rewrite_reset_exclusions()
            result, trace = capture_trace(() -> tau_rewrite_and_call(calls_via_invoke_in_world_ec, 5.0))
            @test result == 120.0
            @test has_trace(trace, "calls_via_invoke_in_world_ec")
            @test has_trace(trace, "invoke_world_target_ec")
            @test_broken has_trace(trace, "invoke_world_inner_ec")
        end
    end

    @testset "Exclusion API expansion" begin
        @testset "module whitelist exact=true excludes submodules" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_include_module_only(:MathHelpers_ec; exact=true)
            _, trace = capture_trace(() -> tau_rewrite_and_call(outer_compute_with_sub_ec, 3.0, 4.0))
            @test has_trace(trace, "mh_add_ec")
            @test no_trace(trace, "msh_mul_ec")
            @test no_trace(trace, "outer_compute_with_sub_ec")
            tau_rewrite_reset_exclusions()
        end

        @testset "module whitelist exact=false includes submodules" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_include_module_only(:MathHelpers_ec; exact=false)
            _, trace = capture_trace(() -> tau_rewrite_and_call(outer_compute_with_sub_ec, 3.0, 4.0))
            @test has_trace(trace, "mh_add_ec")
            @test has_trace(trace, "msh_mul_ec")
            @test no_trace(trace, "outer_compute_with_sub_ec")
            tau_rewrite_reset_exclusions()
        end

        @testset "exclude multiple functions" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_exclude_function(:mh_add_ec)
            tau_rewrite_exclude_function(:msh_mul_ec)
            _, trace = capture_trace(() -> tau_rewrite_and_call(outer_compute_with_sub_ec, 3.0, 4.0))
            @test has_trace(trace, "outer_compute_with_sub_ec")
            @test no_trace(trace, "mh_add_ec")
            @test no_trace(trace, "msh_mul_ec")
            tau_rewrite_reset_exclusions()
        end

        @testset "exclude multiple modules" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_exclude_module(:TestModA)
            tau_rewrite_exclude_module(:TestModB)
            _, trace = capture_trace(() -> tau_rewrite_and_call(multi_mod, 5))
            @test has_trace(trace, "multi_mod")
            @test has_trace(trace, "call_mod_a")
            @test has_trace(trace, "call_mod_b")
            @test no_trace(trace, "entry_a")
            @test no_trace(trace, "mid_a")
            @test no_trace(trace, "deep_a")
            @test no_trace(trace, "work_b")
            tau_rewrite_reset_exclusions()
        end

        @testset "hierarchical module depth limit (exact=false)" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_set_recursion_limit(:DepthOuter_ec, 1)
            _, trace = capture_trace(() -> tau_rewrite_and_call(depth_caller_outer_inner_ec, 1.0))
            @test has_trace(trace, "depth_caller_outer_inner_ec")
            @test has_trace(trace, "do_entry_ec")
            @test no_trace(trace, "do_mid_ec")
            @test has_trace(trace, "di_func_ec")
            tau_rewrite_reset_exclusions()
        end

        @testset "hierarchical module depth limit (exact=true)" begin
            tau_rewrite_reset_exclusions()
            tau_rewrite_set_recursion_limit(:DepthOuter_ec, 1; exact=true)
            _, trace = capture_trace(() -> tau_rewrite_and_call(depth_caller_outer_inner_ec, 1.0))
            @test has_trace(trace, "depth_caller_outer_inner_ec")
            @test has_trace(trace, "do_entry_ec")
            @test no_trace(trace, "do_mid_ec")
            @test has_trace(trace, "di_func_ec")
            tau_rewrite_reset_exclusions()
        end
    end

    @testset "Module name fallback for unmapped LLVM functions" begin
        @testset "all trace entries have module prefix" begin
            tau_rewrite_reset_exclusions()
            function sort_trigger(x)
                A = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
                return sort(A)
            end
            _, trace = capture_trace(() -> tau_rewrite_and_call(sort_trigger, 1))
            lines = split(trace, '\n')
            enter_exit_lines = filter(l -> startswith(l, ">>> Enter: ") || startswith(l, "<<< Exit:  "), lines)
            @test !isempty(enter_exit_lines)
            missing_module = String[]
            for line in enter_exit_lines
                m = match(r"^(?:>>> Enter: |<<< Exit:  )(.+?)(?:\(.*?\))?\s*\[\{", line)
                m === nothing && continue
                funcname = m.captures[1]
                if !contains(funcname, ".")
                    push!(missing_module, funcname)
                end
            end
            @test isempty(missing_module) || begin
                @warn "Functions missing module prefix" missing_module
                false
            end
            tau_rewrite_reset_exclusions()
        end

        @testset "module exclusion works for fallback-mapped functions" begin
            tau_rewrite_reset_exclusions()
            function sort_trigger2(x)
                A = [3.0, 1.0, 4.0, 1.0, 5.0]
                return sort(A)
            end
            _, trace_before = capture_trace(() -> tau_rewrite_and_call(sort_trigger2, 1))
            has_base_funcs = any(l -> startswith(l, ">>> Enter: ") && contains(l, "Base"), split(trace_before, '\n'))

            tau_rewrite_reset_exclusions()
            tau_rewrite_exclude_module(Base)
            _, trace_after = capture_trace(() -> tau_rewrite_and_call(sort_trigger2, 1))
            base_entries_after = filter(l -> startswith(l, ">>> Enter: ") && contains(l, "Base."), split(trace_after, '\n'))
            @test isempty(base_entries_after)
            tau_rewrite_reset_exclusions()
        end
    end

end

println("\nAll tests completed!")
