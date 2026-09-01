#=
test_tau_api.jl — direct TAU API bindings (tau_api.jl).

Phase 1: handle-based timers and profile readback. Runs in both modes: without
TAU_JULIA_LIB every call is a no-op returning its sentinel; with it the
readback functions let us assert on TAU state in-process.
=#

using TAUProfile
using Test

const _TAU_LIB = get(ENV, "TAU_JULIA_LIB", "")
const _TAU_OK  = !isempty(_TAU_LIB) && isfile(_TAU_LIB) && TAUProfile._tau_active()

const _API_SYMBOLS = (
    :TauTimer, :tau_start, :tau_stop, Symbol("@tau"),
    :tau_get_calls, :tau_get_child_calls,
    :tau_get_inclusive, :tau_get_exclusive,
    :tau_counter_names, :tau_function_names,
    :tau_set_name, :tau_set_type, :tau_set_group,
)

# Run `body` (Julia source) in a fresh process with cwd = a temp dir and
# return the concatenated contents of the profile.* files TAU wrote at exit.
function _profile_of(body::String)
    mktempdir() do dir
        script = joinpath(dir, "run.jl")
        write(script, "using TAUProfile\n" * body)
        projdir = dirname(dirname(pathof(TAUProfile)))
        run(Cmd(`$(Base.julia_cmd()) --startup-file=no --project=$projdir $script`; dir=dir))
        profs = filter(f -> startswith(f, "profile."), readdir(dir))
        @test !isempty(profs)
        join(read(joinpath(dir, p), String) for p in profs)
    end
end

@testset "TAU API bindings: handle timers and readback" begin
    @testset "symbols defined and exported" begin
        exported = Set(names(TAUProfile))
        for s in _API_SYMBOLS
            @test isdefined(TAUProfile, s)
            @test s in exported
        end
    end

    @testset "TauTimer rejects an empty name in both modes" begin
        @test_throws ArgumentError TauTimer("")
        @test_throws ArgumentError TauTimer("   ")
    end

    @testset "TauTimer is a plain isbits handle" begin
        @test isbitstype(TauTimer)
    end

    if !_TAU_OK
        @testset "no-op sentinels without libTAU" begin
            t = TauTimer("jtapi_noop")
            @test t.ptr == C_NULL
            @test tau_start(t) === nothing
            @test tau_stop(t) === nothing
            @test tau_get_calls(t) === nothing
            @test tau_get_child_calls(t) === nothing
            @test tau_get_inclusive(t) == Float64[]
            @test tau_get_exclusive(t) == Float64[]
            @test tau_counter_names() == String[]
            @test tau_function_names() == String[]
            @test tau_set_name(t, "other") === nothing
            @test tau_set_type(t, "other") === nothing
            @test tau_set_group(t, "TAU_IO") === nothing
            r = @tau t begin 1 + 1 end
            @test r == 2
        end
    else
        @testset "start/stop on a handle counts calls (AC1.1, AC2.1)" begin
            t = TauTimer("jtapi_handle_timer")
            @test t.ptr != C_NULL
            n0 = tau_get_calls(t)
            @test n0 isa Int
            for _ in 1:3
                @test tau_start(t) === nothing
                @test tau_stop(t) === nothing
            end
            @test tau_get_calls(t) == n0 + 3
        end

        @testset "same name resolves to the same FunctionInfo (AC1.4)" begin
            a = TauTimer("jtapi_dup")
            b = TauTimer("jtapi_dup")
            @test a.ptr == b.ptr
            tau_start(a); tau_stop(a)
            tau_start(b); tau_stop(b)
            @test tau_get_calls(a) == 2
            @test tau_get_calls(b) == 2
        end

        @testset "@tau accepts a handle and stops it on exceptional exit (AC1.3)" begin
            t = TauTimer("jtapi_macro")
            r = @tau t begin 40 + 2 end
            @test r == 42
            @test tau_get_calls(t) == 1
            @test_throws ErrorException (@tau t begin error("boom") end)
            @test tau_get_calls(t) == 2
            # The timer must have been stopped by the finally block: a nested
            # start/stop of a second handle inside a fresh @tau on `t` records
            # exactly one child call, which it would not if `t` were still
            # open from the failed block.
            inner = TauTimer("jtapi_macro_inner")
            @tau t begin
                tau_start(inner); tau_stop(inner)
            end
            @test tau_get_child_calls(t) == 1
        end

        @testset "child calls count nested timers (AC2.1)" begin
            outer = TauTimer("jtapi_outer")
            inner = TauTimer("jtapi_inner")
            tau_start(outer)
            for _ in 1:4
                tau_start(inner); tau_stop(inner)
            end
            tau_stop(outer)
            @test tau_get_child_calls(outer) == 4
            @test tau_get_child_calls(inner) == 0
        end

        @testset "inclusive and exclusive values, one per counter (AC2.2)" begin
            counters = tau_counter_names()
            @test !isempty(counters)
            @test all(!isempty, counters)
            outer = TauTimer("jtapi_values_outer")
            inner = TauTimer("jtapi_values_inner")
            tau_start(outer)
            tau_start(inner)
            s = 0.0
            for i in 1:200_000; s += sqrt(i); end
            tau_stop(inner)
            tau_stop(outer)
            @test s > 0
            inc = tau_get_inclusive(outer)
            exc = tau_get_exclusive(outer)
            @test inc isa Vector{Float64}
            @test length(inc) == length(counters)
            @test length(exc) == length(counters)
            @test all(inc .>= exc)
            @test inc[1] > 0
            # inner has no children, so its inclusive equals its exclusive.
            @test tau_get_inclusive(inner) == tau_get_exclusive(inner)
        end

        @testset "function names list registered timers" begin
            TauTimer("jtapi_listed")
            fnames = tau_function_names()
            @test fnames isa Vector{String}
            @test "jtapi_listed" in fnames
            @test "jtapi_handle_timer" in fnames
        end

        @testset "setters rename in place" begin
            t = TauTimer("jtapi_rename_before")
            @test tau_set_name(t, "jtapi_rename_after") === nothing
            @test "jtapi_rename_after" in tau_function_names()
            @test tau_set_type(t, "renamed type") === nothing
            @test tau_set_group(t, "TAU_UTILITY") === nothing
        end

        @testset "type and group land in the profile (AC1.2)" begin
            content = _profile_of("""
                t = TauTimer("jtapi_typed"; type="Float64 (Int)", group="TAU_UTILITY")
                tau_start(t); tau_stop(t)
                g = TauTimer("jtapi_regrouped")
                tau_set_group(g, "TAU_IO")
                tau_start(g); tau_stop(g)
                """)
            @test occursin("\"jtapi_typed Float64 (Int)\"", content)
            @test occursin(r"\"jtapi_typed Float64 \(Int\)\"[^\n]*GROUP=\"TAU_UTILITY\"", content)
            @test occursin(r"\"jtapi_regrouped\"[^\n]*GROUP=\"TAU_IO\"", content)
        end

        @testset "rewrite timers still appear after moving onto TauTimer" begin
            content = _profile_of("""
                tau_rewrite_time_phase1(true)
                @noinline _jtapi_rw(x) = x * 2 + 1
                tau_rewrite_and_call(_jtapi_rw, 21.0)
                """)
            @test occursin(".TAU Julia Rewrite", content)
            @test occursin(r"\"\.TAU Julia Rewrite\"[^\n]*GROUP=\"TAU_UTILITY\"", content)
        end
    end
end
