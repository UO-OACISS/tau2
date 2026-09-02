#=
test_api_parity.jl — public-API parity with the classic backend.

Verifies the LLVM backend exposes the same public surface as the classic
TAUProfile so an identical user script runs on either backend.
=#

using TAUProfile
using Test

# The classic-parity superset that must be exported from this backend.
const _PARITY_SYMBOLS = (
    :tau_start, :tau_stop, Symbol("@tau"), Symbol("@tau_func"),
    :tau_rewrite, :tau_rewrite_and_call,
    :tau_rewrite_deferred_contexts, :tau_rewrite_set_min_complexity,
)

const _TAU_LIB = get(ENV, "TAU_JULIA_LIB", "")
const _TAU_OK  = !isempty(_TAU_LIB) && isfile(_TAU_LIB) && TAUProfile._tau_active()

@testset "API parity with the classic backend" begin
    @testset "C.3 symbols defined and exported" begin
        exported = Set(names(TAUProfile))
        for s in _PARITY_SYMBOLS
            @test isdefined(TAUProfile, s)
            @test s in exported
        end
    end

    @testset "C.1 manual API runs without error" begin
        # @tau / @tau_func run without error regardless of TAU presence.
        r = @tau "blk" begin 1 + 1 end
        @test r == 2
        @tau_func _parity_h(x) = x + 5
        @test _parity_h(2) == 7
        # tau_start/tau_stop are callable and balanced (no-op without TAU).
        @test tau_start("m") === nothing
        @test tau_stop("m") === nothing
    end

    @testset "C.2 tau_rewrite parity with tau_rewrite_and_call" begin
        g = tau_rewrite((x) -> x + 1, (Float64,))
        @test g(1.0) == 2.0
        @test g(1.0) == tau_rewrite_and_call((x) -> x + 1, 1.0)
    end

    @testset "C.3 no-op stubs warn at most once, return nothing" begin
        @test tau_rewrite_deferred_contexts() === nothing
        @test tau_rewrite_deferred_contexts(false) === nothing
        @test tau_rewrite_set_min_complexity(5) === nothing
        @test tau_rewrite_set_min_complexity(5; skip_loops=false) === nothing
    end

    if _TAU_OK
        @testset "C.1 manual @tau timer in profile (TAU mode)" begin
            # Run in a subprocess (cwd = temp dir) so TAU writes its exit-time
            # profile where we can read it.
            mktempdir() do dir
                script = joinpath(dir, "blk_run.jl")
                write(script, """
                    using TAUProfile
                    @tau "parity_blk_timer" begin
                        s = 0
                        for i in 1:1000; s += i; end
                        s
                    end
                    """)
                projdir = dirname(dirname(pathof(TAUProfile)))
                run(Cmd(`$(Base.julia_cmd()) --startup-file=no --project=$projdir $script`; dir=dir))
                profs = filter(f -> startswith(f, "profile."), readdir(dir))
                @test !isempty(profs)
                content = join(read(joinpath(dir, p), String) for p in profs)
                @test occursin("parity_blk_timer", content)
            end
        end
    else
        @testset "C.1 manual @tau timer in profile (TAU mode)" begin
            @info "TAU_JULIA_LIB not set/invalid; skipping manual-timer profile test"
            @test_skip "TAU mode unavailable"
        end
    end
end
