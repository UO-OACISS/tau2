#=
test_tau_hooks.jl — Tau_start/Tau_stop tests.
=#

using TAUProfile
using Test

TAUProfile._force_text_hooks[] = false

const _TAU_LIB = get(ENV, "TAU_JULIA_LIB", "")
const _TAU_OK  = !isempty(_TAU_LIB) && isfile(_TAU_LIB) && TAUProfile._tau_active()

@noinline _tauhook_fn(x) = x * 2 + 1

@testset "TAU-mode hooks (Route B)" begin
    if !_TAU_OK
        @info "TAU_JULIA_LIB not set/invalid; skipping TAU-mode hook tests"
        @test_skip "TAU mode unavailable"
    else
        ep, xp, tau = TAUProfile._active_hook_ptrs()
        @test tau === true
        @test ep == TAUProfile.TAU_START_FPTR[]
        @test xp == TAUProfile.TAU_STOP_FPTR[]

        ir = trace_code(_tauhook_fn, 1.0)
        start_int = string(reinterpret(UInt, TAUProfile.TAU_START_FPTR[]))
        stop_int  = string(reinterpret(UInt, TAUProfile.TAU_STOP_FPTR[]))

        @test occursin("store i8 1", ir)            # sticky pin emitted
        @test count("store i8 1", ir) == 1
        @test count(start_int, ir) == 1             # exactly one entry hook
        @test count(stop_int, ir) >= 1              # at least one exit hook
        # total embedded hook calls == entry(1) + number of exit paths
        @test count("call void inttoptr", ir) == count(start_int, ir) + count(stop_int, ir)

        mktempdir() do dir
            script = joinpath(dir, "b2_run.jl")
            write(script, """
                using TAUProfile
                @noinline _b2_timer_fn(x) = x * 2 + 1
                tau_rewrite_and_call(_b2_timer_fn, 21.0)
                """)
            projdir = dirname(dirname(pathof(TAUProfile)))
            cmd = Cmd(`$(Base.julia_cmd()) --startup-file=no --project=$projdir $script`; dir=dir)
            run(cmd)
            profs = filter(f -> startswith(f, "profile."), readdir(dir))
            @test !isempty(profs)
            content = join(read(joinpath(dir, p), String) for p in profs)
            @test occursin("_b2_timer_fn", content)
        end
    end
end
