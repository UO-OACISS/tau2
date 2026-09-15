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

# Multi-threaded TAU-mode run. libTAU only registers Julia's worker threads
# when it is preloaded, so this has to go through the installed tau_julia
# (located from TAU_JULIA_LIB: <tau>/<arch>/lib/shared-<tags>/libTAU.so).
function _tau_julia_for(lib::String)
    libdir = dirname(lib)
    tags = basename(libdir)
    startswith(tags, "shared-") || return nothing
    tau_julia = joinpath(dirname(dirname(libdir)), "bin", "tau_julia")
    isfile(tau_julia) || return nothing
    return tau_julia, join(split(chopprefix(tags, "shared-"), "-"), ",")
end

# One line per timer in a TAU profile: "name" calls subrs excl incl profcalls GROUP=...
function _profile_calls(content::String, name::String)
    total = 0
    for m in eachmatch(Regex("^\\\"[^\\\"]*" * name * "[^\\\"]*\\\" (\\d+) ", "m"), content)
        total += parse(Int, m.captures[1])
    end
    return total
end

@testset "TAU-mode hooks under tau_julia with worker threads" begin
    located = _TAU_OK ? _tau_julia_for(_TAU_LIB) : nothing
    if located === nothing
        @info "tau_julia not found next to TAU_JULIA_LIB; skipping the threaded TAU-mode test"
        @test_skip "tau_julia unavailable"
    else
        tau_julia, tags = located
        nthreads = 4
        nitems = 2 * nthreads
        mktempdir() do dir
            script = joinpath(dir, "mt_run.jl")
            write(script, """
                using TAUProfile
                @noinline function _mt_work(i::Int)
                    s = 0
                    for k in 1:1000
                        s += k * i
                    end
                    return s
                end
                function _mt_driver(n::Int)
                    out = zeros(Int, n)
                    Threads.@threads :static for i in 1:n
                        out[i] = Base.invokelatest(_mt_work, i)
                    end
                    return out
                end
                r = tau_rewrite_and_call(_mt_driver, $nitems)
                expected = [sum(k * i for k in 1:1000) for i in 1:$nitems]
                println("RESULT_OK=", r == expected)
                println("NTHREADS=", Threads.nthreads())
                """)
            projdir = dirname(dirname(pathof(TAUProfile)))
            cmd = Cmd(`$tau_julia -T $tags -- -t $nthreads --startup-file=no --project=$projdir $script`; dir=dir)
            outbuf = IOBuffer(); errbuf = IOBuffer()
            ok = success(pipeline(ignorestatus(cmd); stdout=outbuf, stderr=errbuf))
            out = String(take!(outbuf)); err = String(take!(errbuf))
            @test ok
            @test occursin("RESULT_OK=true", out)
            @test occursin("NTHREADS=$nthreads", out)
            @test !occursin("Phase 2 instrumentation failed", err)
            @test !occursin("Runtime overlap", err)
            @test !occursin("Failed to instrument", err)

            # One profile per TAU thread; :static scheduling puts exactly two
            # items on each of the four worker threads, and every call of the
            # Phase 2-patched _mt_work lands in the profile of the thread that
            # ran it.
            profs = sort(filter(f -> startswith(f, "profile."), readdir(dir)))
            @test length(profs) >= nthreads
            calls = Dict(p => _profile_calls(read(joinpath(dir, p), String), "_mt_work") for p in profs)
            @test count(==(nitems ÷ nthreads), values(calls)) == nthreads
            @test sum(values(calls)) == nitems
        end
    end
end
