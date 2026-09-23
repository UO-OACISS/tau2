#=
test_timeroutputs.jl

Runs under `--project=test/timeroutputs/v1` (TimerOutputs 1.x) or `.../v0.5` (0.5.x).

Modes:
  - no TAU:    Without TAU, the standard TimerOutputs timers should still run.
  - dlopen:    Instrument with TAU via libTAU dlopen'ed by Julia
  - tau_exec:  Instrument iwth TAU via LD_PRELOAD'ed libTAU
=#

using TimerOutputs
using Test

const _HOOK = normpath(joinpath(@__DIR__, "..", "..", "..", "julia_timeroutputs", "tau_timeroutputs.jl"))
const _TO_LIB = get(ENV, "TAU_JULIA_LIB", "")
const _TO_TAU_OK = !isempty(_TO_LIB) && isfile(_TO_LIB)
const _TO_V1 = pkgversion(TimerOutputs) >= v"1"

# tau_exec and the -T tags follow from the binding directory of TAU_JULIA_LIB.
const _TAU_EXEC = _TO_TAU_OK ? joinpath(dirname(dirname(dirname(_TO_LIB))), "bin", "tau_exec") : ""
const _TAU_TAGS = _TO_TAU_OK ? replace(chopprefix(basename(dirname(_TO_LIB)), "shared-"), "-" => ",") : ""
const _TO_EXEC_OK = _TO_TAU_OK && isfile(_TAU_EXEC) && startswith(basename(dirname(_TO_LIB)), "shared-")

@test isfile(_HOOK)

# Run `body` in a fresh process with the hook loaded via -L
function _to_run(body::String; mode::Symbol=:dlopen, env=Dict{String,String}(),
                 project::String=Base.active_project(), hook::Bool=true, expect_ok::Bool=true)
    mktempdir() do dir
        script = joinpath(dir, "run.jl")
        write(script, body)
        jl = Base.julia_cmd().exec
        hook = hook && mode !== :auto
        args = hook ? [jl[1], "--startup-file=no", "-L", _HOOK, "--project=$project", script] :
                      [jl[1], "--startup-file=no", "--project=$project", script]
        if mode === :exec || mode === :auto
            args = [_TAU_EXEC, "-T", _TAU_TAGS, args...]
        end
        e = copy(ENV)
        delete!(e, "TAU_TIMEROUTPUTS_REPORT"); delete!(e, "TAU_TIMEROUTPUTS_VERBOSE")
        delete!(e, "TAU_TIMEROUTPUTS_PREFIX"); delete!(e, "TAU_CALLPATH")
        mode === :plain ? delete!(e, "TAU_JULIA_LIB") : (e["TAU_JULIA_LIB"] = _TO_LIB)
        merge!(e, env)
        outbuf, errbuf = IOBuffer(), IOBuffer()
        ok = success(pipeline(Cmd(Cmd(args); dir=dir, env=e); stdout=outbuf, stderr=errbuf))
        out, err = String(take!(outbuf)), String(take!(errbuf))
        ok == expect_ok || @error "subprocess exit status not as expected" mode expect_ok args out err
        @test ok == expect_ok
        profs = filter(f -> startswith(f, "profile."), readdir(dir))
        content = join(read(joinpath(dir, p), String) for p in profs)
        (_rows(content), out, err, profs)
    end
end


function _rows(content::String)
    rows = Dict{String,Tuple{Int,Float64,String}}()
    for m in eachmatch(r"^\"(.*)\" (\d+) \d+ ([\d.eE+-]+) ([\d.eE+-]+) \d+ GROUP=\"([^\"]*)\""m, content)
        rows[m.captures[1]] = (parse(Int, m.captures[2]), parse(Float64, m.captures[4]), m.captures[5])
    end
    rows
end

_calls(rows, name) = haskey(rows, name) ? rows[name][1] : 0

# Three per-"sweep" timers passed down as argument
const _TOY = """
using TimerOutputs
const sweep_timers = Dict{Int,TimerOutput}()
function work(stimer, n)
    @timeit stimer "outer" begin
        s = 0.0
        for i in 1:n
            @timeit stimer "inner" (s += sum(rand(8)))
        end
        try
            @timeit stimer "throws" error("boom")
        catch
        end
        s
    end
end
for sw in 1:3
    work(get!(TimerOutput, sweep_timers, sw), 10)
end
to = merge(values(sweep_timers)...)
println("TO outer=", TimerOutputs.ncalls(to["outer"]), " inner=", TimerOutputs.ncalls(to["outer"]["inner"]),
        " throws=", TimerOutputs.ncalls(to["outer"]["throws"]))
print_timer(stdout, to); println()
"""

# Every documented way of opening a section.
const _FORMS = """
using TimerOutputs
const to = TimerOutput()
@timeit to function fdef(x) x + 1 end
fdef(1); fdef(2)
# Expanding @timeit_debug defines the module's `timeit_debug_enabled() = false`;
# enable_debug_timings then flips it, before the first call compiles the body.
g() = @timeit_debug to "dbg" 1 + 1
TimerOutputs.enable_debug_timings(@__MODULE__)
g()
timeit(to, "fn") do; 1 + 1; end
if isdefined(TimerOutputs, :begin_timed_section!)
    s = begin_timed_section!(to, "manual"); end_timed_section!(to, s)
end
if isdefined(TimerOutputs, Symbol("@timeit_all"))
    eval(Meta.parse("@timeit_all to \\"all\\" begin a = 1 + 1; b = a * 2 end"))
end
@timeit to "plain" 1 + 1
"""

# reset_timer! inside a section empties TimerOutputs' stack; the TAU side
# must still close in order, and the next section must not nest under a
# stale parent.
const _RESET = """
using TimerOutputs
to = TimerOutput()
@timeit to "a" begin
    @timeit to "b" begin
        reset_timer!(to)
    end
end
@timeit to "c" 1 + 1
println("done")
"""

const _DISABLED = """
using TimerOutputs
to = TimerOutput()
disable_timer!(to)
@timeit to "off" 1 + 1
enable_timer!(to)
@timeit to "on" 1 + 1
@notimeit to (@timeit to "notimeit" 1 + 1)
println("has_off=", haskey(to, "off"), " has_on=", haskey(to, "on"), " has_notimeit=", haskey(to, "notimeit"))
"""

@testset "TimerOutputs bridge ($(pkgversion(TimerOutputs)))" begin

    @testset "inert without libTAU" begin
        rows, out, err, profs = _to_run(_TOY; mode=:plain)
        @test isempty(profs)
        @test isempty(err)
        @test occursin("TO outer=3 inner=30 throws=3", out)
        @test occursin("outer", out) && occursin("inner", out)   # the table printed
        rows, out, err, profs = _to_run(_TOY; mode=:plain, env=Dict("TAU_TIMEROUTPUTS_VERBOSE" => "1"))
        @test occursin("libTAU not found", err)
    end

    @testset "in-process: install, unknown version, no libTAU" begin
        withenv("TAU_JULIA_LIB" => nothing) do
            Base.include(Main, _HOOK)
        end
        H = Main.TAUTimerOutputs
        @test H.active == false
        @test H.installed[] == true
        to = TimerOutput()
        @timeit to "x" 1 + 1
        @test TimerOutputs.ncalls(to["x"]) == 1
        @test isempty(H.handles)   C             # nothing created without TAU
        module_without_version = Module(:NotAPackage)
        err = mktemp() do path, io
            redirect_stderr(io) do
                H.installed[] = false
                H.install(module_without_version)
                H.installed[] = true
            end
            flush(io)
            read(path, String)
        end
        @test occursin("cannot determine the TimerOutputs version", err)
    end

    modes = Symbol[]
    _TO_TAU_OK && push!(modes, :dlopen)
    _TO_EXEC_OK && push!(modes, :exec)
    # tau_exec injects the installed copy of the hook
    _TO_EXEC_OK && isfile(joinpath(dirname(dirname(_TAU_EXEC)), "lib", "tau_timeroutputs.jl")) && push!(modes, :auto)

    if :auto in modes
        @testset "tau_exec injects the hook by itself" begin
            # dry run shows -L right after the interpreter, for tau_exec and tau_julia
            jl = Base.julia_cmd().exec[1]
            dry = read(`$_TAU_EXEC -s -T $_TAU_TAGS $jl --startup-file=no x.jl`, String)
            @test occursin(Regex("\\Q$jl\\E -L \\S*tau_timeroutputs\\.jl --startup-file=no x\\.jl"), dry)
            tj = joinpath(dirname(_TAU_EXEC), "tau_julia")
            if isfile(tj)
                dry = read(`$tj -s -T $_TAU_TAGS -- --startup-file=no x.jl`, String)
                @test occursin(r"julia -L \S*tau_timeroutputs\.jl --startup-file=no x\.jl", dry)
            end
            # opt-outs
            @test !occursin("tau_timeroutputs", read(`$_TAU_EXEC -s -no-timeroutputs -T $_TAU_TAGS $jl x.jl`, String))
            rows, out, err, _ = _to_run(_TOY; mode=:auto, env=Dict("TAU_TIMEROUTPUTS" => "0"))
            @test !haskey(rows, "outer")
            @test occursin("TO outer=3", out)
            # non-Julia targets are untouched
            @test !occursin("tau_timeroutputs", read(`$_TAU_EXEC -s -T $_TAU_TAGS /bin/echo hi`, String))
            # TAU_JULIA_LIB is exported for Julia targets (a preset value wins)
            got = read(pipeline(`$_TAU_EXEC -T $_TAU_TAGS $jl --startup-file=no -e 'print(get(ENV, "TAU_JULIA_LIB", ""))'`; stderr=devnull), String)
            @test endswith(got, "libTAU.so") && isfile(got)
            e = copy(ENV); e["TAU_JULIA_LIB"] = "/preset"
            got = read(pipeline(Cmd(`$_TAU_EXEC -T $_TAU_TAGS $jl --startup-file=no -e 'print(ENV["TAU_JULIA_LIB"])'`; env=e); stderr=devnull), String)
            @test got == "/preset"
        end
    end
    if isempty(modes)
        @info "TAU_JULIA_LIB not set; skipping the TAU-mode TimerOutputs tests"
    end

    for mode in modes
        @testset "$mode: sections become TAU timers" begin
            rows, out, err, profs = _to_run(_TOY; mode=mode, env=Dict("TAU_CALLPATH" => "1"))
            @test !isempty(profs)
            @test occursin("TO outer=3 inner=30 throws=3", out)      # TimerOutputs unchanged
            @test _calls(rows, "outer") == 3                           # 3 timers, one TAU timer
            @test _calls(rows, "inner") == 30
            @test _calls(rows, "throws") == 3                          # stopped on exceptional exit
            @test rows["outer"][3] == "TimerOutputs"
            @test _calls(rows, "outer => inner") == 30                 # nesting
            @test _calls(rows, "outer => throws") == 3
            @test !occursin("Tau_stop_timer", err) && !occursin("mismatch", lowercase(err))
            @test rows["outer"][2] >= rows["inner"][2] + rows["throws"][2]
        end

        @testset "$mode: every section form" begin
            rows, out, err, _ = _to_run(_FORMS; mode=mode)
            @test _calls(rows, "fdef") == 2
            @test _calls(rows, "dbg") == 1
            @test _calls(rows, "fn") == 1
            @test _calls(rows, "plain") == 1
            isdefined(TimerOutputs, :begin_timed_section!) && @test _calls(rows, "manual") == 1
            if _TO_V1
                @test _calls(rows, "all") == 1
                # @timeit_all names each statement "<file>:<line>: <source>"
                @test any(k -> endswith(k, ": a = 1 + 1") && _calls(rows, k) == 1, keys(rows))
            end
        end

        @testset "$mode: reset_timer! inside a section" begin
            if _TO_V1
                rows, out, err, _ = _to_run(_RESET; mode=mode, env=Dict("TAU_CALLPATH" => "1"))
                @test occursin("done", out)
                @test _calls(rows, "a") == 1 && _calls(rows, "b") == 1 && _calls(rows, "c") == 1
                @test _calls(rows, "a => b") == 1
                @test !haskey(rows, "a => c") && !haskey(rows, "b => c") && !haskey(rows, "a => b => c")
            else
                rows, out, err, _ = _to_run(_RESET; mode=mode, env=Dict("TAU_CALLPATH" => "1"), expect_ok=false)
                @test occursin("array must be non-empty", err)
                @test _calls(rows, "a") == 1 && _calls(rows, "b") == 1
                @test _calls(rows, "a => b") == 1
            end
            @test !occursin("mismatch", lowercase(err))
        end

        @testset "$mode: disabled timers are skipped on both sides" begin
            rows, out, err, _ = _to_run(_DISABLED; mode=mode)
            @test occursin("has_off=false has_on=true has_notimeit=false", out)
            @test !haskey(rows, "off") && !haskey(rows, "notimeit")
            @test _calls(rows, "on") == 1
        end

        @testset "$mode: prefix, report, verbose" begin
            rows, out, err, _ = _to_run(_TOY; mode=mode,
                env=Dict("TAU_TIMEROUTPUTS_PREFIX" => "sw: ", "TAU_TIMEROUTPUTS_REPORT" => "1",
                         "TAU_TIMEROUTPUTS_VERBOSE" => "1"))
            @test _calls(rows, "sw: outer") == 3 && !haskey(rows, "outer")
            @test occursin("sections are TAU timers", err) && occursin("prefix \"sw: \"", err)
            @test occursin("merged TimerOutputs table (3 timer object(s))", err)
            @test occursin("outer", err) && occursin("inner", err)
        end

        @testset "$mode: no TimerOutputs in the environment" begin
            mktempdir() do emptyenv
                write(joinpath(emptyenv, "Project.toml"), "")
                rows, out, err, profs = _to_run("println(\"ran \", any(id -> id.name == \"TimerOutputs\", keys(Base.loaded_modules)))";
                                                 mode=mode, project=emptyenv)
                @test occursin("ran false", out)
                @test isempty(err)
                @test all(r -> r[3] != "TimerOutputs", values(rows))
            end
        end
    end
end
