#=
test_tau_api.jl — direct TAU API bindings (tau_api.jl): handle-based timers,
profile readback, user events, metadata, dynamic timers, runtime control,
mid-run dumps and snapshots.

The tests can run with or without TAU_JULIA_LIB. Without it, every call is
a no-op; with it, actual TAU calls are made and the readback functions let us 
make assertions on TAU state in-process, and exit-time profiles from a
subprocess cover the rest.
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
    :TauEvent, :tau_event, :tau_context_event,
    :tau_metadata, :tau_context_metadata,
    :tau_dynamic_start, :tau_dynamic_stop,
    :tau_enable_instrumentation, :tau_disable_instrumentation,
    :tau_enable_group, :tau_disable_group,
    :tau_enable_all_groups, :tau_disable_all_groups,
    :tau_dump, :tau_snapshot, :tau_exit,
)

# Run `body` (Julia source) in a fresh process with cwd = a temp dir. Return
# the concatenated contents of the profile.* files TAU wrote at exit,
# everything the process printed to stderr, and the contents of any other
# file the run left in the directory (mid-run dumps, snapshots), by name.
function _run_of(body::String)
    mktempdir() do dir
        script = joinpath(dir, "run.jl")
        write(script, "using TAUProfile\n" * body)
        projdir = dirname(dirname(pathof(TAUProfile)))
        errbuf = IOBuffer()
        run(pipeline(Cmd(`$(Base.julia_cmd()) --startup-file=no --project=$projdir $script`; dir=dir);
                     stderr=errbuf))
        files = readdir(dir)
        profs = filter(f -> startswith(f, "profile."), files)
        @test !isempty(profs)
        content = join(read(joinpath(dir, p), String) for p in profs)
        others = Dict(f => read(joinpath(dir, f), String)
                      for f in files if !(f in profs) && f != "run.jl")
        (content, String(take!(errbuf)), others)
    end
end

_profile_of(body::String) = first(_run_of(body))

@testset "TAU API bindings" begin
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

    @testset "TauTimer and TauEvent are plain isbits handles" begin
        @test isbitstype(TauTimer)
        @test isbitstype(TauEvent)
    end

    @testset "events and metadata reject empty names in both modes" begin
        @test_throws ArgumentError TauEvent("")
        @test_throws ArgumentError tau_event("", 1.0)
        @test_throws ArgumentError tau_context_event(" ", 1.0)
        @test_throws ArgumentError tau_metadata("", "v")
        @test_throws ArgumentError tau_context_metadata("", "v")
    end

    @testset "dynamic timers reject an empty name in both modes" begin
        @test_throws ArgumentError tau_dynamic_start("")
        @test_throws ArgumentError tau_dynamic_stop(" ")
    end

    @testset "group toggles, dump and snapshot reject empty names in both modes" begin
        @test_throws ArgumentError tau_enable_group("")
        @test_throws ArgumentError tau_disable_group(" ")
        @test_throws ArgumentError tau_dump("")
        @test_throws ArgumentError tau_snapshot(" ")
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
            e = TauEvent("jtapi_noop_evt")
            @test e.ptr == C_NULL
            @test tau_event(e, 1.0) === nothing
            @test tau_event("jtapi_noop_evt", 1) === nothing
            @test tau_context_event("jtapi_noop_evt", 1.0) === nothing
            @test tau_metadata("jtapi_noop_meta", 42) === nothing
            @test tau_context_metadata("jtapi_noop_meta", "v") === nothing
            @test tau_dynamic_start("jtapi_noop_dyn") === nothing
            @test tau_dynamic_stop("jtapi_noop_dyn") === nothing
            @test tau_disable_instrumentation() === nothing
            @test tau_enable_instrumentation() === nothing
            @test tau_disable_group("TAU_USER") === nothing
            @test tau_enable_group("TAU_USER") === nothing
            @test tau_disable_all_groups() === nothing
            @test tau_enable_all_groups() === nothing
            @test tau_dump() === nothing
            @test tau_dump("jtapi_noop_dump") === nothing
            @test tau_snapshot("jtapi_noop_snap") === nothing
            @test tau_exit("jtapi_noop_exit") === nothing
            # Nothing was written: no libTAU, no files.
            @test !any(f -> startswith(f, "jtapi_noop_dump."), readdir())
        end
    else
        @testset "start/stop on a handle counts calls" begin
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

        @testset "same name resolves to the same FunctionInfo" begin
            a = TauTimer("jtapi_dup")
            b = TauTimer("jtapi_dup")
            @test a.ptr == b.ptr
            tau_start(a); tau_stop(a)
            tau_start(b); tau_stop(b)
            @test tau_get_calls(a) == 2
            @test tau_get_calls(b) == 2
        end

        @testset "@tau accepts a handle and stops it on exceptional exit" begin
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

        @testset "child calls count nested timers" begin
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

        @testset "inclusive and exclusive values, one per counter" begin
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

        @testset "type and group land in the profile" begin
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

        @testset "user events by name and by handle" begin
            content = _profile_of("""
                for v in (1.0, 3.0, 2.0)
                    tau_event("jtapi_evt_name", v)
                end
                e = TauEvent("jtapi_evt_handle")
                for v in (1, 3, 2)          # Integer values are accepted
                    tau_event(e, v)
                end
                e2 = TauEvent("jtapi_evt_handle")
                tau_event(e2, 2)             # same name -> same event
                """)
            # "name" numevents max min mean sumsqr
            @test occursin("\"jtapi_evt_name\" 3 3 1 2 14\n", content)
            @test occursin("\"jtapi_evt_handle\" 4 3 1 2 18\n", content)
        end

        @testset "context events carry the enclosing timer" begin
            content = _profile_of("""
                t = TauTimer("jtapi_ctx_timer")
                @tau t begin
                    tau_context_event("jtapi_ctx_evt", 5.0)
                end
                tau_context_event("jtapi_ctx_evt_top", 1.0)
                """)
            @test occursin(r"\"jtapi_ctx_evt : [^\"\n]*jtapi_ctx_timer\" 1 5 5 5 25\n", content)
            # At top level the context is the top-level timer.
            @test occursin(r"\"jtapi_ctx_evt_top : \.TAU application\" 1 1 1 1 1\n", content)
        end

        @testset "metadata and context metadata" begin
            content = _profile_of("""
                tau_metadata("jtapi_meta_str", "hello world")
                tau_metadata("jtapi_meta_int", 42)
                tau_metadata("jtapi_meta_float", 2.5)
                t = TauTimer("jtapi_meta_timer")
                @tau t begin
                    tau_context_metadata("jtapi_ctx_meta", 7)
                end
                """)
            @test occursin(r"<name>jtapi_meta_str</name>\s*<value>hello world</value>", content)
            @test occursin(r"<name>jtapi_meta_int</name>\s*<value>42</value>", content)
            @test occursin(r"<name>jtapi_meta_float</name>\s*<value>2\.5</value>", content)
            # TAU writes the context as "name type"; with an empty type that
            # leaves a trailing space inside the element.
            @test occursin(r"<name>jtapi_ctx_meta</name>\s*<timer_context>jtapi_meta_timer\s*</timer_context>", content)
            @test occursin(r"<name>jtapi_ctx_meta</name>[^\n]*?<value>7</value>", content)
        end

        @testset "dynamic timers get one entry per iteration" begin
            K = 3
            for _ in 1:K
                @test tau_dynamic_start("jtapi_dyn") === nothing
                @test tau_dynamic_stop("jtapi_dyn") === nothing
            end
            fnames = tau_function_names()
            for i in 0:K-1
                @test "jtapi_dyn[$i]" in fnames
                # A handle looked up by the suffixed name is the iteration's
                # own timer, and it ran exactly once.
                @test tau_get_calls(TauTimer("jtapi_dyn[$i]")) == 1
            end
            @test !("jtapi_dyn[$K]" in fnames)
            @test !("jtapi_dyn" in fnames)
        end

        @testset "dynamic timers nest and land in the profile" begin
            content = _profile_of("""
                for _ in 1:2
                    tau_dynamic_start("jtapi_dyn_outer")
                    tau_dynamic_start("jtapi_dyn_inner")
                    tau_dynamic_stop("jtapi_dyn_inner")
                    tau_dynamic_stop("jtapi_dyn_outer")
                end
                """)
            # "name" calls subrs excl incl profilecalls GROUP="..."; TAU
            # writes dynamic timers with an empty group name.
            for i in 0:1
                @test occursin("\"jtapi_dyn_outer[$i]\" 1 1 ", content)
                @test occursin("\"jtapi_dyn_inner[$i]\" 1 0 ", content)
            end
        end

        @testset "unmatched dynamic stop reports TAU's message, no Julia error" begin
            content, err = _run_of("""
                tau_dynamic_stop("jtapi_dyn_never_started")
                tau_dynamic_start("jtapi_dyn_after")
                tau_dynamic_stop("jtapi_dyn_after")
                """)
            @test occursin("Routine \"jtapi_dyn_never_started\" does not exist", err)
            @test !occursin("jtapi_dyn_never_started[", content)
            # The process kept going and wrote a normal profile.
            @test occursin("\"jtapi_dyn_after[0]\" 1 0 ", content)
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

        @testset "timers inside a disabled-instrumentation window are not counted" begin
            t = TauTimer("jtapi_disabled_handle")
            try
                @test tau_disable_instrumentation() === nothing
                tau_start(t); tau_stop(t)
                tau_start("jtapi_disabled_name"); tau_stop("jtapi_disabled_name")
            finally
                @test tau_enable_instrumentation() === nothing
            end
            @test tau_get_calls(t) == 0
            @test tau_get_calls(TauTimer("jtapi_disabled_name")) == 0
            tau_start(t); tau_stop(t)
            @test tau_get_calls(t) == 1
        end

        @testset "disabled-window timers are absent from the profile" begin
            content = _profile_of("""
                tau_disable_instrumentation()
                @tau "jtapi_warm_only" begin 1 + 1 end
                tau_enable_instrumentation()
                @tau "jtapi_measured" begin 1 + 1 end
                """)
            @test !occursin("jtapi_warm_only", content)
            @test occursin("\"jtapi_measured\" 1 0 ", content)
        end

        @testset "disabling instrumentation around the compiling call drops the warmup" begin
            content = _profile_of("""
                @noinline _jtapi_warm(x) = x * 2 + 1
                tau_disable_instrumentation()
                tau_rewrite_and_call(_jtapi_warm, 1.0)
                tau_enable_instrumentation()
                tau_rewrite_and_call(_jtapi_warm, 2.0)
                """)
            # The callee ran twice but only the second, measured, call counts.
            @test occursin(r"\"Main\._jtapi_warm \[[^\]]*\]\" 1 0 ", content)
        end

        @testset "group toggles by name suppress and restore timers" begin
            t = TauTimer("jtapi_grouped"; group="JTAPI_GROUP")
            named = TauTimer("jtapi_grouped_user")   # default group TAU_USER
            try
                @test tau_disable_group("JTAPI_GROUP") === nothing
                tau_start(t); tau_stop(t)
                # Other groups keep running.
                tau_start(named); tau_stop(named)
                tau_start("jtapi_grouped_by_name"); tau_stop("jtapi_grouped_by_name")
            finally
                @test tau_enable_group("JTAPI_GROUP") === nothing
            end
            @test tau_get_calls(t) == 0
            @test tau_get_calls(named) == 1
            @test tau_get_calls(TauTimer("jtapi_grouped_by_name")) == 1
            tau_start(t); tau_stop(t)
            @test tau_get_calls(t) == 1

            # Name-based timers (tau_start(name), @tau "name") live in TAU_USER
            # too, so disabling that group covers them as well as handles.
            try
                tau_disable_group("TAU_USER")
                tau_start(named); tau_stop(named)
                tau_start("jtapi_grouped_by_name"); tau_stop("jtapi_grouped_by_name")
            finally
                tau_enable_group("TAU_USER")
            end
            @test tau_get_calls(named) == 1
            @test tau_get_calls(TauTimer("jtapi_grouped_by_name")) == 1
        end

        @testset "all-groups toggles" begin
            t = TauTimer("jtapi_all_groups")
            u = TauTimer("jtapi_all_groups_util"; group="TAU_UTILITY")
            try
                @test tau_disable_all_groups() === nothing
                tau_start(t); tau_stop(t)
                tau_start(u); tau_stop(u)
            finally
                @test tau_enable_all_groups() === nothing
            end
            @test tau_get_calls(t) == 0
            @test tau_get_calls(u) == 0
            tau_start(t); tau_stop(t)
            tau_start(u); tau_stop(u)
            @test tau_get_calls(t) == 1
            @test tau_get_calls(u) == 1
        end

        @testset "tau_dump writes files mid-run" begin
            _, _, others = _run_of("""
                t = TauTimer("jtapi_before_dump")
                tau_start(t); tau_stop(t)
                tau_dump()
                tau_dump("jtapi_pre")
                u = TauTimer("jtapi_after_dump")
                tau_start(u); tau_stop(u)
                """)
            dumps = filter(f -> startswith(f, "dump."), collect(keys(others)))
            pres  = filter(f -> startswith(f, "jtapi_pre."), collect(keys(others)))
            @test length(dumps) == 1
            @test length(pres) == 1
            for f in vcat(dumps, pres)
                # The dump is a complete profile, in the usual format, of what
                # had run at the time it was taken.
                @test startswith(others[f], r"\d+ templated_functions_MULTI_TIME\n")
                @test occursin("\"jtapi_before_dump\" 1 0 ", others[f])
                @test !occursin("jtapi_after_dump", others[f])
            end
        end

        @testset "tau_snapshot records a named snapshot" begin
            _, _, others = _run_of("""
                t = TauTimer("jtapi_snapped")
                tau_start(t); tau_stop(t)
                tau_snapshot("jtapi_snap")
                """)
            snaps = filter(f -> startswith(f, "snapshot."), collect(keys(others)))
            @test length(snaps) == 1
            @test occursin("<name>jtapi_snap</name>", others[first(snaps)])
            @test occursin("jtapi_snapped", others[first(snaps)])
        end

        @testset "tau_exit flushes the profile before exit()" begin
            content = _profile_of("""
                t = TauTimer("jtapi_exit_timer")
                tau_start(t); tau_stop(t)
                tau_exit("jtapi done")
                exit(0)
                """)
            @test occursin("\"jtapi_exit_timer\" 1 0 ", content)
        end
    end
end
