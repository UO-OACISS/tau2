# TimerOutputs.jl sections as TAU timers.
# Run under tau_julia (or tau_exec ... julia).
# Regions annotated with TimerOutputs.@timeit are automatically
# instrumented with TAU in addition to TimerOutputs.
using TimerOutputs

const sweep_timers = Dict{Int,TimerOutput}()

function assemble(to, n)
    @timeit to "assemble" begin
        A = rand(n, n)
        @timeit to "symmetrize" A = (A + A') / 2
        A
    end
end

function solve(to, A)
    @timeit to "solve" begin
        @timeit to "factorize" F = cholesky(A + size(A, 1) * I)
        @timeit to "substitute" x = F \ ones(size(A, 1))
        x
    end
end

using LinearAlgebra

function sweep(sw, n)
    to = get!(TimerOutput, sweep_timers, sw)
    @timeit to "sweep" begin
        A = assemble(to, n)
        x = solve(to, A)
        # A section whose body throws still closes its timer on both sides.
        try
            @timeit to "may_fail" sw == 2 && error("bad sweep")
        catch
        end
        @notimeit to sum(x)
    end
end

for sw in 1:3
    sweep(sw, 400)
end

# TimerOutputs' own report is unchanged.
print_timer(merge(values(sweep_timers)...))
println()
