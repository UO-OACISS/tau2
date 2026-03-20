using Base.Threads
using TAUProfile

function worker(id::Int, n::Int)
    println("  Worker $id started on thread $(threadid())")
    total = zero(Float64)
    for i in 1:n
        total += sin(Float64(i) * 0.001)
    end
    println("  Worker $id finished on thread $(threadid()), result = $(round(total; digits=4))")
    return total
end

function main()
    println("Running with $(nthreads()) thread(s)\n")

    num_workers = 100
    chunk_size  = 1_000_000

    tasks = [@spawn(worker(i, chunk_size)) for i in 1:num_workers]

    results = fetch.(tasks)

    println("\nAll workers done.")
    println("Results: $(round.(results; digits=4))")
    println("Sum:     $(round(sum(results); digits=4))")
end

tau_rewrite_exclude_module(Base)
tau_rewrite_deferred_contexts(true)
@tau_rewrite main()

