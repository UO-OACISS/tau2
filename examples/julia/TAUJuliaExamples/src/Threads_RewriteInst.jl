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

function worker_inst(id::Int, n::Int)
    @tau_rewrite worker(id, n)
end

function main()
    println("Running with $(nthreads()) thread(s)\n")

    num_workers = 100
    chunk_size  = 1_000_000

    tasks = [@spawn(worker_inst(i, chunk_size)) for i in 1:num_workers]

    results = fetch.(tasks)

    println("\nAll workers done.")
    println("Results: $(round.(results; digits=4))")
    println("Sum:     $(round(sum(results); digits=4))")
end

set_rewrite_recursion_limit(Base, 2)
@tau_rewrite main()
#main()

