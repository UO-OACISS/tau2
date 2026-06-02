nthreads_os() = length(readdir("/proc/self/task"))

function task_names()
    foreach(sort(readdir("/proc/self/task"))) do tid
        println(tid, "  ", strip(read("/proc/self/task/$tid/comm", String)))
    end
end

println("OS threads before import: ", nthreads_os())
using LinearAlgebra
println("OS threads after import: ", nthreads_os())

using TAUProfile

n = 2000
A = rand(n, n)
B = rand(n, n)

println("OS threads before any BLAS work: ", nthreads_os())

BLAS.set_num_threads(1)
A * B

@tau_func function do_blas(nt) 
    BLAS.set_num_threads(nt)
    @assert BLAS.get_num_threads() == nt
    A * B
    println("\nBLAS threads = $nt | OS threads = $(nthreads_os())")
    print("  gemm time: ")
    @time A * B
end

for nt in (1, 2, 4, 8)
    do_blas(nt)
end
