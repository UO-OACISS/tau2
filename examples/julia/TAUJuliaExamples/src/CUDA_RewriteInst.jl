using CUDA
using TAUProfile

function main()
    a = CUDA.rand(128,128,128)
    b = sin.(a)
    synchronize()
    b_cpu = Array(b)
    println("Done")
end

@tau_rewrite main()

