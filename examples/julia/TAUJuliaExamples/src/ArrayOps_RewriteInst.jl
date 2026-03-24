using Printf: @printf
using TAUProfile

function profile_test(n)
    @printf "Running loop of size %d\n" n
    for i = 1:n
        A = randn(100,100,20)
        m = maximum(A)
        Am = mapslices(sum, A; dims=2)
        B = A[:,:,5]
        Bsort = mapslices(sort, B; dims=1)
        b = rand(100)
        C = B.*b
    end
end

function main()
    profile_test(1)
    profile_test(10)
    profile_test(100)
    profile_test(1000)
    println("Done")
end

tau_rewrite_set_recursion_limit(4)
tau_rewrite_set_recursion_limit(Base, 2)
@tau_rewrite main()

