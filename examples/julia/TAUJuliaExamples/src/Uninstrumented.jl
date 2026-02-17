using Printf: @printf

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

# Compile code once
profile_test(1)
profile_test(1000)
profile_test(10000)
println("Done")
