using TAUProfile

function fibonacci(n)
    if n <= 1
        return n
    end
    return fibonacci(n-1) + fibonacci(n-2)
end

function recursive_example()
    result = fibonacci(10)
    return result
end

my_sum(arr) = sum(arr)

function rewrite_example()
    println(recursive_example())
    arr = collect(1:100)
    s = my_sum(arr)
    println(s)
end

@tau_rewrite rewrite_example()

