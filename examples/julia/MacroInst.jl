using TAUProfile

function macro_timing_example()
    result = @tau "macro_timer" begin
        sleep(0.3)
        sum(rand(500, 500))
    end
    return result
end

function nested_macro_timing_example()
    result = @tau "outer_timer" begin
        sleep(0.2)

        result1 = @tau "inner_computation_1" begin
            sleep(0.1)
            sum(rand(100,100))
        end

        result2 = @tau "inner_computation_2" begin
            sleep(0.1)
            sum(rand(200,200))
        end

        result1 + result2
    end
    return result
end

function fibonacci(n)
    @tau "fibonnaci" begin
        if n <= 1
            return n
        end
        return fibonacci(n-1) + fibonacci(n-2)
    end
end

function recursive_example()
    result=fibonacci(10)
    return result
end

@tau_func function matrix_multiply(A, B)
    return A * B
end

@tau_func function vector_operations(n)
    x = rand(n)
    y = rand(n)
    return sum(x .* y)
end

@tau_func quick_sum(arr) = sum(arr)

@tau_func function tau_func_example()
    A = rand(100, 100)
    B = rand(100, 100)
    C = matrix_multiply(A, B)
    result = vector_operations(1000)
    arr = collect(1:100)
    s = quick_sum(arr)
end

macro_timing_example()
nested_macro_timing_example()
recursive_example()
tau_func_example()

