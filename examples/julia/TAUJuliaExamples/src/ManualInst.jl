using TAUProfile

function manual_timing_example()
    tau_start("manual_timing_example")
    println("Hello world!")
    sleep(0.5)
    result = sum(rand(1000, 1000))
    tau_stop("manual_timing_example")
    return result
end

manual_timing_example()

