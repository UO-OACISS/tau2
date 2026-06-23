module ExternalSimPkg_ec

module Inner
    export inner_step
    @noinline function inner_step(x::Float64, y::Float64)
        return x * 0.9 + y * 0.1
    end
end
import .Inner

@noinline function sim_main_ec(args::Vector{String})
    x = 1.0
    y = 2.0
    for _ in 1:3
        x = Inner.inner_step(x, y)
    end
    return x
end

end # module ExternalSimPkg_ec
