module TAUProfile

export tau_start, tau_stop, @tau, @tau_func

using IRTools: @dynamo, IR, xcall, arguments, insertafter!, recurse!

# Global variable to store the library path
const libTAU = Ref{String}()

"""
    __init__()

Module initialization function. Reads the TAU_JULIA_LIB environment variable
to determine the path to libTAU.so, then initializes TAU and creates the
top-level timer.
"""
function __init__()
    tau_lib_path = get(ENV, "TAU_JULIA_LIB", "")

    if isempty(tau_lib_path)
        @warn "TAU_JULIA_LIB environment variable not set. TAU profiling will not be available."
        libTAU[] = ""
    elseif !isfile(tau_lib_path)
        @warn "TAU library not found at: $tau_lib_path. TAU profiling will not be available."
        libTAU[] = ""
    else
        libTAU[] = tau_lib_path
        #@info "TAU library loaded from: $tau_lib_path"

        # Initialize TAU
        init_result = ccall((:Tau_init_initializeTAU, libTAU[]), Cint, ())
        #if init_result != 0
        #    @info "TAU initialized successfully (return code: $init_result)"
        #end

        # Create top-level timer
        ccall((:Tau_create_top_level_timer_if_necessary, libTAU[]), Cvoid, ())
        #@info "TAU top-level timer created"
    end
end

"""
    tau_start(name::String)

Start a TAU timer with the given name.

# Arguments
- `name::String`: The name of the timer to start

# Example
```julia
tau_start("my_function")
```
"""
function tau_start(name::String)
    if isempty(libTAU[])
        return  # Silently skip if library not loaded
    end

    ccall((:Tau_start, libTAU[]), Cvoid, (Cstring,), name)
end

"""
    tau_stop(name::String)

Stop a TAU timer with the given name.

# Arguments
- `name::String`: The name of the timer to stop

# Example
```julia
tau_stop("my_function")
```
"""
function tau_stop(name::String)
    if isempty(libTAU[])
        return  # Silently skip if library not loaded
    end

    ccall((:Tau_stop, libTAU[]), Cvoid, (Cstring,), name)
end

"""
    @tau(name::String, expr)

Macro to automatically wrap an expression with TAU timing.

# Arguments
- `name::String`: The name for the TAU timer
- `expr`: The expression to time

# Example
```julia
@tau "computation" begin
    result = expensive_computation()
end
```
"""
macro tau(name, expr)
    quote
        tau_start($(esc(name)))
        try
            $(esc(expr))
        finally
            tau_stop($(esc(name)))
        end
    end
end

"""
    @tau_func function_definition

Macro to automatically wrap a function definition with TAU timing.
The timer name is automatically derived from the function name.

# Example
```julia
@tau_func function my_computation(x, y)
    return x * y + sqrt(x^2 + y^2)
end

# Equivalent to:
function my_computation(x, y)
    @tau "my_computation" begin
        return x * y + sqrt(x^2 + y^2)
    end
end
```

Also works with short-form function definitions:
```julia
@tau_func fast_add(x, y) = x + y
```
"""
macro tau_func(func_expr)
    if func_expr.head == :function || func_expr.head == :(=)
        # Extract function signature and body
        func_sig = func_expr.args[1]
        func_body = func_expr.args[2]

        # Get the function name
        # Handle: func_name(args...), Type.func_name(args...), or just func_name
        func_name = if func_sig isa Symbol
            string(func_sig)
        elseif func_sig.head == :call
            # Extract the function name from the call signature
            name_part = func_sig.args[1]
            if name_part isa Symbol
                string(name_part)
            elseif name_part.head == :(.)
                # Handle Type.method_name
                string(name_part.args[2].value)
            else
                "unknown"
            end
        elseif func_sig.head == :where
            # Handle parameterized functions: func_name(args...) where {T}
            inner_sig = func_sig.args[1]
            if inner_sig.head == :call
                string(inner_sig.args[1])
            else
                "unknown"
            end
        else
            "unknown"
        end

        # Wrap the body with @tau
        new_body = quote
            @tau $func_name begin
                $(func_body)
            end
        end

        # Return the modified function
        return esc(Expr(func_expr.head, func_sig, new_body))
    else
        error("@tau_func must be used with a function definition")
    end
end


function tau_entry_hook(f, args...)
    tau_start(string(f))
end

function tau_exit_hook(f, result)
    tau_stop(string(f))
    return result
end

function tau_exception_hook(f, exc)
    tau_stop(string(f))
    return exc
end

# Functions to exclude from instrumentation
const EXCLUDED_FUNCTIONS = Set([
    typeof(tau_start),
    typeof(tau_stop),
])



@dynamo function tau_rewrite_function(m...)
  # Skip instrumentation for excluded functions
  if length(m) > 0 && m[1] in EXCLUDED_FUNCTIONS
    return
  end

  ir = IR(m...)
  ir == nothing && return
  recurse!(ir) # Recurse into functions called by this function

  f = arguments(ir)[1]

  # Instrument entry to function
  pushfirst!(ir, xcall(Main, :tau_entry_hook, arguments(ir)...))

  # We need to add a block to handle leaving the try/finally block
  # before each return. To do this, we need to count the number
  # of returns so that we know how to number the added catch block.
  num_returns = 0
  for block in blocks(ir)
    for branch in branches(block)
      if isreturn(branch)
        num_returns += 1
      end
    end
  end

  # Calculate catch block ID
  # The catch block ID will be: the number of existing blocks,
  # plus the number of returns (for each of which we have to add a block),
  # plus one for the catch block itself
  catch_block_id = length(blocks(ir)) + num_returns + 1

  # Wrap the whole function in a catch/finally so that we can
  # stop the TAU timer upon exit, no matter how we exit.
  # This refers to the ID of the block which will be added
  # at the end of the function.
  token = pushfirst!(ir, Expr(:enter, catch_block_id))
  insertafter!(ir, token, Expr(:catch, catch_block_id))

  # Create new blocks for each return.
  # We have to leave the try/finally context before any return.
  for (block_idx, block) in enumerate(blocks(ir))
    brs = branches(block)
    for (br_idx, branch) in enumerate(brs)
      if isreturn(branch)
        retval = returnvalue(branch)

        new_block_id = length(blocks(ir)) + 1
        new_blk = block!(ir, new_block_id)

        push!(new_blk, Expr(:leave, token))
        logged_ret = push!(new_blk, xcall(Main, :tau_exit_hook, f, retval))
        return!(new_blk, logged_ret)

        branches(block)[br_idx] = Branch(nothing, new_block_id, [])
      end
    end
  end

  # Create catch block at end of function
  catch_blk = block!(ir, catch_block_id)
  exc = push!(catch_blk, Expr(:the_exception))
  push!(catch_blk, xcall(Main, :tau_exception_hook, f, exc))
  rethrow_result = push!(catch_blk, xcall(:rethrow))
  push!(catch_blk, Expr(:pop_exception, token))
  return!(catch_blk, rethrow_result)

  return ir
end

"""
    @tau(expr)

Convenience macro to automatically rewrite the IR of a function call with TAU timing.
This recurses into all functions called within the expression 
and inserts TAU instrumentation around each function.
Invokes tau_rewrite_function on the supplied function.

# Arguments
- `expr`: The expression to instrument.

# Example
```julia
function add(x, y)
    return x + y
end

result = @tau_rewrite add(3, 4)
```
"""
macro tau_rewrite(expr)
    if expr.head == :call
        func = expr.args[1]
        args = expr.args[2:end]
        return :(tau_rewrite_function($(esc(func)), $(map(esc, args)...)))
    else
        error("@tau_rewrite expects a function call expression, got: $expr")
    end
end

end # module TAUProfile
