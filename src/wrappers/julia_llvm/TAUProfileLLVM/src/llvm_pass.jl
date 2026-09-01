# ============================================================================
# llvm_pass.jl — LLVM IR instrumentation pass -- hook insertion and exception wrapping
#
# Part of the TAUProfile module (LLVM backend).
# ============================================================================

"""
    _within_depth_limit(mi, depth, mod_depth) -> Bool

Check whether an MI at the given depth should be instrumented.
"""
function _within_depth_limit(mi::Core.MethodInstance, depth::Int, mod_depth::Int)
    global_max = _max_depth[]
    global_max == typemax(Int) && isempty(_module_depth_limits) && return true

    depth >= global_max && return false

    method = mi.def
    isa(method, Method) || return true
    mod = method.module

    if !isempty(_module_depth_limits)
        mod_limit = _module_limit_for(mod)
        if mod_limit !== nothing
            return mod_depth < mod_limit
        end
    end

    return true
end

# ============================================================================
# LLVM instrumentation pass
# ============================================================================

"""
    _should_instrument(fn::LLVM.Function) -> Bool

Determine whether an LLVM function should be instrumented.
"""
function _should_instrument(fn::LLVM.Function)
    LLVM.isdeclaration(fn) && return false
    LLVM.isintrinsic(fn) && return false
    fname = LLVM.name(fn)
    (startswith(fname, "julia_") || startswith(fname, "j_")) || return false

    # Name, prefix, module exclusion and whitelist via the MethodInstance if mapped
    mi = get(_llvm_func_mi_map, fname, nothing)
    method = mi === nothing ? nothing : mi.def
    mod = isa(method, Method) ? method.module : nothing
    _passes_config_filter(_get_function_name(fn), mod) || return false

    if mi === nothing
        # Fall back to module name string from the map
        mod_name = get(_llvm_func_module_map, fname, nothing)
        if mod_name !== nothing
            mod_sym = Symbol(mod_name)
            haskey(_excluded_modules, mod_sym) && return false
            if !isempty(_whitelisted_modules)
                haskey(_whitelisted_modules, mod_sym) || return false
            end
        elseif !isempty(_whitelisted_modules)
            # No module info available
            return false
        end
    end

    # Depth limit check
    if haskey(_llvm_func_depth_map, fname)
        depth, mod_depth = _llvm_func_depth_map[fname]
        if mi !== nothing
            _within_depth_limit(mi, depth, mod_depth) || return false
        else
            # No MI available
            depth >= _max_depth[] && return false
        end
    end

    return true
end

# ============================================================================
# Exception handler helpers
# ============================================================================

"""
    _get_or_declare_fn!(mod, name, ft) -> LLVM.Function

Get an existing function declaration in the module, or create one.
"""
function _get_or_declare_fn!(mod::LLVM.Module, name::String, ft::LLVM.FunctionType)
    ref = LLVM.API.LLVMGetNamedFunction(mod, name)
    if ref != C_NULL
        return LLVM.Function(ref)
    else
        return LLVM.Function(mod, name, ft)
    end
end

"""
    _find_pgcstack(entry_bb) -> Union{LLVM.Value, Nothing}

Find the `tls_pgcstack` SSA value in the entry block.
"""
function _find_pgcstack(entry_bb::LLVM.BasicBlock)
    for inst in LLVM.instructions(entry_bb)
        iname = LLVM.name(inst)
        if iname == "tls_pgcstack" || iname == "pgcstack"
            return inst
        end
    end
    # Fallback: look for inline asm "movq %fs:0
    for inst in LLVM.instructions(entry_bb)
        if inst isa LLVM.CallInst
            s = string(inst)
            if contains(s, "movq %fs:0") || contains(s, r"%fs:")
                # Next instructions should be GEP then load
                found_gep = nothing
                past_asm = false
                for inst2 in LLVM.instructions(entry_bb)
                    if inst2 === inst
                        past_asm = true
                        continue
                    end
                    past_asm || continue
                    s2 = string(inst2)
                    if found_gep === nothing && contains(s2, "getelementptr") && contains(s2, "-8")
                        found_gep = inst2
                    elseif found_gep !== nothing && inst2 isa LLVM.LoadInst
                        return inst2
                    elseif found_gep !== nothing
                        break
                    end
                end
            end
        end
    end
    return nothing
end

"""
    _find_setup_end(entry_bb) -> Union{LLVM.Instruction, Nothing}

Find the last instruction of the pgcstack/safepoint setup in the entry block.
The setup consists of TLS access, GEP/load for pgcstack, and the safepoint
polling sequence (fence + volatile load + fence). Returns the instruction
AFTER which the entry hook should be inserted.
"""
function _find_setup_end(entry_bb::LLVM.BasicBlock)
    last_fence = nothing
    pgcstack_inst = nothing
    for inst in LLVM.instructions(entry_bb)
        iname = LLVM.name(inst)
        if iname == "tls_pgcstack" || iname == "pgcstack"
            pgcstack_inst = inst
        end
        if inst isa LLVM.FenceInst
            last_fence = inst
        end
    end
    # Return last fence if found, otherwise pgcstack load
    return last_fence !== nothing ? last_fence : pgcstack_inst
end

"""
    _fix_phi_incoming_blocks!(fn, entry_bb, try_body_bb)

After splitting the entry block, update PHI nodes in ALL blocks of the function
that reference entry_bb to reference try_body_bb instead.
"""
function _fix_phi_incoming_blocks!(fn::LLVM.Function, entry_bb::LLVM.BasicBlock, try_body_bb::LLVM.BasicBlock)
    entry_bb_ref = Base.unsafe_convert(LLVM.API.LLVMBasicBlockRef, entry_bb)

    for bb in LLVM.blocks(fn)
        bb === entry_bb && continue
        bb === try_body_bb && continue

        # Collect PHI nodes that need fixing
        phis_to_fix = LLVM.PHIInst[]
        for inst in LLVM.instructions(bb)
            inst isa LLVM.PHIInst || break
            inc = LLVM.incoming(inst)
            for i in 1:length(inc)
                _, block = inc[i]
                if Base.unsafe_convert(LLVM.API.LLVMBasicBlockRef, block) == entry_bb_ref
                    push!(phis_to_fix, inst)
                    break
                end
            end
        end

        isempty(phis_to_fix) && continue

        # Rebuild each affected PHI
        for old_phi in phis_to_fix
            phi_type = LLVM.value_type(old_phi)
            phi_name = LLVM.name(old_phi)

            # Collect corrected incoming edges
            inc = LLVM.incoming(old_phi)
            n = length(inc)
            edges = Tuple{LLVM.Value, LLVM.BasicBlock}[]
            for i in 1:n
                val, block = inc[i]
                if Base.unsafe_convert(LLVM.API.LLVMBasicBlockRef, block) == entry_bb_ref
                    push!(edges, (val, try_body_bb))
                else
                    push!(edges, (val, block))
                end
            end

            # Create new PHI before the old one
            @dispose builder=LLVM.IRBuilder() begin
                LLVM.position!(builder, old_phi)
                new_phi = LLVM.phi!(builder, phi_type, phi_name)
                append!(LLVM.incoming(new_phi), edges)
                LLVM.replace_uses!(old_phi, new_phi)
                LLVM.erase!(old_phi)
            end
        end
    end
end

"""
    _wrap_with_exception_handler!(fn, mod, entry_bb, entry_hook_call, name_gv, exit_ptr_val, hook_ft)

Wrap the function body in a julia.except_enter / jl_pop_handler try/catch
so that the exit hook fires on all exit paths including uncaught exceptions.

Transforms the entry block by:
1. Keeping pgcstack/safepoint setup + entry hook call in entry_bb
2. Moving remaining instructions to a new try_body block
3. Adding exception handler setup (julia.except_enter + ct->eh store)
4. Creating a catch block that fires exit hook then rethrows
5. Inserting ijl_pop_handler_noexcept before each ret's exit hook
"""
function _wrap_with_exception_handler!(fn::LLVM.Function, mod::LLVM.Module,
        entry_bb::LLVM.BasicBlock, entry_hook_call::LLVM.Instruction,
        name_gv::LLVM.Value, exit_ptr_val::LLVM.Value, hook_ft::LLVM.FunctionType)

    # Find pgcstack
    pgcstack = _find_pgcstack(entry_bb)
    if pgcstack === nothing
        @warn "TAUProfile: Could not find pgcstack in entry block, skipping exception wrapping"
        return false
    end

    ptr_type = LLVM.PointerType()
    i8_type = LLVM.IntType(8)
    i32_type = LLVM.IntType(32)

    # Collect instructions to move (everything after entry_hook_call)
    to_move = LLVM.Instruction[]
    past_hook = false
    for inst in LLVM.instructions(entry_bb)
        if inst === entry_hook_call
            past_hook = true
            continue
        end
        if past_hook
            push!(to_move, inst)
        end
    end

    # Create new basic blocks
    try_body_bb = LLVM.BasicBlock(fn, "try_body")
    catch_bb = LLVM.BasicBlock(fn, "catch")
    # Position try_body right after entry
    LLVM.move_after(try_body_bb, entry_bb)

    # Move instructions from entry_bb to try_body_bb
    for inst in to_move
        LLVM.remove!(inst)
    end
    @dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, try_body_bb)
        for inst in to_move
            Base.insert!(builder, inst)
        end
    end

    # Fix PHI nodes in all blocks that referenced entry_bb
    _fix_phi_incoming_blocks!(fn, entry_bb, try_body_bb)

    # --- Declare runtime functions ---
    except_enter_ret_type = LLVM.StructType([i32_type, ptr_type])
    except_enter_ft = LLVM.FunctionType(except_enter_ret_type, [ptr_type])
    except_enter_fn = _get_or_declare_fn!(mod, "julia.except_enter", except_enter_ft)

    pop_handler_noexcept_ft = LLVM.FunctionType(LLVM.VoidType(), [ptr_type, i32_type])
    pop_handler_noexcept_fn = _get_or_declare_fn!(mod, "ijl_pop_handler_noexcept", pop_handler_noexcept_ft)

    pop_handler_ft = LLVM.FunctionType(LLVM.VoidType(), [ptr_type, i32_type])
    pop_handler_fn = _get_or_declare_fn!(mod, "ijl_pop_handler", pop_handler_ft)

    rethrow_ft = LLVM.FunctionType(LLVM.VoidType(), LLVM.LLVMType[])
    rethrow_fn = _get_or_declare_fn!(mod, "ijl_rethrow", rethrow_ft)

    # Build exception handler setup in entry_bb
    local current_task
    @dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, entry_bb)

        # current_task = pgcstack + task offset
        current_task = LLVM.gep!(builder, i8_type, pgcstack,
            [LLVM.ConstantInt(LLVM.IntType(64), _TASK_OFFSET_FROM_PGCSTACK[])], "current_task")

        # except = call {i32, ptr} @julia.except_enter(ptr %current_task)
        except_call = LLVM.call!(builder, except_enter_ft, except_enter_fn, [current_task], "except")
        push!(LLVM.function_attributes(except_call), LLVM.EnumAttribute("returns_twice", 0))

        # setjmp_result = extractvalue {i32, ptr} %except, 0
        setjmp_result = LLVM.extract_value!(builder, except_call, 0, "setjmp_result")

        # handler_buf = extractvalue {i32, ptr} %except, 1
        handler_buf = LLVM.extract_value!(builder, except_call, 1, "handler_buf")

        # eh_field = gep i8, pgcstack, 32  (ct->eh is at pgcstack + 32)
        eh_field = LLVM.gep!(builder, i8_type, pgcstack,
            [LLVM.ConstantInt(LLVM.IntType(64), 32)], "eh_field")

        # store ptr %handler_buf, ptr %eh_field
        LLVM.store!(builder, handler_buf, eh_field)

        # is_normal = icmp eq i32 %setjmp_result, 0
        is_normal = LLVM.icmp!(builder, LLVM.API.LLVMIntEQ, setjmp_result,
            LLVM.ConstantInt(i32_type, 0), "is_normal")

        # br i1 %is_normal, label %try_body, label %catch
        LLVM.br!(builder, is_normal, try_body_bb, catch_bb)
    end

    # Insert ijl_pop_handler_noexcept + exit_hook before each ret
    ret_insts = LLVM.Instruction[]
    for bb in LLVM.blocks(fn)
        bb === catch_bb && continue  # skip catch block
        term = LLVM.terminator(bb)
        if term !== nothing && term isa LLVM.RetInst
            push!(ret_insts, term)
        end
    end
    for ret_inst in ret_insts
        @dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, ret_inst)
            LLVM.call!(builder, pop_handler_noexcept_ft, pop_handler_noexcept_fn,
                [current_task, LLVM.ConstantInt(i32_type, 1)])
            LLVM.call!(builder, hook_ft, exit_ptr_val, [name_gv])
        end
    end

    # Build catch block
    @dispose builder=LLVM.IRBuilder() begin
        LLVM.position!(builder, catch_bb)
        LLVM.call!(builder, pop_handler_ft, pop_handler_fn,
            [current_task, LLVM.ConstantInt(i32_type, 1)])
        LLVM.call!(builder, hook_ft, exit_ptr_val, [name_gv])
        LLVM.call!(builder, rethrow_ft, rethrow_fn, LLVM.Value[])
        LLVM.unreachable!(builder)
    end

    return true
end

# ============================================================================
# Main instrumentation function
# ============================================================================

# Pin the current task to its thread (task.sticky = true) before the entry hook,
# so task migration can't corrupt TAU's per-thread timer stack while a timer is
# open.
function _emit_sticky_pin!(builder, pgcstack)
    i8 = LLVM.IntType(8)
    addr = LLVM.gep!(builder, i8, pgcstack,
        [LLVM.ConstantInt(LLVM.IntType(64), _STICKY_BYTE_OFFSET_FROM_PGCSTACK[])], "sticky_addr")
    LLVM.store!(builder, LLVM.ConstantInt(i8, 1), addr)
end

"""
    instrument_function!(fn::LLVM.Function, entry_hook_ptr::Ptr{Cvoid}, exit_hook_ptr::Ptr{Cvoid})

Insert entry/exit hook calls into an LLVM function with exception-safe exit hooks.
"""
function instrument_function!(fn::LLVM.Function, entry_hook_ptr::Ptr{Cvoid}, exit_hook_ptr::Ptr{Cvoid}, tau_mode::Bool=false)
    _should_instrument(fn) || return false

    # The sticky pin (TAU mode) and the exception handler's current-task gep both
    # depend on the pgcstack offsets computed in __init__.
    tau_mode && !_offsets_ready[] && error("TAUProfile: pgcstack/sticky offsets not " *
        "computed; __init__ did not run before instrumentation.")

    # Propagate Julia @noinline to LLVM noinline attribute so the optimizer
    # doesn't inline instrumented functions and eliminate their hooks
    fname = LLVM.name(fn)
    if fname in _llvm_func_noinline_set
        push!(LLVM.function_attributes(fn), LLVM.EnumAttribute("noinline", 0))
    end

    mod = LLVM.parent(fn)
    label = String(_build_function_label(fn))

    # Create a global string constant for the function label
    gv_name = String("trace_label_$(hash(label))")
    name_gv = LLVM.globalstring_ptr!(mod, label, gv_name)

    # Hook function type: void(ptr)
    ptr_type = LLVM.PointerType()
    hook_ft = LLVM.FunctionType(LLVM.VoidType(), [ptr_type])

    # Embed hook pointers as constants via inttoptr
    int_type = LLVM.IntType(sizeof(Ptr{Cvoid}) * 8)
    entry_ptr_int = LLVM.ConstantInt(int_type, reinterpret(UInt, entry_hook_ptr))
    entry_ptr_val = LLVM.const_inttoptr(entry_ptr_int, ptr_type)
    exit_ptr_int = LLVM.ConstantInt(int_type, reinterpret(UInt, exit_hook_ptr))
    exit_ptr_val = LLVM.const_inttoptr(exit_ptr_int, ptr_type)

    # Insert entry hook AFTER pgcstack/safepoint setup (not at function start)
    # The setup must remain in the entry block for Julia's LLVM passes
    entry_bb = first(LLVM.blocks(fn))
    setup_end = _find_setup_end(entry_bb)
    # pgcstack for the TAU-mode sticky pin (emitted right before the entry hook,
    # where pgcstack is already in scope after the safepoint setup).
    pgcstack = _find_pgcstack(entry_bb)
    local entry_hook_call
    if setup_end !== nothing
        # Find the instruction after setup_end to position before it
        found = false
        local insert_before
        for inst in LLVM.instructions(entry_bb)
            if found
                insert_before = inst
                @goto found_insert_point
            end
            if inst === setup_end
                found = true
            end
        end
        # setup_end was the last instruction — position at end of block
        @dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, entry_bb)
            tau_mode && pgcstack !== nothing && _emit_sticky_pin!(builder, pgcstack)
            entry_hook_call = LLVM.call!(builder, hook_ft, entry_ptr_val, [name_gv])
        end
        @goto done_entry_hook
        @label found_insert_point
        @dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, insert_before)
            tau_mode && pgcstack !== nothing && _emit_sticky_pin!(builder, pgcstack)
            entry_hook_call = LLVM.call!(builder, hook_ft, entry_ptr_val, [name_gv])
        end
        @label done_entry_hook
    else
        # Fallback: insert at function start
        first_inst = first(LLVM.instructions(entry_bb))
        @dispose builder=LLVM.IRBuilder() begin
            LLVM.position!(builder, first_inst)
            tau_mode && pgcstack !== nothing && _emit_sticky_pin!(builder, pgcstack)
            entry_hook_call = LLVM.call!(builder, hook_ft, entry_ptr_val, [name_gv])
        end
    end

    # Wrap function body in exception handler for exit hook safety
    _wrap_with_exception_handler!(fn, mod, entry_bb, entry_hook_call,
        name_gv, exit_ptr_val, hook_ft)

    return true
end

"""
    instrument_module!(mod::LLVM.Module)

Run the instrumentation pass on all eligible functions in an LLVM module.
"""
function instrument_module!(mod::LLVM.Module)
    entry_ptr, exit_ptr, tau_mode = _active_hook_ptrs()
    if entry_ptr == C_NULL || exit_ptr == C_NULL
        @warn "TAUProfile: Hook pointers not initialized"
        return 0
    end

    count = 0
    for fn in LLVM.functions(mod)
        try
            if instrument_function!(fn, entry_ptr, exit_ptr, tau_mode)
                count += 1
            end
        catch ex
            fname = LLVM.name(fn)
            @warn "TAUProfile: Failed to instrument $fname" exception=(ex, catch_backtrace())
        end
    end
    return count
end
