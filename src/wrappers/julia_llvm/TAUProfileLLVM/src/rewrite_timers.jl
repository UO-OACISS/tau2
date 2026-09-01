# ============================================================================
# rewrite_timers.jl — TAU_UTILITY timers around the Phase 1 / Phase 2 rewriter
#
# Part of the TAUProfile module (LLVM backend).
# ============================================================================

# Labels for the rewrite timers.
const _PHASE1_TIMER_NAME = ".TAU Julia Rewrite"
const _PHASE2_TIMER_NAME = ".TAU Julia Phase 2 Rewrite"

# Profile group for the rewrite timers.
const _REWRITE_TIMER_GROUP = "TAU_UTILITY"

# Cached handles, created on first use so that libTAU is loaded by then.
const _PHASE1_TIMER = Ref{TauTimer}(TauTimer(C_NULL))
const _PHASE2_TIMER = Ref{TauTimer}(TauTimer(C_NULL))

# Get-or-create the TAU_UTILITY timer for `name` in `cache`.
function _rewrite_timer(cache::Ref{TauTimer}, name::String)
    if cache[].ptr == C_NULL
        cache[] = TauTimer(name; group=_REWRITE_TIMER_GROUP)
    end
    return cache[]
end

# --- Phase 1 rewrite timer ---
_tau_start_rewrite() = tau_start(_rewrite_timer(_PHASE1_TIMER, _PHASE1_TIMER_NAME))
_tau_stop_rewrite()  = tau_stop(_PHASE1_TIMER[])

# --- Phase 2 rewrite timer ---

# Start the Phase 2 timer per _phase2_timing_mode; returns the handle to stop.
function _tau_start_phase2()
    mode = _phase2_timing_mode[]
    mode === :off && return TauTimer(C_NULL)
    t = mode === :combined ?
        _rewrite_timer(_PHASE1_TIMER, _PHASE1_TIMER_NAME) :
        _rewrite_timer(_PHASE2_TIMER, _PHASE2_TIMER_NAME)
    tau_start(t)
    return t
end

_tau_stop_phase2(t::TauTimer) = tau_stop(t)
