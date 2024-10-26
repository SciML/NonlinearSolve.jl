"""
    enable_timer_outputs()

Enable `TimerOutput` for all `NonlinearSolve` algorithms. This is useful for debugging
but has some overhead, so it is disabled by default.
"""
function enable_timer_outputs()
    set_preferences!(NonlinearSolveBase, "enable_timer_outputs" => true; force = true)
    @info "Timer Outputs Enabled. Restart the Julia session for this to take effect."
end

"""
    disable_timer_outputs()

Disable `TimerOutput` for all `NonlinearSolve` algorithms. This should be used when
`NonlinearSolve` is being used in performance-critical code.
"""
function disable_timer_outputs()
    set_preferences!(NonlinearSolveBase, "enable_timer_outputs" => false; force = true)
    @info "Timer Outputs Disabled. Restart the Julia session for this to take effect."
end
