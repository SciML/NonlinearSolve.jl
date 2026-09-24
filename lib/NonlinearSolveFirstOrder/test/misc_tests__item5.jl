using NonlinearSolveFirstOrder
using NonlinearSolveFirstOrder: RadiusUpdateSchemes

# The Bastin and Yuan radius update schemes construct VJP operators, whose default
# backend is selected from whichever reverse-mode AD packages happen to be loaded.
# The default NLLS polyalgorithm must not contain such stages, so that loading a
# reverse-mode AD package (e.g. Enzyme) can never change the behavior of a default
# solve. See NonlinearSolve.jl#837. `RobustMultiNewton` intentionally keeps its
# Bastin stage as an explicit opt-in robustness battery.
function stage_needs_reverse_mode(stage)
    hasproperty(stage, :trustregion) || return false
    tr = stage.trustregion
    (tr === missing || tr === nothing) && return false
    hasproperty(tr, :method) || return false
    return tr.method isa
        Union{RadiusUpdateSchemes.__Bastin, RadiusUpdateSchemes.__Yuan}
end

@test all(!stage_needs_reverse_mode, FastShortcutNLLSPolyalg().algs)
