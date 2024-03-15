"""
    Halley(; concrete_jac = nothing, linsolve = nothing, linesearch = NoLineSearch(),
        precs = DEFAULT_PRECS, autodiff = nothing)

An experimental Halley's method implementation.
"""
function Halley(; concrete_jac = nothing, linsolve = nothing,
        linesearch = NoLineSearch(), precs = DEFAULT_PRECS, autodiff = nothing)
    descent = HalleyDescent(; linsolve, precs)
    return GeneralizedFirstOrderAlgorithm(;
        concrete_jac, name = :Halley, linesearch, descent, jacobian_ad = autodiff)
end
