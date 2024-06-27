"""
    Halley(; concrete_jac = nothing, linsolve = nothing, linesearch = NoLineSearch(),
        precs = DEFAULT_PRECS, autodiff = nothing)

An experimental Halley's method implementation. Improves the convergence rate of Newton's method by using second-order derivative information to correct the descent direction.

Currently depends on TaylorDiff.jl to handle the correction terms,
might have more general implementation in the future.
"""
function Halley(; concrete_jac = nothing, linsolve = nothing,
        linesearch = NoLineSearch(), precs = DEFAULT_PRECS, autodiff = nothing)
    descent = HalleyDescent(; linsolve, precs)
    return GeneralizedFirstOrderAlgorithm(;
        concrete_jac, name = :Halley, linesearch, descent, jacobian_ad = autodiff)
end
