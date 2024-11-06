"""
    Halley(; concrete_jac = nothing, linsolve = nothing, linesearch = missing,
        autodiff = nothing)

An experimental Halley's method implementation. Improves the convergence rate of Newton's method by using second-order derivative information to correct the descent direction.

Currently depends on TaylorDiff.jl to handle the correction terms,
might have more general implementation in the future.
"""
function Halley(; concrete_jac = nothing, linsolve = nothing,
        linesearch = missing, autodiff = nothing)
    return GeneralizedFirstOrderAlgorithm(;
        concrete_jac, name = :Halley, linesearch,
        descent = HalleyDescent(; linsolve), autodiff)
end
