"""
    NewtonRaphson(; concrete_jac = nothing, linsolve = nothing, linesearch = NoLineSearch(),
        precs = DEFAULT_PRECS, autodiff = nothing)

An advanced NewtonRaphson implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear systems.
"""
function NewtonRaphson(; concrete_jac = nothing, linsolve = nothing,
        linesearch = NoLineSearch(), precs = DEFAULT_PRECS, autodiff = nothing)
    descent = NewtonDescent(; linsolve, precs)
    return GeneralizedFirstOrderAlgorithm(; concrete_jac, name = :NewtonRaphson,
        linesearch, descent, jacobian_ad = autodiff)
end
