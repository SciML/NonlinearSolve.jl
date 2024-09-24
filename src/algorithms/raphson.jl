"""
    NewtonRaphson(; concrete_jac = nothing, linsolve = nothing, linesearch = NoLineSearch(),
        autodiff = nothing)

An advanced NewtonRaphson implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear systems.
"""
function NewtonRaphson(; concrete_jac = nothing, linsolve = nothing,
        linesearch = NoLineSearch(), autodiff = nothing)
    descent = NewtonDescent(; linsolve)
    return GeneralizedFirstOrderAlgorithm(;
        concrete_jac, name = :NewtonRaphson, linesearch, descent, jacobian_ad = autodiff)
end
