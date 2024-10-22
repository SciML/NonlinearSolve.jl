"""
    NewtonRaphson(; concrete_jac = nothing, linsolve = nothing, linesearch = missing,
        precs = DEFAULT_PRECS, autodiff = nothing, vjp_autodiff = nothing,
        jvp_autodiff = nothing)

An advanced NewtonRaphson implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear systems.
"""
function NewtonRaphson(; concrete_jac = nothing, linsolve = nothing, linesearch = nothing,
        precs = DEFAULT_PRECS, autodiff = nothing, vjp_autodiff = nothing,
        jvp_autodiff = nothing)
    return GeneralizedFirstOrderAlgorithm{concrete_jac, :NewtonRaphson}(; linesearch,
        descent = NewtonDescent(; linsolve, precs), autodiff, vjp_autodiff, jvp_autodiff)
end
