"""
    GaussNewton(; concrete_jac = nothing, linsolve = nothing, linesearch = nothing,
        precs = DEFAULT_PRECS, adkwargs...)

An advanced GaussNewton implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear least squares problems.
"""
function GaussNewton(; concrete_jac = nothing, linsolve = nothing, precs = DEFAULT_PRECS,
        linesearch = NoLineSearch(), vjp_autodiff = nothing, autodiff = nothing)
    descent = NewtonDescent(; linsolve, precs)
    return GeneralizedFirstOrderRootFindingAlgorithm(; concrete_jac, name = :GaussNewton,
        descent, jacobian_ad = autodiff, reverse_ad = vjp_autodiff)
end
