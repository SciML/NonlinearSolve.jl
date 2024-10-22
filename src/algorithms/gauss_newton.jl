"""
    GaussNewton(; concrete_jac = nothing, linsolve = nothing, precs = DEFAULT_PRECS,
        linesearch = nothing, vjp_autodiff = nothing, autodiff = nothing,
        jvp_autodiff = nothing)

An advanced GaussNewton implementation with support for efficient handling of sparse
matrices via colored automatic differentiation and preconditioned linear solvers. Designed
for large-scale and numerically-difficult nonlinear least squares problems.
"""
function GaussNewton(; concrete_jac = nothing, linsolve = nothing, precs = DEFAULT_PRECS,
        linesearch = nothing, vjp_autodiff = nothing, autodiff = nothing,
        jvp_autodiff = nothing)
    return GeneralizedFirstOrderAlgorithm{concrete_jac, :GaussNewton}(; linesearch,
        descent = NewtonDescent(; linsolve, precs), autodiff, vjp_autodiff, jvp_autodiff)
end
