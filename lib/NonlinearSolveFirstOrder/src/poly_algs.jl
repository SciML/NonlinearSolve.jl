"""
    RobustMultiNewton(
        ::Type{T} = Float64;
        concrete_jac = nothing,
        linsolve = nothing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing
    )

A polyalgorithm focused on robustness. It uses a mixture of Newton methods with different
globalizing techniques (trust region updates, line searches, etc.) in order to find a
method that is able to adequately solve the minimization problem.

Basically, if this algorithm fails, then "most" good ways of solving your problem fail and
you may need to think about reformulating the model (either there is an issue with the model,
or more precision / more stable linear solver choice is required).

### Arguments

  - `T`: The eltype of the initial guess. It is only used to check if some of the algorithms
    are compatible with the problem type. Defaults to `Float64`.
"""
function RobustMultiNewton(
        ::Type{T} = Float64;
        concrete_jac = nothing,
        linsolve = nothing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing
    ) where {T}
    common_kwargs = (; concrete_jac, linsolve, autodiff, vjp_autodiff, jvp_autodiff)
    if T <: Complex # Let's atleast have something here for complex numbers
        algs = (
            NewtonRaphson(; common_kwargs...),
        )
    else
        algs = (
            TrustRegion(; common_kwargs...),
            TrustRegion(; common_kwargs..., radius_update_scheme = RUS.Bastin),
            NewtonRaphson(; common_kwargs...),
            NewtonRaphson(; common_kwargs..., linesearch = BackTracking()),
            TrustRegion(; common_kwargs..., radius_update_scheme = RUS.NLsolve),
            TrustRegion(; common_kwargs..., radius_update_scheme = RUS.Fan),
        )
    end
    return NonlinearSolvePolyAlgorithm(algs)
end

"""
    FastShortcutNLLSPolyalg(
        ::Type{T} = Float64;
        concrete_jac = nothing,
        linsolve = nothing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing
    )

A polyalgorithm focused on balancing speed and robustness. It first tries less robust methods
for more performance and then tries more robust techniques if the faster ones fail.

### Arguments

  - `T`: The eltype of the initial guess. It is only used to check if some of the algorithms
    are compatible with the problem type. Defaults to `Float64`.
"""
function FastShortcutNLLSPolyalg(
        ::Type{T} = Float64;
        concrete_jac = nothing,
        linsolve = nothing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing
    ) where {T}
    common_kwargs = (; linsolve, autodiff, vjp_autodiff, jvp_autodiff)
    if T <: Complex
        algs = (
            GaussNewton(; common_kwargs..., concrete_jac),
            LevenbergMarquardt(; common_kwargs..., disable_geodesic = Val(true)),
            LevenbergMarquardt(; common_kwargs...),
        )
    else
        algs = (
            GaussNewton(; common_kwargs..., concrete_jac),
            LevenbergMarquardt(; common_kwargs..., disable_geodesic = Val(true)),
            TrustRegion(; common_kwargs..., concrete_jac),
            GaussNewton(; common_kwargs..., linesearch = BackTracking(), concrete_jac),
            TrustRegion(;
                common_kwargs..., radius_update_scheme = RUS.Bastin, concrete_jac
            ),
            LevenbergMarquardt(; common_kwargs...),
        )
    end
    return NonlinearSolvePolyAlgorithm(algs)
end
