"""
    FastShortcutNonlinearPolyalg(
        ::Type{T} = Float64;
        concrete_jac = nothing,
        linsolve = nothing,
        must_use_jacobian::Val = Val(false),
        prefer_simplenonlinearsolve::Val = Val(false),
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing,
        u0_len::Union{Int, Nothing} = nothing
    ) where {T}

A polyalgorithm focused on balancing speed and robustness. It first tries less robust methods
for more performance and then tries more robust techniques if the faster ones fail.

### Arguments

  - `T`: The eltype of the initial guess. It is only used to check if some of the algorithms
    are compatible with the problem type. Defaults to `Float64`.

### Keyword Arguments

  - `u0_len`: The length of the initial guess. If this is `nothing`, then the length of the
    initial guess is not checked. If this is an integer and it is less than `25`, we use
    jacobian based methods.
"""
function FastShortcutNonlinearPolyalg(
        ::Type{T} = Float64;
        concrete_jac = nothing,
        linsolve = nothing,
        must_use_jacobian::Val = Val(false),
        prefer_simplenonlinearsolve::Val = Val(false),
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing,
        u0_len::Union{Int, Nothing} = nothing
) where {T}
    start_index = 1
    common_kwargs = (; concrete_jac, linsolve, autodiff, vjp_autodiff, jvp_autodiff)
    if must_use_jacobian isa Val{true}
        if T <: Complex
            algs = (NewtonRaphson(; common_kwargs...),)
        else
            algs = (
                NewtonRaphson(; common_kwargs...),
                NewtonRaphson(; common_kwargs..., linesearch = BackTracking()),
                TrustRegion(; common_kwargs...),
                TrustRegion(; common_kwargs..., radius_update_scheme = RUS.Bastin)
            )
        end
    else
        # SimpleNewtonRaphson and SimpleTrustRegion are not robust to singular Jacobians
        # and thus are not included in the polyalgorithm
        if prefer_simplenonlinearsolve isa Val{true}
            if T <: Complex
                algs = (
                    SimpleBroyden(),
                    Broyden(; init_jacobian = Val(:true_jacobian), autodiff),
                    SimpleKlement(),
                    NewtonRaphson(; common_kwargs...)
                )
            else
                start_index = u0_len !== nothing ? (u0_len ≤ 25 ? 4 : 1) : 1
                algs = (
                    SimpleBroyden(),
                    Broyden(; init_jacobian = Val(:true_jacobian), autodiff),
                    SimpleKlement(),
                    NewtonRaphson(; common_kwargs...),
                    NewtonRaphson(; common_kwargs..., linesearch = BackTracking()),
                    TrustRegion(; common_kwargs...),
                    TrustRegion(; common_kwargs..., radius_update_scheme = RUS.Bastin)
                )
            end
        else
            if T <: Complex
                algs = (
                    Broyden(; autodiff),
                    Broyden(; init_jacobian = Val(:true_jacobian), autodiff),
                    Klement(; linsolve, autodiff),
                    NewtonRaphson(; common_kwargs...)
                )
            else
                # TODO: This number requires a bit rigorous testing
                start_index = u0_len !== nothing ? (u0_len ≤ 25 ? 4 : 1) : 1
                algs = (
                    Broyden(; autodiff),
                    Broyden(; init_jacobian = Val(:true_jacobian), autodiff),
                    Klement(; linsolve, autodiff),
                    NewtonRaphson(; common_kwargs...),
                    NewtonRaphson(; common_kwargs..., linesearch = BackTracking()),
                    TrustRegion(; common_kwargs...),
                    TrustRegion(; common_kwargs..., radius_update_scheme = RUS.Bastin)
                )
            end
        end
    end
    return NonlinearSolvePolyAlgorithm(algs; start_index)
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
            LevenbergMarquardt(; common_kwargs...)
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
            LevenbergMarquardt(; common_kwargs...)
        )
    end
    return NonlinearSolvePolyAlgorithm(algs)
end
