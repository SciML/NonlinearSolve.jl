"""
    RobustMultiNewton(
        ::Type{T} = Float64;
        concrete_jac = nothing,
        linsolve = nothing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing,
        jacobian_reuse = nothing
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
  - `jacobian_reuse`: forwarded to each first-order method in the polyalgorithm.
"""
function RobustMultiNewton(
        ::Type{T} = Float64;
        concrete_jac = nothing,
        linsolve = nothing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing,
        jacobian_reuse = nothing
    ) where {T}
    common_kwargs = (;
        concrete_jac, linsolve, autodiff, vjp_autodiff, jvp_autodiff, jacobian_reuse,
    )
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
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing,
        jacobian_reuse = nothing
    )

A polyalgorithm focused on balancing speed and robustness. It first tries less robust methods
for more performance and then tries more robust techniques if the faster ones fail.

### Arguments

  - `T`: The eltype of the initial guess. It is only used to check if some of the algorithms
    are compatible with the problem type. Defaults to `Float64`.
  - `jacobian_reuse`: forwarded to each first-order method in the polyalgorithm.
"""
function FastShortcutNLLSPolyalg(
        ::Type{T} = Float64;
        concrete_jac = nothing,
        linsolve = nothing,
        autodiff = nothing, vjp_autodiff = nothing, jvp_autodiff = nothing,
        jacobian_reuse = nothing
    ) where {T}
    common_kwargs = (; linsolve, autodiff, vjp_autodiff, jvp_autodiff, jacobian_reuse)
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
                common_kwargs..., radius_update_scheme = RUS.Fan, concrete_jac
            ),
            LevenbergMarquardt(; common_kwargs...),
        )
    end
    return NonlinearSolvePolyAlgorithm(algs)
end

"""
    FastShortcutBoundedPolyalg(; concrete_jac = nothing, linsolve = nothing,
        autodiff = nothing, jvp_autodiff = nothing, vjp_autodiff = nothing, gtol = nothing, must_support_postcondition = false)

A polyalgorithm for real, box-constrained nonlinear systems and nonlinear least squares.
It tries [`BoundedTrustRegion`](@ref), followed by [`BoundedGaussNewton`](@ref) with
projected backtracking if the first method fails. Each stage starts from the supplied
initial guess and operates in the original bounded coordinates.

This is the default for `NonlinearProblem` and `NonlinearLeastSquaresProblem` with `lb`
or `ub`. Explicit use requires an initial guess inside the box; `solve(prob)` projects
an out-of-bounds initial guess onto the box before selecting the default. Least-squares problems may succeed
at a constrained stationary point with nonzero residual; nonlinear systems must satisfy
the residual tolerance. These are local methods and do not guarantee a global minimum.

`concrete_jac`, `linsolve`, and the differentiation backends are forwarded to both stages.
Sparse Jacobian prototypes and matrix-free Krylov solvers use the usual Jacobian and
linear-solver caches. `gtol` sets the projected-gradient tolerance; its default is
`sqrt(eps(T))` for least squares and zero for nonlinear systems.

Set `must_support_postcondition = true` to restrict the sequence to `BoundedTrustRegion`,
which supports iterate correctors. Default algorithm selection sets this when a
`postcondition` is supplied to the solve; pass the corrector itself to `solve` or `init`.

See the [bounded solver recommendations](@ref bounded-solvers) for benchmark results
and alternatives. The stage order may change as the benchmark coverage grows.
"""
function FastShortcutBoundedPolyalg(;
        concrete_jac = nothing, linsolve = nothing, autodiff = nothing,
        jvp_autodiff = nothing, vjp_autodiff = nothing, gtol = nothing,
        must_support_postcondition = false
    )
    kwargs = (; concrete_jac, linsolve, autodiff, jvp_autodiff, vjp_autodiff, gtol)
    first_stage = BoundedTrustRegion(; kwargs...)
    algs = must_support_postcondition ? (first_stage,) :
        (first_stage, BoundedGaussNewton(; kwargs...))
    return NonlinearSolvePolyAlgorithm(algs)
end
