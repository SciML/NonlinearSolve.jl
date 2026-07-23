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
    common_kwargs_nocj = (; linsolve, autodiff, vjp_autodiff, jvp_autodiff)
    if must_use_jacobian isa Val{true}
        if T <: Complex
            algs = (NewtonRaphson(; common_kwargs...),)
        else
            algs = (
                NewtonRaphson(; common_kwargs...),
                TrustRegion(; common_kwargs...),
                TrustRegion(; common_kwargs..., radius_update_scheme = RUS.Fan),
                LevenbergMarquardt(; common_kwargs_nocj...),
            )
        end
    else
        # SimpleNewtonRaphson and SimpleTrustRegion are not robust to singular Jacobians
        # and thus are not included in the polyalgorithm
        if prefer_simplenonlinearsolve isa Val{true}
            if T <: Complex
                algs = (
                    SimpleBroyden(),
                    SimpleKlement(),
                    NewtonRaphson(; common_kwargs...),
                )
            else
                start_index = u0_len !== nothing ? (u0_len ≤ 25 ? 3 : 1) : 1
                algs = (
                    SimpleBroyden(),
                    SimpleKlement(),
                    NewtonRaphson(; common_kwargs...),
                    TrustRegion(; common_kwargs...),
                    TrustRegion(; common_kwargs..., radius_update_scheme = RUS.Fan),
                    LevenbergMarquardt(; common_kwargs_nocj...),
                )
            end
        else
            if T <: Complex
                algs = (
                    Broyden(; autodiff),
                    Klement(; linsolve, autodiff),
                    NewtonRaphson(; common_kwargs...),
                )
            else
                # TODO: This number requires a bit rigorous testing
                start_index = u0_len !== nothing ? (u0_len ≤ 25 ? 3 : 1) : 1
                algs = (
                    Broyden(; autodiff),
                    Klement(; linsolve, autodiff),
                    NewtonRaphson(; common_kwargs...),
                    TrustRegion(; common_kwargs...),
                    TrustRegion(; common_kwargs..., radius_update_scheme = RUS.Fan),
                    LevenbergMarquardt(; common_kwargs_nocj...),
                )
            end
        end
    end
    return NonlinearSolvePolyAlgorithm(algs; start_index)
end

"""
    FastShortcutHomotopyPolyalg(
        ::Type{T} = Float64;
        autodiff = nothing, concrete_jac = nothing, linsolve = nothing,
        vjp_autodiff = nothing, jvp_autodiff = nothing,
        warm_handoff::Bool = true, store_original::Val = Val(false)
    ) where {T}

The recommended default [`HomotopyPolyAlgorithm`](@ref) for solving a
[`SciMLBase.HomotopyProblem`](@ref) — e.g. a Modelica `homotopy(actual, simplified)`
initialization system — by continuation. It is the homotopy analogue of
[`FastShortcutNonlinearPolyalg`](@ref): a fast [`HomotopySweep`](@ref) (natural-parameter
continuation) escalating to a robust [`ArcLengthContinuation`](@ref) (pseudo-arclength) on
failure, with a [`FastShortcutNonlinearPolyalg`](@ref) built from the requested `autodiff`
threaded in as the *inner corrector* of both stages.

Solving a `HomotopyProblem` with a plain nonlinear algorithm instead fixes ``λ`` at the
target and solves only the `actual` system, which can converge to the wrong branch. This
sweeps ``λ`` from the `simplified` anchor to the `actual` system, tracking the intended
branch.

### Arguments

  - `T`: the eltype of the initial guess, forwarded to the inner
    [`FastShortcutNonlinearPolyalg`](@ref). Defaults to `Float64`.

### Keyword Arguments

  - `autodiff`, `concrete_jac`, `linsolve`, `vjp_autodiff`, `jvp_autodiff`: forwarded to the
    inner [`FastShortcutNonlinearPolyalg`](@ref) that corrects each continuation step — this
    is where the differentiation backend is chosen. A `HomotopyProblem` whose residual is not
    ForwardDiff-safe is solved by passing `autodiff = AutoFiniteDiff()`.
  - `warm_handoff`, `store_original`: forwarded to [`HomotopyPolyAlgorithm`](@ref).
"""
function FastShortcutHomotopyPolyalg(
        ::Type{T} = Float64;
        autodiff = nothing, concrete_jac = nothing, linsolve = nothing,
        vjp_autodiff = nothing, jvp_autodiff = nothing,
        warm_handoff::Bool = true, store_original::Val = Val(false)
    ) where {T}
    inner = FastShortcutNonlinearPolyalg(
        T; autodiff, concrete_jac, linsolve, vjp_autodiff, jvp_autodiff
    )
    return HomotopyPolyAlgorithm(; inner, warm_handoff, store_original)
end
