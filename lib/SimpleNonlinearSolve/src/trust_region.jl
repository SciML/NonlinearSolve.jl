"""
    SimpleTrustRegion(;
        autodiff = AutoForwardDiff(), max_trust_radius = 0.0,
        initial_trust_radius = 0.0, step_threshold = nothing,
        shrink_threshold = nothing, expand_threshold = nothing,
        shrink_factor = 0.25, expand_factor = 2.0, max_shrink_times::Int = 32,
        nlsolve_update_rule = Val(false)
    )

A low-overhead implementation of a trust-region solver. This method is non-allocating on
scalar and static array problems.

### Keyword Arguments

  - `autodiff`: determines the backend used for the Jacobian. Defaults to `nothing` (i.e.
    automatic backend selection). Valid choices include jacobian backends from
    `DifferentiationInterface.jl`.
  - `max_trust_radius`: the maximum radius of the trust region. Defaults to
    `max(norm(f(u0)), maximum(u0) - minimum(u0))`.
  - `initial_trust_radius`: the initial trust region radius. Defaults to
    `max_trust_radius / 11`.
  - `step_threshold`: the threshold for taking a step. In every iteration, the threshold is
    compared with a value `r`, which is the actual reduction in the objective function
    divided by the predicted reduction. If `step_threshold > r` the model is not a good
    approximation, and the step is rejected. Defaults to `0.0001`. For more details, see
    [Rahpeymaii, F.](https://link.springer.com/article/10.1007/s40096-020-00339-4)
  - `shrink_threshold`: the threshold for shrinking the trust region radius. In every
    iteration, the threshold is compared with a value `r` which is the actual reduction in
    the objective function divided by the predicted reduction. If `shrink_threshold > r` the
    trust region radius is shrunk by `shrink_factor`. Defaults to `0.25`. For more details,
    see [Rahpeymaii, F.](https://link.springer.com/article/10.1007/s40096-020-00339-4)
  - `expand_threshold`: the threshold for expanding the trust region radius. If a step is
    taken, i.e `step_threshold < r` (with `r` defined in `shrink_threshold`), a check is
    also made to see if `expand_threshold < r`. If that is true, the trust region radius is
    expanded by `expand_factor`. Defaults to `0.75`.
  - `shrink_factor`: the factor to shrink the trust region radius with if
    `shrink_threshold > r` (with `r` defined in `shrink_threshold`). Defaults to `0.25`.
  - `expand_factor`: the factor to expand the trust region radius with if
    `expand_threshold < r` (with `r` defined in `shrink_threshold`). Defaults to `2.0`.
  - `max_shrink_times`: the maximum number of times to shrink the trust region radius in a
    row, `max_shrink_times` is exceeded, the algorithm returns. Defaults to `32`.
  - `nlsolve_update_rule`: If set to `Val(true)`, updates the trust region radius using the
    update rule from NLSolve.jl. Defaults to `Val(false)`. If set to `Val(true)`, few of the
    radius update parameters -- `step_threshold = 0.05`, `expand_threshold = 0.9`, and
    `shrink_factor = 0.5` -- have different defaults.
"""
@kwdef @concrete struct SimpleTrustRegion <: AbstractSimpleNonlinearSolveAlgorithm
    autodiff = nothing
    max_trust_radius = 0.0
    initial_trust_radius = 0.0
    step_threshold = 0.0001
    shrink_threshold = nothing
    expand_threshold = nothing
    shrink_factor = nothing
    expand_factor = 2.0
    max_shrink_times::Int = 32
    nlsolve_update_rule = Val(false)
end

function SciMLBase.__solve(
        prob::Union{ImmutableNonlinearProblem, NonlinearLeastSquaresProblem},
        alg::SimpleTrustRegion, args...;
        abstol = nothing, reltol = nothing, maxiters = 1000,
        alias_u0 = false, termination_condition = nothing, kwargs...
)
    x = NLBUtils.maybe_unaliased(prob.u0, alias_u0)
    T = eltype(x)
    Δₘₐₓ = T(alg.max_trust_radius)
    Δ = T(alg.initial_trust_radius)
    η₁ = T(alg.step_threshold)

    if alg.shrink_threshold === nothing
        η₂ = T(ifelse(NLBUtils.unwrap_val(alg.nlsolve_update_rule), 0.05, 0.25))
    else
        η₂ = T(alg.shrink_threshold)
    end

    if alg.expand_threshold === nothing
        η₃ = T(ifelse(NLBUtils.unwrap_val(alg.nlsolve_update_rule), 0.9, 0.75))
    else
        η₃ = T(alg.expand_threshold)
    end

    if alg.shrink_factor === nothing
        t₁ = T(ifelse(NLBUtils.unwrap_val(alg.nlsolve_update_rule), 0.5, 0.25))
    else
        t₁ = T(alg.shrink_factor)
    end

    t₂ = T(alg.expand_factor)
    max_shrink_times = alg.max_shrink_times

    autodiff = SciMLBase.has_jac(prob.f) ? alg.autodiff :
               NonlinearSolveBase.select_jacobian_autodiff(prob, alg.autodiff)

    fx = NLBUtils.evaluate_f(prob, x)
    norm_fx = L2_NORM(fx)

    @bb xo = copy(x)
    fx_cache = (SciMLBase.isinplace(prob) && !SciMLBase.has_jac(prob.f)) ?
               NLBUtils.safe_similar(fx) : fx
    jac_cache = Utils.prepare_jacobian(prob, autodiff, fx_cache, x)
    J = Utils.compute_jacobian!!(nothing, prob, autodiff, fx_cache, x, jac_cache)

    abstol, reltol,
    tc_cache = NonlinearSolveBase.init_termination_cache(
        prob, abstol, reltol, fx, x, termination_condition, Val(:simple)
    )

    # Set default trust region radius if not specified by user.
    iszero(Δₘₐₓ) && (Δₘₐₓ = max(L2_NORM(fx), maximum(x) - minimum(x)))
    if iszero(Δ)
        if NLBUtils.unwrap_val(alg.nlsolve_update_rule)
            norm_x = L2_NORM(x)
            Δ = T(ifelse(norm_x > 0, norm_x, 1))
        else
            Δ = T(Δₘₐₓ / 11)
        end
    end

    fₖ = 0.5 * norm_fx^2
    H = transpose(J) * J
    g = NLBUtils.restructure(x, J' * NLBUtils.safe_vec(fx))
    shrink_counter = 0

    @bb δsd = copy(x)
    @bb δN_δsd = copy(x)
    @bb δN = copy(x)
    @bb Hδ = copy(x)
    dogleg_cache = (; δsd, δN_δsd, δN)

    solved, retcode, fx_sol, x_sol = Utils.check_termination(
        tc_cache, fx, x, xo, prob
    )
    solved && return SciMLBase.build_solution(prob, alg, x_sol, fx_sol; retcode)

    for _ in 1:maxiters
        # Solve the trust region subproblem.
        δ = dogleg_method!!(dogleg_cache, J, fx, g, Δ)
        @bb @. x = xo + δ

        fx = NLBUtils.evaluate_f!!(prob, fx, x)

        fₖ₊₁ = L2_NORM(fx)^2 / T(2)

        # Compute the ratio of the actual to predicted reduction.
        @bb Hδ = H × vec(δ)
        r = (fₖ₊₁ - fₖ) / (dot(δ, g) + (dot(δ, Hδ) / T(2)))

        # Update the trust region radius.
        if r ≥ η₂
            shrink_counter = 0
        else
            Δ = t₁ * Δ
            shrink_counter += 1
            shrink_counter > max_shrink_times && return SciMLBase.build_solution(
                prob, alg, x, fx; retcode = ReturnCode.ShrinkThresholdExceeded)
        end

        if r ≥ η₁
            # Termination Checks
            solved, retcode, fx_sol,
            x_sol = Utils.check_termination(
                tc_cache, fx, x, xo, prob
            )
            solved && return SciMLBase.build_solution(prob, alg, x_sol, fx_sol; retcode)

            # Take the step.
            @bb copyto!(xo, x)

            J = Utils.compute_jacobian!!(J, prob, autodiff, fx_cache, x, jac_cache)
            fx = NLBUtils.evaluate_f!!(prob, fx, x)

            # Update the trust region radius.
            if !NLBUtils.unwrap_val(alg.nlsolve_update_rule) && r > η₃
                Δ = min(t₂ * Δ, Δₘₐₓ)
            end
            fₖ = fₖ₊₁

            @bb H = transpose(J) × J
            @bb g = transpose(J) × vec(fx)
        end

        if NLBUtils.unwrap_val(alg.nlsolve_update_rule)
            if r > η₃
                Δ = t₂ * L2_NORM(δ)
            elseif r > 0.5
                Δ = max(Δ, t₂ * L2_NORM(δ))
            end
        end
    end

    return SciMLBase.build_solution(prob, alg, x, fx; retcode = ReturnCode.MaxIters)
end

function dogleg_method!!(cache, J, f::F, g, Δ) where F
    (; δsd, δN_δsd, δN) = cache

    # Compute the Newton step
    @bb δN .= NLBUtils.restructure(δN, J \ NLBUtils.safe_vec(f))
    @bb δN .*= -1
    # Test if the full step is within the trust region
    (L2_NORM(δN) ≤ Δ) && return δN

    # Calculate Cauchy point, optimum along the steepest descent direction
    @bb δsd .= g
    @bb @. δsd *= -1
    norm_δsd = L2_NORM(δsd)

    if (norm_δsd ≥ Δ)
        @bb @. δsd *= Δ / norm_δsd
        return δsd
    end

    # Find the intersection point on the boundary
    @bb @. δN_δsd = δN - δsd
    dot_δN_δsd = dot(δN_δsd, δN_δsd)
    dot_δsd_δN_δsd = dot(δsd, δN_δsd)
    dot_δsd = dot(δsd, δsd)
    fact = dot_δsd_δN_δsd^2 - dot_δN_δsd * (dot_δsd - Δ^2)
    tau = (-dot_δsd_δN_δsd + sqrt(fact)) / dot_δN_δsd
    @bb @. δsd += tau * δN_δsd
    return δsd
end
