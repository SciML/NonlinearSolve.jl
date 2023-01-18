"""
```julia
SimpleTrustRegion(; chunk_size = Val{0}(),
                    autodiff = Val{true}(),
                    diff_type = Val{:forward},
                    max_trust_radius::Real = 0.0,
                    initial_trust_radius::Real = 0.0,
                    step_threshold::Real = 0.1,
                    shrink_threshold::Real = 0.25,
                    expand_threshold::Real = 0.75,
                    shrink_factor::Real = 0.25,
                    expand_factor::Real = 2.0,
                    max_shrink_times::Int = 32
```

A low-overhead implementation of a
[trust-region](https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods)
solver

### Keyword Arguments

- `chunk_size`: the chunk size used by the internal ForwardDiff.jl automatic differentiation
  system. This allows for multiple derivative columns to be computed simultaneously,
  improving performance. Defaults to `0`, which is equivalent to using ForwardDiff.jl's
  default chunk size mechanism. For more details, see the documentation for
  [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/).
- `autodiff`: whether to use forward-mode automatic differentiation for the Jacobian.
  Note that this argument is ignored if an analytical Jacobian is passed; as that will be
  used instead. Defaults to `Val{true}`, which means ForwardDiff.jl is used by default.
  If `Val{false}`, then FiniteDiff.jl is used for finite differencing.
- `diff_type`: the type of finite differencing used if `autodiff = false`. Defaults to
  `Val{:forward}` for forward finite differences. For more details on the choices, see the
  [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl) documentation.
- `max_trust_radius`: the maximum radius of the trust region. Defaults to
  `max(norm(f(u0)), maximum(u0) - minimum(u0))`.
- `initial_trust_radius`: the initial trust region radius. Defaults to
  `max_trust_radius / 11`.
- `step_threshold`: the threshold for taking a step. In every iteration, the threshold is
  compared with a value `r`, which is the actual reduction in the objective function divided
  by the predicted reduction. If `step_threshold > r` the model is not a good approximation,
  and the step is rejected. Defaults to `0.1`. For more details, see
  [Trust-region methods](https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods)
- `shrink_threshold`: the threshold for shrinking the trust region radius. In every
  iteration, the threshold is compared with a value `r` which is the actual reduction in the
  objective function divided by the predicted reduction. If `shrink_threshold > r` the trust
  region radius is shrunk by `shrink_factor`. Defaults to `0.25`. For more details, see
  [Trust-region methods](https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods)
- `expand_threshold`: the threshold for expanding the trust region radius. If a step is
  taken, i.e `step_threshold < r` (with `r` defined in `shrink_threshold`), a check is also
  made to see if `expand_threshold < r`. If that is true, the trust region radius is
  expanded by `expand_factor`. Defaults to `0.75`.
- `shrink_factor`: the factor to shrink the trust region radius with if
  `shrink_threshold > r` (with `r` defined in `shrink_threshold`). Defaults to `0.25`.
- `expand_factor`: the factor to expand the trust region radius with if
  `expand_threshold < r` (with `r` defined in `shrink_threshold`). Defaults to `2.0`.
- `max_shrink_times`: the maximum number of times to shrink the trust region radius in a
  row, `max_shrink_times` is exceeded, the algorithm returns. Defaults to `32`.
"""
struct SimpleTrustRegion{T, CS, AD, FDT} <: AbstractNewtonAlgorithm{CS, AD, FDT}
    max_trust_radius::T
    initial_trust_radius::T
    step_threshold::T
    shrink_threshold::T
    expand_threshold::T
    shrink_factor::T
    expand_factor::T
    max_shrink_times::Int
    function SimpleTrustRegion(; chunk_size = Val{0}(),
                               autodiff = Val{true}(),
                               diff_type = Val{:forward},
                               max_trust_radius::Real = 0.0,
                               initial_trust_radius::Real = 0.0,
                               step_threshold::Real = 0.1,
                               shrink_threshold::Real = 0.25,
                               expand_threshold::Real = 0.75,
                               shrink_factor::Real = 0.25,
                               expand_factor::Real = 2.0,
                               max_shrink_times::Int = 32)
        new{typeof(initial_trust_radius),
            SciMLBase._unwrap_val(chunk_size),
            SciMLBase._unwrap_val(autodiff),
            SciMLBase._unwrap_val(diff_type)}(max_trust_radius,
                                              initial_trust_radius,
                                              step_threshold,
                                              shrink_threshold,
                                              expand_threshold,
                                              shrink_factor,
                                              expand_factor,
                                              max_shrink_times)
    end
end

function SciMLBase.__solve(prob::NonlinearProblem,
                           alg::SimpleTrustRegion, args...; abstol = nothing,
                           reltol = nothing,
                           maxiters = 1000, kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)
    T = typeof(x)
    Δₘₐₓ = float(alg.max_trust_radius)
    Δ = float(alg.initial_trust_radius)
    η₁ = float(alg.step_threshold)
    η₂ = float(alg.shrink_threshold)
    η₃ = float(alg.expand_threshold)
    t₁ = float(alg.shrink_factor)
    t₂ = float(alg.expand_factor)
    max_shrink_times = alg.max_shrink_times

    if SciMLBase.isinplace(prob)
        error("SimpleTrustRegion currently only supports out-of-place nonlinear problems")
    end

    atol = abstol !== nothing ? abstol :
           real(oneunit(eltype(T))) * (eps(real(one(eltype(T)))))^(4 // 5)
    rtol = reltol !== nothing ? reltol : eps(real(one(eltype(T))))^(4 // 5)

    if alg_autodiff(alg)
        F, ∇f = value_derivative(f, x)
    elseif x isa AbstractArray
        F = f(x)
        ∇f = FiniteDiff.finite_difference_jacobian(f, x, diff_type(alg), eltype(x), F)
    else
        F = f(x)
        ∇f = FiniteDiff.finite_difference_derivative(f, x, diff_type(alg), eltype(x), F)
    end

    # Set default trust region radius if not specified by user.
    if Δₘₐₓ == 0.0
        Δₘₐₓ = max(norm(F), maximum(x) - minimum(x))
    end
    if Δ == 0.0
        Δ = Δₘₐₓ / 11
    end

    fₖ = 0.5 * norm(F)^2
    H = ∇f * ∇f
    g = ∇f * F
    shrink_counter = 0

    for k in 1:maxiters
        # Solve the trust region subproblem.
        δ = dogleg_method(H, g, Δ)
        xₖ₊₁ = x + δ
        Fₖ₊₁ = f(xₖ₊₁)
        fₖ₊₁ = 0.5 * norm(Fₖ₊₁)^2

        # Compute the ratio of the actual to predicted reduction.
        model = -(δ' * g + 0.5 * δ' * H * δ)
        r = model \ (fₖ - fₖ₊₁)

        # Update the trust region radius.
        if r < η₂
            Δ = t₁ * Δ
            shrink_counter += 1
            if shrink_counter > max_shrink_times
                return SciMLBase.build_solution(prob, alg, x, F;
                                                retcode = ReturnCode.Success)
            end
        else
            shrink_counter = 0
        end
        if r > η₁
            if isapprox(xₖ₊₁, x, atol = atol, rtol = rtol)
                return SciMLBase.build_solution(prob, alg, xₖ₊₁, Fₖ₊₁;
                                                retcode = ReturnCode.Success)
            end
            # Take the step.
            x = xₖ₊₁
            F = Fₖ₊₁
            if alg_autodiff(alg)
                F, ∇f = value_derivative(f, x)
            elseif x isa AbstractArray
                ∇f = FiniteDiff.finite_difference_jacobian(f, x, diff_type(alg), eltype(x),
                                                           F)
            else
                ∇f = FiniteDiff.finite_difference_derivative(f, x, diff_type(alg),
                                                             eltype(x),
                                                             F)
            end

            iszero(F) &&
                return SciMLBase.build_solution(prob, alg, x, F;
                                                retcode = ReturnCode.Success)

            # Update the trust region radius.
            if r > η₃ && norm(δ) ≈ Δ
                Δ = min(t₂ * Δ, Δₘₐₓ)
            end
            fₖ = fₖ₊₁
            H = ∇f * ∇f
            g = ∇f * F
        end
    end
    return SciMLBase.build_solution(prob, alg, x, F; retcode = ReturnCode.MaxIters)
end
