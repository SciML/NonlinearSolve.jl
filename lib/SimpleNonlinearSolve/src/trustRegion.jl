"""
```julia
TrustRegion(max_trust_radius::Number; chunk_size = Val{0}(),
                               autodiff = Val{true}(), diff_type = Val{:forward})
```

A low-overhead implementation of a
[trust-region](https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods)
solver


### Keyword Arguments
- `max_trust_radius`: the maximum radius of the trust region. The step size in the algorithm
  will change dynamically. However, it will never be greater than the `max_trust_radius`.

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
"""
struct TrustRegion{CS, AD, FDT} <: AbstractNewtonAlgorithm{CS, AD, FDT}
    max_trust_radius::Number
    function TrustRegion(max_turst_radius::Number; chunk_size = Val{0}(),
                                            autodiff = Val{true}(),
                                            diff_type = Val{:forward})
        new{SciMLBase._unwrap_val(chunk_size), SciMLBase._unwrap_val(autodiff),
            SciMLBase._unwrap_val(diff_type)}(max_trust_radius)
    end
end

function SciMLBase.solve(prob::NonlinearProblem,
                         alg::TrustRegion, args...; abstol = nothing,
                         reltol = nothing,
                         maxiters = 1000, kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    x = float(prob.u0)
    T = typeof(x)
    Δₘₐₓ = float(alg.max_trust_radius)  # The maximum trust region radius.
    Δ = Δₘₐₓ / 5  # Initial trust region radius.
    η₁ = 0.1   # Threshold for taking a step.
    η₂ = 0.25  # Threshold for shrinking the trust region.
    η₃ = 0.75  # Threshold for expanding the trust region.
    t₁ = 0.25  # Factor to shrink the trust region with.
    t₂ = 2.0   # Factor to expand the trust region with.

    if SciMLBase.isinplace(prob)
        error("TrustRegion currently only supports out-of-place nonlinear problems")
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

    fₖ = 0.5 * norm(F)^2
    H = ∇f * ∇f
    g = ∇f * F

    for k in 1:maxiters
        # Solve the trust region subproblem.
        δ = dogleg_method(H, g, Δ)
        xₖ₊₁ = x + δ
        Fₖ₊₁ = f(xₖ₊₁)
        fₖ₊₁ = 0.5 * norm(Fₖ₊₁)^2

        # Compute the ratio of the actual to predicted reduction.
        model = -(δ' * g + 0.5 * δ' * H * δ)
        r = (fₖ - fₖ₊₁) / model

        # Update the trust region radius.
        if r < η₂
            Δ *= t₁
        if r > η₁
            if isapprox(x̂, x, atol = atol, rtol = rtol)
                return SciMLBase.build_solution(prob, alg, x, F;
                                                retcode = ReturnCode.Success)
            end

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
            fₖ = f̂
            H = ∇f * ∇f
            g = ∇f * F
        end
    end

    return SciMLBase.build_solution(prob, alg, x, F; retcode = ReturnCode.MaxIters)
end
