"""
    ITP(; k1::Real = 0.007, k2::Real = 1.5, n0::Int = 10)

ITP (Interpolate Truncate & Project)

Use the [ITP method](https://en.wikipedia.org/wiki/ITP_method) to find a root of a bracketed
function, with a convergence rate between 1 and 1.62.

This method was introduced in the paper "An Enhancement of the Bisection Method Average
Performance Preserving Minmax Optimality" (https://doi.org/10.1145/3423597) by
I. F. D. Oliveira and R. H. C. Takahashi.

### Tuning Parameters

The following keyword parameters are accepted.

  - `n₀::Int = 10`, the 'slack'. Must not be negative. When n₀ = 0 the worst-case is
    identical to that of bisection, but increasing n₀ provides greater opportunity for
    superlinearity.
  - `scaled_κ₁::Float64 = 0.2`. Must not be negative. The recommended value is `0.2`.
    Lower values produce tighter asymptotic behaviour, while higher values improve the
    steady-state behaviour when truncation is not helpful.
  - `κ₂::Real = 2`. Must lie in [1, 1+ϕ ≈ 2.62). Higher values allow for a greater
    convergence rate, but also make the method more succeptable to worst-case performance.
    In practice, κ₂=1, 2 seems to work well due to the computational simplicity, as κ₂ is
    used as an exponent in the method.

### Computation of κ₁

In the current implementation, we compute κ₁ = scaled_κ₁·|Δx₀|^(1 - κ₂); this allows κ₁ to
adapt to the length of the interval and keep the proposed steps proportional to Δx.

### Worst Case Performance

n½ + `n₀` iterations, where n½ is the number of iterations using bisection
(n½ = ⌈log2(Δx)/2`tol`⌉).

### Asymptotic Performance

If `f` is twice differentiable and the root is simple, then with `n₀` > 0 the convergence
rate is √`κ₂`.
"""
@concrete struct ITP <: AbstractBracketingAlgorithm
    scaled_k1
    k2
    n0::Int
end

function ITP(; scaled_k1::Real = 0.2, k2::Real = 2, n0::Int = 10)
    scaled_k1 < 0 && error("Hyper-parameter κ₁ should not be negative")
    n0 < 0 && error("Hyper-parameter n₀ should not be negative")
    if k2 < 1 || k2 > (1.5 + sqrt(5) / 2)
        throw(ArgumentError("Hyper-parameter κ₂ should be between 1 and 1 + ϕ where \
                             ϕ ≈ 1.618... is the golden ratio"))
    end
    return ITP(scaled_k1, k2, n0)
end

function CommonSolve.solve(prob::IntervalNonlinearProblem, alg::ITP, args...;
        maxiters = 1000, abstol = nothing, verbose::Bool = true, kwargs...)
    @assert !SciMLBase.isinplace(prob) "`ITP` only supports out-of-place problems."

    f = Base.Fix2(prob.f, prob.p)
    left, right = prob.tspan
    fl, fr = f(left), f(right)

    abstol = NonlinearSolveBase.get_tolerance(
        abstol, promote_type(eltype(left), eltype(right)))

    if iszero(fl)
        return SciMLBase.build_solution(
            prob, alg, left, fl; retcode = ReturnCode.ExactSolutionLeft, left, right)
    end

    if iszero(fr)
        return SciMLBase.build_solution(
            prob, alg, right, fr; retcode = ReturnCode.ExactSolutionRight, left, right)
    end

    if sign(fl) == sign(fr)
        verbose &&
            @warn "The interval is not an enclosing interval, opposite signs at the \
                   boundaries are required."
        return SciMLBase.build_solution(
            prob, alg, left, fl; retcode = ReturnCode.InitialFailure, left, right)
    end

    ϵ = abstol
    k2 = alg.k2
    k1 = alg.scaled_k1 * abs(right - left)^(1 - k2)
    n0 = alg.n0
    n_h = ceil(log2(abs(right - left) / (2 * ϵ)))
    mid = (left + right) / 2
    x_f = left + (right - left) * (fl / (fl - fr))
    xt = left
    xp = left
    r = zero(left) # minmax radius
    δ = zero(left) # truncation error
    σ = 1.0
    ϵ_s = ϵ * 2^(n_h + n0)

    i = 1
    while i ≤ maxiters
        span = abs(right - left)
        r = ϵ_s - (span / 2)
        δ = k1 * span^k2

        x_f = left + (right - left) * (fl / (fl - fr))  # Interpolation Step

        diff = mid - x_f
        σ = sign(diff)
        xt = ifelse(δ ≤ diff, x_f + σ * δ, mid)  # Truncation Step

        xp = ifelse(abs(xt - mid) ≤ r, xt, mid - σ * r)  # Projection Step

        if abs((left - right) / 2) < ϵ
            return SciMLBase.build_solution(
                prob, alg, xt, f(xt); retcode = ReturnCode.Success, left, right)
        end

        # update
        tmin, tmax = minmax(xt, xp)
        xp ≥ tmax && (xp = prevfloat(tmax))
        xp ≤ tmin && (xp = nextfloat(tmin))
        yp = f(xp)
        yps = yp * sign(fr)
        T0 = zero(yps)
        if yps > T0
            right, fr = xp, yp
        elseif yps < T0
            left, fl = xp, yp
        else
            return SciMLBase.build_solution(
                prob, alg, xp, yps; retcode = ReturnCode.Success, left, right)
        end

        i += 1
        mid = (left + right) / 2
        ϵ_s /= 2

        if Impl.nextfloat_tdir(left, prob.tspan...) == right
            return SciMLBase.build_solution(
                prob, alg, right, fr; retcode = ReturnCode.FloatingPointLimit, left, right)
        end
    end

    return SciMLBase.build_solution(
        prob, alg, left, fl; retcode = ReturnCode.MaxIters, left, right)
end
