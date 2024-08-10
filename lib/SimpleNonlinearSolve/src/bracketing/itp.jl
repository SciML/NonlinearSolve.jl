"""
    ITP(; k1::Real = 0.007, k2::Real = 1.5, n0::Int = 10)

ITP (Interpolate Truncate & Project)

Use the [ITP method](https://en.wikipedia.org/wiki/ITP_method) to find a root of a bracketed
function, with a convergence rate between 1 and 1.62.

This method was introduced in the paper "An Enhancement of the Bisection Method Average
Performance Preserving Minmax Optimality" (https://doi.org/10.1145/3423597) by
I. F. D. Oliveira and R. H. C. Takahashi.

# Tuning Parameters

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
struct ITP{T₁, T₂} <: AbstractBracketingAlgorithm
    scaled_k1::T₁
    k2::T₂
    n0::Int
    function ITP(;
            scaled_k1::T₁ = 0.2, k2::T₂ = 2, n0::Int = 10) where {T₁ <: Real, T₂ <: Real}
        scaled_k1 < 0 && error("Hyper-parameter κ₁ should not be negative")
        n0 < 0 && error("Hyper-parameter n₀ should not be negative")
        if k2 < 1 || k2 > (1.5 + sqrt(5) / 2)
            throw(ArgumentError("Hyper-parameter κ₂ should be between 1 and 1 + ϕ where \
                                 ϕ ≈ 1.618... is the golden ratio"))
        end
        return new{T₁, T₂}(scaled_k1, k2, n0)
    end
end

function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::ITP, args...;
        maxiters = 1000, abstol = nothing, kwargs...)
    @assert !isinplace(prob) "`Bisection` only supports OOP problems."
    f = Base.Fix2(prob.f, prob.p)
    left, right = prob.tspan
    fl, fr = f(left), f(right)

    abstol = __get_tolerance(
        nothing, abstol, promote_type(eltype(first(prob.tspan)), eltype(last(prob.tspan))))

    if iszero(fl)
        return build_solution(
            prob, alg, left, fl; retcode = ReturnCode.ExactSolutionLeft, left, right)
    end

    if iszero(fr)
        return build_solution(
            prob, alg, right, fr; retcode = ReturnCode.ExactSolutionRight, left, right)
    end
    ϵ = abstol
    #defining variables/cache
    k2 = alg.k2
    k1 = alg.scaled_k1 * abs(right - left)^(1 - k2)
    n0 = alg.n0
    n_h = ceil(log2(abs(right - left) / (2 * ϵ)))
    mid = (left + right) / 2
    x_f = left + (right - left) * (fl / (fl - fr))
    xt = left
    xp = left
    r = zero(left) #minmax radius
    δ = zero(left) # truncation error
    σ = 1.0
    ϵ_s = ϵ * 2^(n_h + n0)
    i = 0 #iteration
    while i <= maxiters
        span = abs(right - left)
        r = ϵ_s - (span / 2)
        δ = k1 * ((k2 == 2) ? span^2 : (span^k2))

        ## Interpolation step ##
        x_f = left + (right - left) * (fl / (fl - fr))

        ## Truncation step ##
        σ = sign(mid - x_f)
        if δ <= abs(mid - x_f)
            xt = x_f + (σ * δ)
        else
            xt = mid
        end

        ## Projection step ##
        if abs(xt - mid) <= r
            xp = xt
        else
            xp = mid - (σ * r)
        end

        if abs((left - right) / 2) < ϵ
            return build_solution(
                prob, alg, mid, f(mid); retcode = ReturnCode.Success, left, right)
        end

        ## Update ##
        tmin, tmax = minmax(left, right)
        xp >= tmax && (xp = prevfloat(tmax))
        xp <= tmin && (xp = nextfloat(tmin))
        yp = f(xp)
        yps = yp * sign(fr)
        T0 = zero(yps)
        if yps > T0
            right = xp
            fr = yp
        elseif yps < T0
            left = xp
            fl = yp
        else
            return build_solution(
                prob, alg, xp, yps; retcode = ReturnCode.Success, left = xp, right = xp)
        end
        i += 1
        mid = (left + right) / 2
        ϵ_s /= 2

        if __nextfloat_tdir(left, prob.tspan...) == right
            return build_solution(
                prob, alg, left, fl; left, right, retcode = ReturnCode.FloatingPointLimit)
        end
    end

    return build_solution(prob, alg, left, fl; retcode = ReturnCode.MaxIters, left, right)
end
