"""
```julia
ITP(; k1::Real = 0.007, k2::Real = 1.5, n0::Int = 10)
```

ITP (Interpolate Truncate & Project)

Use the [ITP method](https://en.wikipedia.org/wiki/ITP_method) to find
a root of a bracketed function, with a convergence rate between 1 and 1.62.

This method was introduced in the paper "An Enhancement of the Bisection Method
Average Performance Preserving Minmax Optimality"
(https://doi.org/10.1145/3423597) by I. F. D. Oliveira and R. H. C. Takahashi.

# Tuning Parameters

The following keyword parameters are accepted.

- `n₀::Int = 1`, the 'slack'. Must not be negative.\n
  When n₀ = 0 the worst-case is identical to that of bisection,
  but increacing n₀ provides greater oppotunity for superlinearity.
- `κ₁::Float64 = 0.1`. Must not be negative.\n
  The recomended value is `0.2/(x₂ - x₁)`.
  Lower values produce tighter asymptotic behaviour, while higher values
  improve the steady-state behaviour when truncation is not helpful.
- `κ₂::Real = 2`. Must lie in [1, 1+ϕ ≈ 2.62).\n
  Higher values allow for a greater convergence rate,
  but also make the method more succeptable to worst-case performance.
  In practice, κ=1,2 seems to work well due to the computational simplicity,
  as κ₂ is used as an exponent in the method.

### Worst Case Performance

n½ + `n₀` iterations, where n½ is the number of iterations using bisection
(n½ = ⌈log2(Δx)/2`tol`⌉).

### Asymptotic Performance

If `f` is twice differentiable and the root is simple,
then with `n₀` > 0 the convergence rate is √`κ₂`.
"""
struct ITP{T} <: AbstractBracketingAlgorithm
    k1::T
    k2::T
    n0::Int
    function ITP(; k1::Real = 0.007, k2::Real = 1.5, n0::Int = 10)
        if k1 < 0
            error("Hyper-parameter κ₁ should not be negative")
        end
        if n0 < 0
            error("Hyper-parameter n₀ should not be negative")
        end
        if k2 < 1 || k2 > (1.5 + sqrt(5) / 2)
            ArgumentError("Hyper-parameter κ₂ should be between 1 and 1 + ϕ where ϕ ≈ 1.618... is the golden ratio")
        end
        T = promote_type(eltype(k1), eltype(k2))
        return new{T}(k1, k2, n0)
    end
end

function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::ITP,
    args...; abstol = 1.0e-15,
    maxiters = 1000, kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    left, right = prob.tspan # a and b
    fl, fr = f(left), f(right)
    ϵ = abstol
    if iszero(fl)
        return SciMLBase.build_solution(prob, alg, left, fl;
            retcode = ReturnCode.ExactSolutionLeft, left = left,
            right = right)
    elseif iszero(fr)
        return SciMLBase.build_solution(prob, alg, right, fr;
            retcode = ReturnCode.ExactSolutionRight, left = left,
            right = right)
    end
    #defining variables/cache
    k1 = alg.k1
    k2 = alg.k2
    n0 = alg.n0
    n_h = ceil(log2(abs(right - left) / (2 * ϵ)))
    mid = (left + right) / 2
    x_f = (fr * left - fl * right) / (fr - fl)
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
        δ = k1 * (span^k2)

        ## Interpolation step ##
        x_f = (fr * left - fl * right) / (fr - fl)

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

        ## Update ##
        tmin, tmax = minmax(left, right)
        xp >= tmax && (xp = prevfloat(tmax))
        xp <= tmin && (xp = nextfloat(tmin))
        yp = f(xp)
        yps = yp * sign(fr)
        if yps > 0
            right = xp
            fr = yp
        elseif yps < 0
            left = xp
            fl = yp
        else
            return SciMLBase.build_solution(prob, alg, xp, yps;
                                            retcode = ReturnCode.Success, left = left,
                                            right = right)
        end
        i += 1
        mid = (left + right) / 2
        ϵ_s /= 2

        if nextfloat_tdir(left, prob.tspan...) == right
            return SciMLBase.build_solution(prob, alg, left, fl;
                retcode = ReturnCode.FloatingPointLimit, left = left,
                right = right)
        end
    end
    return SciMLBase.build_solution(prob, alg, left, fl; retcode = ReturnCode.MaxIters,
        left = left, right = right)
end
