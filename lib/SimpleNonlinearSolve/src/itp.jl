"""
```julia
Itp(; k1 = Val{1}(), k2 = Val{2}(), n0 = Val{1}())
```
ITP (Interpolate Truncate & Project)


"""

struct Itp <: AbstractBracketingAlgorithm
    k1::Real
    k2::Real
    n0::Int
    function Itp(; k1::Real = 0.007, k2::Real = 1.5, n0::Int = 10)
        if k1 < 0
            error("Hyper-parameter κ₁ should not be negative")
        end
        if n0 < 0
            error("Hyper-parameter n₀ should not be negative")
        end
        if k2 < 1 || k2 > (1.5 + sqrt(5) / 2)
            ArgumentError("Hyper-parameter κ₂ should be between 1 and 1 + ϕ where ϕ ≈ 1.618... is the golden ratio")
        end
        return new(k1, k2, n0)
    end
end

function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::Itp,
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
    n_h = ceil(log2((right - left) / (2 * ϵ)))
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
        #mid = (left + right) / 2
        r = ϵ_s - ((right - left) / 2)
        δ = k1 * ((right - left)^k2)

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
        yp = f(xp)
        if yp > 0
            right = xp
            fr = yp
        elseif yp < 0
            left = xp
            fl = yp
        else
            left = xp
            right = xp
        end
        i += 1
        mid = (left + right) / 2
        ϵ_s /= 2

        if (right - left < 2 * ϵ)
            return SciMLBase.build_solution(prob, alg, mid, f(mid);
                retcode = ReturnCode.Success, left = left,
                right = right)
        end
    end
    return SciMLBase.build_solution(prob, alg, left, fl; retcode = ReturnCode.MaxIters,
        left = left, right = right)
end
