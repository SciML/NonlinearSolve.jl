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
end

function Itp(k1::Real = Val{1}(), k2::Real = Val{2}(), n0::Int = Val{1}())
    if k1 < 0
        ArgumentError("Hyper-parameter κ₁ should not be negative")
    end
    if !isa(n0, Int)
        ArgumentError("Hyper-parameter n₀ should be an Integer")
    end
    Itp(k1, k2, n0)
end

function SciMLBase.__solve(prob::IntervalNonlinearProblem, alg::Itp,
                            args..., abstol = nothing, reltol = nothing, 
                            maxiters = 1000, kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    left, right = prob.tspan # a and b
    fl, fr = f(left), f(right)
    ϵ = abstol
    if iszero(fl)
        return SciMLBase.build_solution(prob, alg, left, fl;
            retcode = ReturnCode.ExactSolutionLeft, left = left,
            right = right)
    end

    if iszero(fr)

    end
    #defining variables/cache
    k1 = alg.k1
    k2 = alg.k2
    n0 = alg.k3
    n_h = ceil(log2((right - left) / (2 * ϵ)))
    n_max = n_h + n0
    mid = (left + right) / 2
    x_f = (fr * left - fl * right) / (fr - fl)
    xt = left
    xp = left
    r = zero(left) #minmax radius
    δ = zero(left) # truncation error
    σ = 1.0
    i = 0 #iteration
    while i <= maxiters
        #mid = (left + right) / 2
        r = ϵ * 2 ^ (n_max - i) - ((right - left) / 2)
        δ = k1 * (right - left) ^ k2

        ## Interpolation step ##
        x_f =  (fr * left - fl * right) / (fr - fl)

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

        if (right - left < 2 * ϵ)
            return SciMLBase.build_solution(prob, alg, mid, fl;
            retcode = ReturnCode.Success, left = left,
            right = right)
        end
    end
    return SciMLBase.build_solution(prob, alg, left, fl; retcode = ReturnCode.MaxIters,
        left = left, right = right)
                               
end