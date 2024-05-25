"""
    Bisection(; exact_left = false, exact_right = false)

A common bisection method.

### Keyword Arguments

  - `exact_left`: whether to enforce whether the left side of the interval must be exactly
    zero for the returned result. Defaults to false.
  - `exact_right`: whether to enforce whether the right side of the interval must be exactly
    zero for the returned result. Defaults to false.

!!! warning

    Currently, the keyword arguments are not implemented.
"""
@kwdef struct Bisection <: AbstractBracketingAlgorithm
    exact_left::Bool = false
    exact_right::Bool = false
end

function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::Bisection,
        args...; maxiters = 1000, abstol = nothing, kwargs...)
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

    i = 1
    if !iszero(fr)
        while i < maxiters
            mid = (left + right) / 2
            (mid == left || mid == right) &&
                return build_solution(prob, alg, left, fl; left, right,
                    retcode = ReturnCode.FloatingPointLimit)
            fm = f(mid)
            if abs((right - left) / 2) < abstol
                return build_solution(
                    prob, alg, mid, fm; retcode = ReturnCode.Success, left, right)
            end
            if iszero(fm)
                right = mid
                break
            end
            if sign(fl) == sign(fm)
                fl = fm
                left = mid
            else
                fr = fm
                right = mid
            end
            i += 1
        end
    end

    sol, i, left, right, fl, fr = __bisection(
        left, right, fl, fr, f; abstol, maxiters = maxiters - i, prob, alg)

    sol !== nothing && return sol

    return build_solution(prob, alg, left, fl; retcode = ReturnCode.MaxIters, left, right)
end

function __bisection(left, right, fl, fr, f::F; abstol, maxiters, prob, alg) where {F}
    i = 1
    sol = nothing
    while i < maxiters
        mid = (left + right) / 2
        if (mid == left || mid == right)
            sol = build_solution(
                prob, alg, left, fl; left, right, retcode = ReturnCode.FloatingPointLimit)
            break
        end

        fm = f(mid)
        if abs((right - left) / 2) < abstol
            sol = build_solution(
                prob, alg, mid, fm; left, right, retcode = ReturnCode.Success)
            break
        end

        if iszero(fm)
            right = mid
            fr = fm
        else
            left = mid
            fl = fm
        end

        i += 1
    end

    return sol, i, left, right, fl, fr
end
