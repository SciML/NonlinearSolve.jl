"""
    Bisection(; exact_left = false, exact_right = false)

A common bisection method.

### Keyword Arguments

  - `exact_left`: whether to enforce whether the left side of the interval must be exactly
    zero for the returned result. Defaults to false.
  - `exact_right`: whether to enforce whether the right side of the interval must be exactly
    zero for the returned result. Defaults to false.
"""
@kwdef struct Bisection <: AbstractBracketingAlgorithm
    exact_left::Bool = false
    exact_right::Bool = false
end

function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::Bisection, args...;
        maxiters = 1000, abstol = min(eps(prob.tspan[1]), eps(prob.tspan[2])),
        kwargs...)
    @assert !isinplace(prob) "Bisection only supports OOP problems."
    f = Base.Fix2(prob.f, prob.p)
    left, right = prob.tspan
    fl, fr = f(left), f(right)

    if iszero(fl)
        return build_solution(prob, alg, left, fl; retcode = ReturnCode.ExactSolutionLeft,
            left, right)
    end

    if iszero(fr)
        return build_solution(prob, alg, right, fr; retcode = ReturnCode.ExactSolutionRight,
            left, right)
    end

    for _ in 1:maxiters
        mid = (left + right) / 2
        if (mid == left || mid == right)
            return build_solution(prob, alg, left, fl; left, right,
                retcode = ReturnCode.FloatingPointLimit)
        end

        fm = f(mid)
        if abs((right - left) / 2) < abstol || iszero(fm)
            return build_solution(prob, alg, mid, fm; left, right,
                retcode = ReturnCode.Success)
        end

        if sign(fl * fm) < 0
            right, fr = mid, fm
        else
            left, fl = mid, fm
        end
    end

    return build_solution(prob, alg, left, fl; retcode = ReturnCode.MaxIters, left, right)
end
