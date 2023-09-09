"""
`Bisection(; exact_left = false, exact_right = false)`

A common bisection method.

### Keyword Arguments

- `exact_left`: whether to enforce whether the left side of the interval must be exactly
  zero for the returned result. Defaults to false.
- `exact_right`: whether to enforce whether the right side of the interval must be exactly
  zero for the returned result. Defaults to false.
"""
struct Bisection <: AbstractBracketingAlgorithm
    exact_left::Bool
    exact_right::Bool
end

function Bisection(; exact_left = false, exact_right = false)
    Bisection(exact_left, exact_right)
end

function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::Bisection, args...;
    maxiters = 1000, abstol = nothing,
    kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    left, right = prob.tspan
    fl, fr = f(left), f(right)
    atol = abstol !== nothing ? abstol : min(eps(left), eps(right))
    if iszero(fl)
        return SciMLBase.build_solution(prob, alg, left, fl;
            retcode = ReturnCode.ExactSolutionLeft, left = left,
            right = right)
    end
    if iszero(fr)
        return SciMLBase.build_solution(prob, alg, right, fr;
            retcode = ReturnCode.ExactSolutionRight, left = left,
            right = right)
    end

    i = 1
    if !iszero(fr)
        while i < maxiters
            mid = (left + right) / 2
            (mid == left || mid == right) &&
                return SciMLBase.build_solution(prob, alg, left, fl;
                    retcode = ReturnCode.FloatingPointLimit,
                    left = left, right = right)
            fm = f(mid)
            if abs((right - left) / 2) < atol
                return SciMLBase.build_solution(prob, alg, mid, fm;
                    retcode = ReturnCode.Success,
                    left = left, right = right)
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

    while i < maxiters
        mid = (left + right) / 2
        (mid == left || mid == right) &&
            return SciMLBase.build_solution(prob, alg, left, fl;
                retcode = ReturnCode.FloatingPointLimit,
                left = left, right = right)
        fm = f(mid)
        if abs((right - left) / 2) < atol
            return SciMLBase.build_solution(prob, alg, mid, fm;
                retcode = ReturnCode.Success,
                left = left, right = right)
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

    return SciMLBase.build_solution(prob, alg, left, fl; retcode = ReturnCode.MaxIters,
        left = left, right = right)
end
