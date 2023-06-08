"""
`Ridder()`

A non-allocating ridder method

"""
struct Ridder <: AbstractBracketingAlgorithm end

function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::Ridder, args...;
    maxiters = 1000,
    kwargs...)
    f = Base.Fix2(prob.f, prob.p)
    left, right = prob.tspan
    fl, fr = f(left), f(right)

    if iszero(fl)
        return SciMLBase.build_solution(prob, alg, left, fl;
            retcode = ReturnCode.ExactSolutionLeft, left = left,
            right = right)
    end

    xo = oftype(left, Inf)
    i = 1
    if !iszero(fr)
        while i < maxiters
            mid = (left + right) / 2
            (mid == left || mid == right) &&
                return SciMLBase.build_solution(prob, alg, left, fl;
                    retcode = ReturnCode.FloatingPointLimit,
                    left = left, right = right)
            fm = f(mid)
            s = sqrt(fm^2 - fl * fr)
            iszero(s) &&
                return SciMLBase.build_solution(prob, alg, left, fl;
                    retcode = ReturnCode.Failure,
                    left = left, right = right)
            x = mid + (mid - left) * sign(fl - fr) * fm / s
            fx = f(x)
            xo = x
            if iszero(fx)
                right = x
                fr = fx
                break
            end
            if sign(fx) != sign(fm)
                left = mid
                fl = fm
                right = x
                fr = fx
            elseif sign(fx) != sign(fl)
                right = x
                fr = fx
            else
                @assert sign(fx) != sign(fr)
                left = x
                fl = fx
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
