"""
    Ridder()

A non-allocating ridder method.
"""
struct Ridder <: AbstractBracketingAlgorithm end

function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::Ridder, args...;
        maxiters = 1000, abstol = nothing, kwargs...)
    @assert !isinplace(prob) "`Ridder` only supports OOP problems."
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

    if sign(fl) == sign(fr)
        @warn "The interval is not an enclosing interval (does not contain a root). Returning boundary value."
        return build_solution(
            prob, alg, left, fl; retcode = ReturnCode.InitialFailure, left, right)
    end

    xo = oftype(left, Inf)
    i = 1
    if !iszero(fr)
        while i < maxiters
            mid = (left + right) / 2
            (mid == left || mid == right) &&
                return build_solution(prob, alg, left, fl; left, right,
                    retcode = ReturnCode.FloatingPointLimit)
            fm = f(mid)
            s = sqrt(fm^2 - fl * fr)
            if iszero(s)
                return build_solution(
                    prob, alg, left, fl; left, right, retcode = ReturnCode.Failure)
            end
            x = mid + (mid - left) * sign(fl - fr) * fm / s
            fx = f(x)
            xo = x
            if abs((right - left) / 2) < abstol
                return build_solution(
                    prob, alg, mid, fm; retcode = ReturnCode.Success, left, right)
            end
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

    sol, i, left, right, fl, fr = __bisection(
        left, right, fl, fr, f; abstol, maxiters = maxiters - i, prob, alg)
    sol !== nothing && return sol

    return SciMLBase.build_solution(
        prob, alg, left, fl; retcode = ReturnCode.MaxIters, left, right)
end
