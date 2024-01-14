"""
    Falsi()

A non-allocating regula falsi method.
"""
struct Falsi <: AbstractBracketingAlgorithm end

function SciMLBase.solve(prob::IntervalNonlinearProblem, alg::Falsi, args...;
        maxiters = 1000, abstol = nothing, kwargs...)
    @assert !isinplace(prob) "`Falsi` only supports OOP problems."
    f = Base.Fix2(prob.f, prob.p)
    left, right = prob.tspan
    fl, fr = f(left), f(right)

    abstol = __get_tolerance(nothing, abstol,
        promote_type(eltype(first(prob.tspan)), eltype(last(prob.tspan))))

    if iszero(fl)
        return build_solution(prob, alg, left, fl; retcode = ReturnCode.ExactSolutionLeft,
            left, right)
    end

    if iszero(fr)
        return build_solution(prob, alg, right, fr; retcode = ReturnCode.ExactSolutionRight,
            left, right)
    end

    # Regula Falsi Steps
    i = 0
    if !iszero(fr)
        while i < maxiters
            if __nextfloat_tdir(left, prob.tspan...) == right
                return build_solution(prob, alg, left, fl; left, right,
                    retcode = ReturnCode.FloatingPointLimit)
            end

            mid = (fr * left - fl * right) / (fr - fl)
            for _ in 1:10
                mid = __max_tdir(left, __prevfloat_tdir(mid, prob.tspan...), prob.tspan...)
            end

            (mid == left || mid == right) && break

            fm = f(mid)
            if abs((right - left) / 2) < abstol
                return build_solution(prob, alg, mid, fm; left, right,
                    retcode = ReturnCode.Success)
            end

            if abs(fm) < abstol
                right = mid
                break
            end

            if sign(fl) == sign(fm)
                fl, left = fm, mid
            else
                fr, right = fm, mid
            end
            i += 1
        end
    end

    sol, i, left, right, fl, fr = __bisection(left, right, fl, fr, f; abstol,
        maxiters = maxiters - i, prob, alg)
    sol !== nothing && return sol

    return SciMLBase.build_solution(prob, alg, left, fl; retcode = ReturnCode.MaxIters,
        left, right)
end
