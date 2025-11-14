"""
    Ridder()

A non-allocating ridder method.
"""
struct Ridder <: AbstractBracketingAlgorithm end

function SciMLBase.__solve(
        prob::IntervalNonlinearProblem, alg::Ridder, args...;
        maxiters = 1000, abstol = nothing, verbose::NonlinearVerbosity = NonlinearVerbosity(), kwargs...
)
    @assert !SciMLBase.isinplace(prob) "`Ridder` only supports out-of-place problems."

    f = Base.Fix2(prob.f, prob.p)
    left, right = prob.tspan
    fl, fr = f(left), f(right)

    abstol = NonlinearSolveBase.get_tolerance(
        left, abstol, promote_type(eltype(left), eltype(right))
    )

    if iszero(fl)
        return SciMLBase.build_solution(
            prob, alg, left, fl; retcode = ReturnCode.ExactSolutionLeft, left, right
        )
    end

    if iszero(fr)
        return SciMLBase.build_solution(
            prob, alg, right, fr; retcode = ReturnCode.ExactSolutionRight, left, right
        )
    end

    if sign(fl) == sign(fr)
        @SciMLMessage("The interval is not an enclosing interval, opposite signs at the \
        boundaries are required.",
            verbose, :non_enclosing_interval)
        return SciMLBase.build_solution(
            prob, alg, left, fl; retcode = ReturnCode.InitialFailure, left, right
        )
    end

    xo = oftype(left, Inf)
    i = 1
    while i ≤ maxiters
        mid = (left + right) / 2

        if mid == left || mid == right
            return SciMLBase.build_solution(
                prob, alg, left, fl; retcode = ReturnCode.FloatingPointLimit, left, right
            )
        end

        fm = f(mid)
        s = sqrt(fm^2 - fl * fr)
        if iszero(s)
            return SciMLBase.build_solution(
                prob, alg, left, fl; retcode = ReturnCode.Failure, left, right
            )
        end

        x = mid + (mid - left) * sign(fl - fm) * fm / s
        fx = f(x)
        xo = x
        if abs((right - left) / 2) < abstol
            return SciMLBase.build_solution(
                prob, alg, mid, fm; retcode = ReturnCode.Success, left, right
            )
        end

        if iszero(fx)
            right, fr = x, fx
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

    sol, i, left, right,
    fl, fr = Impl.bisection(
        left, right, fl, fr, f, abstol, maxiters - i, prob, alg
    )

    sol !== nothing && return sol

    return SciMLBase.build_solution(
        prob, alg, left, fl; retcode = ReturnCode.MaxIters, left, right
    )
end
