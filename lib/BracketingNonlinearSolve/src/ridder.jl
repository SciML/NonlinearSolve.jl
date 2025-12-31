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
        return build_exact_solution(prob, alg, left, fl, ReturnCode.ExactSolutionLeft)
    end

    if iszero(fr)
        return build_exact_solution(prob, alg, right, fr, ReturnCode.ExactSolutionRight)
    end

    if sign(fl) == sign(fr)
        @SciMLMessage("The interval is not an enclosing interval, opposite signs at the \
        boundaries are required.",
            verbose, :non_enclosing_interval)
        return build_bracketing_solution(prob, alg, left, fl, left, right, ReturnCode.InitialFailure)
    end

    i = 1
    while i ≤ maxiters
        mid = (left + right) / 2

        if mid == left || mid == right
            return build_bracketing_solution(prob, alg, left, fl, left, right, ReturnCode.FloatingPointLimit)
        end

        fm = f(mid)
        if iszero(fm)
            return build_exact_solution(prob, alg, mid, fm, ReturnCode.Success)
        end

        s = sqrt(fm^2 - fl * fr)
        if iszero(s)
            return build_bracketing_solution(prob, alg, left, fl, left, right, ReturnCode.Failure)
        end

        x = mid + (mid - left) * sign(fl - fm) * fm / s
        fx = f(x)
        if iszero(fx)
            return build_exact_solution(prob, alg, x, fx, ReturnCode.Success)
        end

        if abs((right - left) / 2) < abstol
            return build_bracketing_solution(prob, alg, mid, fm, left, right, ReturnCode.Success)
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

    return build_bracketing_solution(prob, alg, left, fl, left, right, ReturnCode.MaxIters)
end
