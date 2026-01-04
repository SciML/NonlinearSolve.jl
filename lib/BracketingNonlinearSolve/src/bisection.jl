"""
    Bisection(; exact_left = false, exact_right = false)

A common bisection method.

### Keyword Arguments

  - `exact_left`: whether to enforce whether the left side of the interval must be exactly
    zero for the returned result. Defaults to false.
  - `exact_right`: whether to enforce whether the right side of the interval must be exactly
    zero for the returned result. Defaults to false.

!!! danger "Keyword Arguments"

    Currently, the keyword arguments are not implemented.
"""
@kwdef struct Bisection <: AbstractBracketingAlgorithm
    exact_left::Bool = false
    exact_right::Bool = false
end

function SciMLBase.__solve(
        prob::IntervalNonlinearProblem, alg::Bisection, args...;
        maxiters = 1000, abstol = nothing, verbose::NonlinearVerbosity = NonlinearVerbosity(), kwargs...
    )
    @assert !SciMLBase.isinplace(prob) "`Bisection` only supports out-of-place problems."

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
        @SciMLMessage(
            "The interval is not an enclosing interval, opposite signs at the \
        boundaries are required.",
            verbose, :non_enclosing_interval
        )
        return build_bracketing_solution(prob, alg, left, fl, left, right, ReturnCode.InitialFailure)
    end

    return internal_bisection(f, left, right, fl, fr, abstol, maxiters, prob, alg)
end

# Bisection main loop is implemented in separate function so that it can be reused
# as a fallback solver in other solvers
function internal_bisection(f::F, left, right, fl, fr, abstol, maxiters, prob, alg) where {F}
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

        if abs((right - left) / 2) < abstol
            return build_bracketing_solution(prob, alg, mid, fm, left, right, ReturnCode.Success)
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

    return build_bracketing_solution(prob, alg, left, fl, left, right, ReturnCode.MaxIters)
end
